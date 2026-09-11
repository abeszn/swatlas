"""Backtest the configured live strategy on history pulled from the terminal.

    .venv\\Scripts\\python.exe scripts\\backtest.py --bars 20000
    .venv\\Scripts\\python.exe scripts\\backtest.py --bars 20000 --spread 0.25 --csv out.csv

For strategy research use the research/ scripts instead - they run on cached
history with proper in-sample/out-of-sample separation.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import logging
import sys

import pandas as pd

from swatlas.backtest import BacktestParams, run_backtest
from swatlas.config import load_config
from swatlas.costs import Costs
from swatlas.logsetup import setup_logging
from swatlas.mt5_client import MT5Client, MT5Error
from swatlas.strategy import generate_signals

BARS_PER_YEAR = {"M1": 1440 * 252, "M5": 288 * 252, "M15": 96 * 252,
                 "M30": 48 * 252, "H1": 24 * 252, "H4": 6 * 252, "D1": 252}


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest the live strategy.")
    parser.add_argument("--bars", type=int, default=10_000)
    parser.add_argument("--balance", type=float, default=10_000.0)
    parser.add_argument("--spread", type=float, default=None,
                        help="spread in PRICE units (e.g. 0.25 for gold, 0.00012 for FX). "
                             "Defaults to the symbol's live spread, which on a demo "
                             "server is often an unrealistic zero.")
    parser.add_argument("--swap-long", type=float, default=-5.0,
                        help="long financing, %% of notional per year")
    parser.add_argument("--swap-short", type=float, default=-1.0)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    setup_logging(logging.WARNING)
    config = load_config()

    try:
        with MT5Client(config) as client:
            frame = client.bars(args.bars)
            contract = client.contract
            live_spread = client.symbol_info.spread * contract.point
    except MT5Error as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    spread = args.spread if args.spread is not None else live_spread
    if spread <= 0:
        print("WARNING: spread is zero. Demo servers report fictional spreads; "
              "results below assume free trading and are not meaningful. "
              "Pass --spread with a realistic value.\n")

    costs = Costs(spread=spread, slippage=0.0,
                  swap_long_annual_pct=args.swap_long,
                  swap_short_annual_pct=args.swap_short)
    params = BacktestParams(
        risk_per_trade_pct=config.risk.risk_per_trade_pct,
        stop_atr_mult=config.risk.stop_loss_atr_mult,
        target_atr_mult=config.risk.take_profit_atr_mult,
        min_lot=config.risk.min_lot,
        max_lot=config.risk.max_lot,
        starting_balance=args.balance,
        bars_per_year=BARS_PER_YEAR.get(config.timeframe_name, 252),
    )

    result = run_backtest(generate_signals(frame, config), contract, costs, params)

    print(f"\n{config.symbol} {config.timeframe_name} | {len(frame)} bars | "
          f"{frame.index[0]:%Y-%m-%d} -> {frame.index[-1]:%Y-%m-%d}")
    print(contract.sanity_report())
    print(f"costs: spread {spread:g}, financing {args.swap_long}%/{args.swap_short}% per year\n")

    summary = result.summary()
    if summary.get("trades", 0) == 0:
        print("No trades were taken over this period.")
        return 0

    width = max(len(k) for k in summary)
    for key, value in summary.items():
        print(f"  {key.replace('_', ' '):<{width}}  {value}")

    yearly = result.yearly()
    if not yearly.empty:
        print(f"\nper year:\n{yearly.to_string()}")

    print("\nNo slippage, commission, requotes, or gap-through-stop is modelled. "
          "A positive result is a reason to forward-test on demo, not to go live.\n")

    if args.csv:
        pd.DataFrame([vars(t) for t in result.trades]).to_csv(args.csv, index=False)
        print(f"Trades written to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
