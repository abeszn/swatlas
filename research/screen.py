"""Broad screen: do any of the candidate families show an edge on gold?

Runs every candidate on the IN-SAMPLE period only, three ways:
  - zero cost   : does the raw signal contain any directional information?
  - retail cost : does that information survive the spread and swap?
  - vs controls : does it beat coin-flip entries and buy-and-hold?

The out-of-sample period is deliberately untouched here. Look at it once, at the
end, after the design is frozen - every extra peek spends its credibility.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

import strategies
from swatlas.backtest import BacktestParams, run_backtest
from swatlas.contract import Contract
from swatlas.costs import Costs

DATA_DIR = Path(__file__).resolve().parent / "data"

IN_SAMPLE = ("2004-01-01", "2016-12-31")
OUT_SAMPLE = ("2017-01-01", "2026-12-31")

BARS_PER_YEAR = {"M15": 96 * 252, "H1": 24 * 252, "H4": 6 * 252, "D1": 252}


def load(label: str) -> tuple[pd.DataFrame, Contract]:
    frame = pd.read_parquet(DATA_DIR / f"XAUUSD_{label}.parquet")
    contract = Contract.from_dict(
        json.loads((DATA_DIR / "XAUUSD_contract.json").read_text()))
    return frame, contract


def slice_period(frame: pd.DataFrame, period: tuple[str, str]) -> pd.DataFrame:
    start, end = period
    return frame.loc[start:end]


# Exit style matters as much as entry. Trend systems must let winners run - a
# fixed target caps exactly the trades they depend on - so they get a trailing
# stop and no target. Mean-reversion systems want the opposite.
TREND_EXITS: dict[str, Any] = dict(stop_atr_mult=3.0, target_atr_mult=None,
                                   trail_atr_mult=3.0, exit_on_flat=True)
REVERSION_EXITS: dict[str, Any] = dict(stop_atr_mult=2.0, target_atr_mult=2.0,
                                       exit_on_flat=True)
NEUTRAL_EXITS: dict[str, Any] = dict(stop_atr_mult=2.0, target_atr_mult=3.0)

# name -> (strategy, kwargs, backtest params)
CANDIDATES: dict[str, tuple[str, dict, dict]] = {
    "donchian 20/10":        ("donchian", dict(entry=20, exit_n=10), TREND_EXITS),
    "donchian 55/20":        ("donchian", dict(entry=55, exit_n=20), TREND_EXITS),
    "donchian 55/20 +trend": ("donchian", dict(entry=55, exit_n=20, trend_filter=200),
                              TREND_EXITS),
    "donchian 55/20 target":  ("donchian", dict(entry=55, exit_n=20), NEUTRAL_EXITS),
    "ma 20/50":              ("ma_cross", dict(fast=20, slow=50), TREND_EXITS),
    "ma 50/200":             ("ma_cross", dict(fast=50, slow=200), TREND_EXITS),
    "ma 50/200 buffered":    ("ma_cross", dict(fast=50, slow=200, buffer_atr=0.5), TREND_EXITS),
    "momentum 60":           ("momentum", dict(lookback=60), TREND_EXITS),
    "momentum 250":          ("momentum", dict(lookback=250), TREND_EXITS),
    "momentum 250 long-only": ("momentum", dict(lookback=250, long_only=True), TREND_EXITS),
    "bollinger fade 20/2":   ("bollinger_fade", dict(period=20, sd=2.0), REVERSION_EXITS),
    "bollinger fade 20/2.5": ("bollinger_fade", dict(period=20, sd=2.5), REVERSION_EXITS),
    "rsi fade 14 30/70":     ("rsi_fade", dict(period=14, low=30, high=70), REVERSION_EXITS),
    "rsi fade +trend200":    ("rsi_fade", dict(period=14, low=35, high=65, trend_filter=200),
                              REVERSION_EXITS),
    # SMC entries are sparse triggers, not a continuously-held bias, so they use
    # stop/target exits - exit_on_flat would close them on the very next bar.
    "smc OB+FVG":            ("smc", dict(), NEUTRAL_EXITS),
    "smc OB+FVG +equilib":   ("smc", dict(require_equilibrium=True), NEUTRAL_EXITS),
    "smc FVG only":          ("smc", dict(use_order_blocks=False), NEUTRAL_EXITS),
    "smc OB only":           ("smc", dict(use_fvg=False), NEUTRAL_EXITS),
    "smc wide swings 5/5":   ("smc", dict(swing_left=5, swing_right=5), NEUTRAL_EXITS),
    "smc big FVG (1.0 ATR)": ("smc", dict(min_fvg_atr=1.0), NEUTRAL_EXITS),
    "smc trail exits":       ("smc", dict(), dict(stop_atr_mult=2.0,
                                                  target_atr_mult=None,
                                                  trail_atr_mult=2.5)),
    "-- buy & hold":         ("buy_hold", {}, dict(stop_atr_mult=50.0, target_atr_mult=None)),
    "-- random seed 0":      ("random_entry", dict(seed=0), NEUTRAL_EXITS),
    "-- random seed 1":      ("random_entry", dict(seed=1), NEUTRAL_EXITS),
    "-- random seed 2":      ("random_entry", dict(seed=2), NEUTRAL_EXITS),
}


def evaluate(frame: pd.DataFrame, contract: Contract, name: str, kwargs: dict,
             overrides: dict, costs: Costs, bars_per_year: int) -> dict:
    signalled = strategies.build(frame, name, **kwargs)
    # Annotated Any: this dict mixes floats, ints, bools and None before being
    # splatted into BacktestParams, so a narrower inferred type is wrong.
    settings: dict[str, Any] = dict(
        risk_per_trade_pct=0.5,
        starting_balance=10_000.0,
        bars_per_year=bars_per_year,
    )
    settings.update(overrides)
    return run_backtest(signalled, contract, costs, BacktestParams(**settings)).summary()


def main() -> int:
    timeframe = sys.argv[1] if len(sys.argv) > 1 else "H4"
    frame, contract = load(timeframe)
    bars_per_year = BARS_PER_YEAR[timeframe]

    train = slice_period(frame, IN_SAMPLE)
    print(f"\nXAUUSD {timeframe} | IN-SAMPLE {train.index[0]:%Y-%m-%d} -> "
          f"{train.index[-1]:%Y-%m-%d} | {len(train)} bars")
    print(contract.sanity_report())
    print(f"buy & hold over this period: "
          f"{(train['close'].iloc[-1] / train['close'].iloc[0] - 1) * 100:+.1f}%")

    retail = Costs.xauusd_retail()
    free = Costs.free()
    print(f"retail costs: spread ${retail.spread:.2f} + slippage ${retail.slippage:.2f}, "
          f"financing {retail.swap_long_annual_pct:.1f}%/yr long, "
          f"{retail.swap_short_annual_pct:.1f}%/yr short\n")

    header = (f"{'candidate':<24}{'trades':>7}{'ZERO-COST':>12}{'RETAIL':>10}"
              f"{'exp R':>8}{'PF':>7}{'maxDD':>8}{'sharpe':>8}")
    print(header)
    print(f"{'':<24}{'':>7}{'CAGR %':>12}{'CAGR %':>10}{'(net)':>8}{'':>7}{'%':>8}{'':>8}")
    print("-" * len(header))

    results = {}
    for label, (name, kwargs, overrides) in CANDIDATES.items():
        try:
            gross = evaluate(train, contract, name, kwargs, overrides, free, bars_per_year)
            net = evaluate(train, contract, name, kwargs, overrides, retail, bars_per_year)
        except Exception as exc:
            print(f"{label:<24}  ERROR: {exc}")
            continue

        results[label] = net
        if net.get("trades", 0) == 0:
            print(f"{label:<24}{0:>7}   no trades")
            continue

        print(f"{label:<24}{net['trades']:>7}{gross.get('cagr_pct', 0):>12.2f}"
              f"{net['cagr_pct']:>10.2f}{net['expectancy_R']:>8.3f}"
              f"{net['profit_factor']:>7.2f}{net['max_dd_pct']:>8.1f}"
              f"{net['sharpe']:>8.2f}")

    print("\nHow to read this:")
    print("  ZERO-COST vs RETAIL - the gap is what the broker takes. A candidate that")
    print("    is only profitable at zero cost is not a strategy.")
    print("  expectancy R - average profit per trade in units of risked capital.")
    print("    Below ~0.05 net, realistic execution noise will erase it.")
    print("  Compare everything against the random-entry controls. Beating buy & hold")
    print("    on a sample where gold rose is easy; beating random entries is the")
    print("    test of whether the SIGNAL does anything.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
