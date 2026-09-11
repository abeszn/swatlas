"""Does the Donchian design survive on instruments a small account can trade?

Gold is where the research was done, but XAUUSD needs roughly $17,000 to size a
0.5% risk trade at the broker minimum lot. On a $1,000 account the only viable
instruments are FX majors. That makes this the question that actually decides
what the live bot should run: not "is Donchian good?" but "is Donchian good on
something I can trade?"

Same frozen parameters as research/validate.py (20/10, mid-plateau, chosen on
gold in-sample data). Transferring an unchanged design to a new instrument is a
genuine out-of-sample test - nothing here was tuned on FX.

    .venv\\Scripts\\python.exe research\\validate_fx.py
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

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
TIMEFRAMES = ["H4", "H1"]
BARS_PER_YEAR = {"H1": 24 * 252, "H4": 6 * 252, "D1": 252}

# Frozen on gold, unchanged here. Retuning per symbol is how you fit noise.
DESIGNS: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {
    "donchian 20/10": ("donchian", dict(entry=20, exit_n=10),
                       dict(stop_atr_mult=3.0, target_atr_mult=None,
                            trail_atr_mult=3.0, exit_on_flat=True)),
    "smc 3/0.25": ("smc", dict(swing=3, min_fvg_atr=0.25),
                   dict(stop_atr_mult=2.0, target_atr_mult=None,
                        trail_atr_mult=2.5, exit_on_flat=False)),
    "random": ("random_entry", dict(seed=0),
               dict(stop_atr_mult=2.0, target_atr_mult=3.0)),
}


def costs_for(symbol: str) -> Costs:
    return Costs.fx_jpy_retail() if "JPY" in symbol else Costs.fx_major_retail()


def load(symbol: str, timeframe: str):
    path = DATA_DIR / f"{symbol}_{timeframe}.parquet"
    spec = DATA_DIR / f"{symbol}_contract.json"
    if not path.exists() or not spec.exists():
        return None, None
    frame = pd.read_parquet(path)
    contract = Contract.from_dict(json.loads(spec.read_text()))
    return frame, contract


def evaluate(frame, contract, costs, bars_per_year, design):
    name, kwargs, exits = design
    signalled = strategies.build(frame, name, **kwargs)
    params = BacktestParams(risk_per_trade_pct=0.5, starting_balance=10_000.0,
                            bars_per_year=bars_per_year, **exits)
    return run_backtest(signalled, contract, costs, params)


def main() -> int:
    print("\nDonchian 20/10 and SMC 3/0.25, frozen on gold, transferred unchanged")
    print("to FX majors. Retail FX costs (~1.2 pip effective, -2%/yr financing).")

    rows = []
    for timeframe in TIMEFRAMES:
        print(f"\n{'=' * 104}\n{timeframe}\n{'=' * 104}")
        print(f"{'symbol':<9}{'design':<17}{'trades':>7}{'exp R':>9}{'PF':>7}"
              f"{'CAGR%':>9}{'sharpe':>8}{'maxDD%':>9}{'span':>22}")
        print("-" * 104)

        for symbol in SYMBOLS:
            frame, contract = load(symbol, timeframe)
            if frame is None:
                print(f"{symbol:<9}no cached data - run research/fetch_data.py {symbol}")
                continue
            costs = costs_for(symbol)
            span = f"{frame.index[0]:%Y-%m}..{frame.index[-1]:%Y-%m}"

            for label, design in DESIGNS.items():
                result = evaluate(frame, contract, costs,
                                  BARS_PER_YEAR[timeframe], design)
                s = result.summary()
                if s.get("trades", 0) < 30:
                    print(f"{symbol:<9}{label:<17}{s.get('trades', 0):>7}  too few trades")
                    continue
                print(f"{symbol:<9}{label:<17}{s['trades']:>7}{s['expectancy_R']:>9.3f}"
                      f"{s['profit_factor']:>7.2f}{s['cagr_pct']:>9.2f}"
                      f"{s['sharpe']:>8.2f}{s['max_dd_pct']:>9.1f}{span:>22}")
                rows.append({"tf": timeframe, "symbol": symbol, "design": label,
                             **{k: s[k] for k in
                                ("trades", "expectancy_R", "profit_factor",
                                 "cagr_pct", "sharpe")}})

    if not rows:
        print("\nNo data. Run: research/fetch_data.py EURUSD GBPUSD USDJPY AUDUSD")
        return 1

    table = pd.DataFrame(rows)
    print(f"\n{'=' * 104}\nSUMMARY BY DESIGN (averaged across symbols and timeframes)\n{'=' * 104}")
    grouped = table.groupby("design").agg(
        runs=("expectancy_R", "size"),
        mean_expectancy_R=("expectancy_R", "mean"),
        median_expectancy_R=("expectancy_R", "median"),
        positive=("expectancy_R", lambda s: f"{(s > 0).sum()}/{len(s)}"),
        mean_PF=("profit_factor", "mean"),
        mean_CAGR=("cagr_pct", "mean"),
    ).round(3)
    print(grouped.to_string())

    print("\nRead: compare each design against the 'random' row. A design that")
    print("does not clear random entries on the SAME data has no edge - it is")
    print("just expressing the sizing rules and whatever drift the sample had.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
