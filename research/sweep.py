"""Parameter robustness sweep, in-sample only.

The question is NOT "which parameters made the most money". It is "is the
profitable region a broad plateau or an isolated spike?" A spike is noise you
have fitted; a plateau is an effect that tolerates being slightly wrong, which
is the only kind that survives contact with the future.

    .venv\\Scripts\\python.exe research\\sweep.py donchian H4
    .venv\\Scripts\\python.exe research\\sweep.py momentum H4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

import strategies
from screen import BARS_PER_YEAR, IN_SAMPLE, slice_period
from swatlas.backtest import BacktestParams, run_backtest
from swatlas.contract import Contract
from swatlas.costs import Costs

DATA_DIR = Path(__file__).resolve().parent / "data"

TREND_EXITS: dict[str, Any] = dict(stop_atr_mult=3.0, target_atr_mult=None,
                                   trail_atr_mult=3.0, exit_on_flat=True)
# SMC fires sparse entry triggers rather than holding a bias, so it must not
# exit on FLAT - that would close on the bar after entry.
SMC_EXITS: dict[str, Any] = dict(stop_atr_mult=2.0, target_atr_mult=None,
                                 trail_atr_mult=2.5, exit_on_flat=False)

GRIDS = {
    "donchian": {
        "entry": [10, 15, 20, 30, 40, 55, 70, 100],
        "exit_n": [5, 10, 15, 20, 30],
    },
    "momentum": {
        "lookback": [20, 40, 60, 100, 150, 250, 400],
        "threshold_pct": [0.0, 1.0, 2.0, 4.0],
    },
    "smc": {
        "swing": [2, 3, 4, 5, 6, 8],
        "min_fvg_atr": [0.0, 0.25, 0.5, 1.0],
    },
}

EXITS_FOR = {"donchian": TREND_EXITS, "momentum": TREND_EXITS, "smc": SMC_EXITS}


def load(label: str) -> tuple[pd.DataFrame, Contract]:
    frame = pd.read_parquet(DATA_DIR / f"XAUUSD_{label}.parquet")
    contract = Contract.from_dict(
        json.loads((DATA_DIR / "XAUUSD_contract.json").read_text()))
    return frame, contract


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "donchian"
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "H4"
    if name not in GRIDS:
        print(f"no grid for {name!r}; have {sorted(GRIDS)}", file=sys.stderr)
        return 1

    frame, contract = load(timeframe)
    train = slice_period(frame, IN_SAMPLE)
    costs = Costs.xauusd_retail()
    bars_per_year = BARS_PER_YEAR[timeframe]

    grid = GRIDS[name]
    exits = EXITS_FOR[name]
    (key_a, values_a), (key_b, values_b) = grid.items()

    print(f"\n{name} on XAUUSD {timeframe}, IN-SAMPLE "
          f"{train.index[0]:%Y-%m}..{train.index[-1]:%Y-%m}, retail costs")
    print(f"exits: stop {exits['stop_atr_mult']}xATR, "
          f"trailing {exits['trail_atr_mult']}xATR, no fixed target\n")

    records = []
    for a in values_a:
        for b in values_b:
            kwargs = {key_a: a, key_b: b}
            try:
                signalled = strategies.build(train, name, **kwargs)
                params = BacktestParams(
                    risk_per_trade_pct=0.5, starting_balance=10_000.0,
                    bars_per_year=bars_per_year, **exits)
                summary = run_backtest(signalled, contract, costs, params).summary()
            except Exception as exc:
                print(f"  {kwargs}: ERROR {exc}")
                continue
            if summary.get("trades", 0) < 30:
                continue
            records.append({key_a: a, key_b: b, **summary})

    if not records:
        print("no parameter set produced enough trades")
        return 1

    table = pd.DataFrame(records)

    for metric, label in (("expectancy_R", "EXPECTANCY (R per trade)"),
                          ("cagr_pct", "CAGR % (net of retail costs)"),
                          ("sharpe", "SHARPE")):
        pivot = table.pivot(index=key_a, columns=key_b, values=metric)
        print(f"\n{label}")
        print(f"{'':<8}" + "".join(f"{c:>9}" for c in pivot.columns) + f"   <- {key_b}")
        for idx, row in pivot.iterrows():
            cells = "".join(
                f"{v:>9.3f}" if metric == "expectancy_R" else f"{v:>9.2f}"
                for v in row
            )
            print(f"{idx:<8}{cells}")
        print(f"^ {key_a}")

    positive = (table["expectancy_R"] > 0).mean() * 100
    print(f"\nfraction of the grid with positive expectancy: {positive:.0f}%")
    print(f"median expectancy across the grid: {table['expectancy_R'].median():.3f} R")
    print(f"best cell: {table.loc[table['expectancy_R'].idxmax(), [key_a, key_b]].to_dict()} "
          f"at {table['expectancy_R'].max():.3f} R")

    if positive > 70:
        print("\n=> Broad plateau. The effect is not an artefact of one lucky setting.")
    elif positive > 40:
        print("\n=> Mixed. Some real signal, but parameter choice matters more than it should.")
    else:
        print("\n=> Mostly negative. The few good cells are almost certainly noise.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
