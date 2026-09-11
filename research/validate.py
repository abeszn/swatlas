"""Out-of-sample validation of a FROZEN design.

    .venv\\Scripts\\python.exe research\\validate.py donchian
    .venv\\Scripts\\python.exe research\\validate.py smc

Each design below was chosen from in-sample data only (2004-2016) and is not
touched again. Everything here runs on 2017-2026.

A STATISTICAL CAVEAT WORTH STATING PLAINLY: the out-of-sample period is now
being used to judge more than one hypothesis. Each additional design tested
against the same held-out data spends a little of its credibility, because the
best of N designs beats the best of 1 by luck alone. Two is still defensible;
if this list grows to ten, the out-of-sample result stops meaning much and the
honest move is to hold back a fresh period.

Parameters come from the MIDDLE of each in-sample plateau, not its peak.
Donchian's best cell was 100/20 at 0.074 R; SMC's was swing 4 / 0.25 ATR at
0.044 R. Picking peaks is how you fit noise.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

import strategies
from screen import BARS_PER_YEAR, IN_SAMPLE, OUT_SAMPLE, slice_period
from swatlas.backtest import BacktestParams, run_backtest
from swatlas.contract import Contract
from swatlas.costs import Costs

DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class Design:
    name: str
    strategy: str
    kwargs: dict[str, Any]
    exits: dict[str, Any]
    timeframe: str = "H4"
    risk_pct: float = 0.5
    note: str = ""


DESIGNS: dict[str, Design] = {
    "donchian": Design(
        name="Donchian 20/10 breakout",
        strategy="donchian",
        kwargs=dict(entry=20, exit_n=10),
        exits=dict(stop_atr_mult=3.0, target_atr_mult=None,
                   trail_atr_mult=3.0, exit_on_flat=True),
        note="exit on channel flip; conventional Turtle-style parameters",
    ),
    "smc": Design(
        name="SMC order block + FVG",
        strategy="smc",
        kwargs=dict(swing=3, min_fvg_atr=0.25,
                    use_order_blocks=True, use_fvg=True,
                    require_equilibrium=False),
        # Sparse entry triggers: must NOT exit on flat or it closes next bar.
        exits=dict(stop_atr_mult=2.0, target_atr_mult=None,
                   trail_atr_mult=2.5, exit_on_flat=False),
        note="structure sets bias, entry on retrace into OB/FVG, trailing exit",
    ),
}

CONTROL_EXITS = dict(stop_atr_mult=2.0, target_atr_mult=3.0)


def load(label: str) -> tuple[pd.DataFrame, Contract]:
    frame = pd.read_parquet(DATA_DIR / f"XAUUSD_{label}.parquet")
    contract = Contract.from_dict(
        json.loads((DATA_DIR / "XAUUSD_contract.json").read_text()))
    return frame, contract


def evaluate(frame, contract, costs, bars_per_year, design: Design,
             strategy: str | None = None, kwargs: dict | None = None,
             exits: dict | None = None, risk: float | None = None):
    signalled = strategies.build(frame, strategy or design.strategy,
                                 **(design.kwargs if kwargs is None else kwargs))
    params = BacktestParams(
        risk_per_trade_pct=risk if risk is not None else design.risk_pct,
        starting_balance=10_000.0, bars_per_year=bars_per_year,
        **(exits if exits is not None else design.exits))
    return run_backtest(signalled, contract, costs, params)


KEYS = ("trades", "expectancy_R", "profit_factor", "cagr_pct", "sharpe",
        "max_dd_pct", "win_rate_pct", "exposure_pct")


def header() -> None:
    labels = ("trades", "exp R", "PF", "CAGR%", "sharpe", "maxDD%", "win%", "expo%")
    print(f"  {'':<28}" + "".join(f"{k:>12}" for k in labels))
    print("  " + "-" * 124)


def show(label: str, result) -> dict:
    s = result.summary()
    if s.get("trades", 0) == 0:
        print(f"  {label:<28} no trades")
        return s
    print(f"  {label:<28}" + "".join(
        f"{s.get(k, 0):>12.3f}" if k == "expectancy_R" else f"{s.get(k, 0):>12}"
        for k in KEYS))
    return s


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a frozen design OOS.")
    parser.add_argument("design", nargs="?", default="donchian", choices=sorted(DESIGNS))
    args = parser.parse_args()
    design = DESIGNS[args.design]

    frame, contract = load(design.timeframe)
    bars_per_year = BARS_PER_YEAR[design.timeframe]
    retail = Costs.xauusd_retail()

    train = slice_period(frame, IN_SAMPLE)
    test = slice_period(frame, OUT_SAMPLE)

    print("=" * 128)
    print(f"FROZEN DESIGN: {design.name}  on XAUUSD {design.timeframe}")
    print(f"  params: {design.kwargs}")
    print(f"  exits : {design.exits}")
    print(f"  {design.note}")
    print(f"  risk  : {design.risk_pct}% per trade, compounding")
    print(f"  costs : spread ${retail.spread:.2f} + slippage ${retail.slippage:.2f}, "
          f"financing {retail.swap_long_annual_pct}%/{retail.swap_short_annual_pct}%/yr")
    print("=" * 128)

    print("\n1. IN-SAMPLE vs OUT-OF-SAMPLE")
    print(f"   in  : {train.index[0]:%Y-%m-%d} -> {train.index[-1]:%Y-%m-%d} "
          f"(gold {(train['close'].iloc[-1] / train['close'].iloc[0] - 1) * 100:+.0f}%)")
    print(f"   out : {test.index[0]:%Y-%m-%d} -> {test.index[-1]:%Y-%m-%d} "
          f"(gold {(test['close'].iloc[-1] / test['close'].iloc[0] - 1) * 100:+.0f}%)\n")
    header()
    show("IN-SAMPLE (design set)", evaluate(train, contract, retail, bars_per_year, design))
    out_res = evaluate(test, contract, retail, bars_per_year, design)
    out_summary = show("OUT-OF-SAMPLE (held out)", out_res)

    print("\n   controls on the OUT-OF-SAMPLE period:")
    for seed in (0, 1, 2):
        show(f"random entries seed {seed}",
             evaluate(test, contract, retail, bars_per_year, design,
                      strategy="random_entry", kwargs=dict(seed=seed),
                      exits=CONTROL_EXITS))
    bh = (test["close"].iloc[-1] / test["close"].iloc[0] - 1) * 100
    years = len(test) / bars_per_year
    bh_cagr = ((1 + bh / 100) ** (1 / years) - 1) * 100
    print(f"  {'buy & hold (unlevered)':<28}{'':>36}{bh_cagr:>12.2f}")

    print("\n2. PER-YEAR OUT-OF-SAMPLE  (consistent, or one lucky year?)")
    yearly = out_res.yearly()
    if not yearly.empty:
        print("   " + yearly.to_string().replace("\n", "\n   "))
        print(f"\n   profitable years: {(yearly['pnl'] > 0).sum()}/{len(yearly)}")

    print("\n3. COST SENSITIVITY (out-of-sample)")
    header()
    for label, costs in (
        ("free (no costs at all)", Costs.free()),
        ("tight ECN ($0.12 + comm)", Costs.xauusd_tight()),
        ("retail ($0.25 + $0.05 slip)", retail),
        ("wide ($0.50 spread)", Costs(spread=0.50, slippage=0.05,
                                      swap_long_annual_pct=-5.0,
                                      swap_short_annual_pct=-1.0)),
    ):
        show(label, evaluate(test, contract, costs, bars_per_year, design))

    print("\n4. RISK SCALING (out-of-sample)")
    header()
    for risk in (0.5, 1.0, 2.0):
        show(f"{risk}% risk per trade",
             evaluate(test, contract, retail, bars_per_year, design, risk=risk))

    print("\n5. IS THE EDGE DISTINGUISHABLE FROM ZERO? (out-of-sample)")
    trades = out_res.closed
    if len(trades) >= 30:
        rs = np.array([t.r_multiple for t in trades])
        n = len(rs)
        mean_r, se = rs.mean(), rs.std(ddof=1) / np.sqrt(n)
        t_stat = mean_r / se if se else 0.0
        print(f"   {n} trades, expectancy {mean_r:+.4f} R, standard error {se:.4f} R")
        print(f"   t-statistic {t_stat:+.2f}   "
              f"(|t| > 2 is weak evidence; |t| > 3 is worth acting on)")
        print(f"   95% confidence interval: "
              f"[{mean_r - 1.96 * se:+.3f}, {mean_r + 1.96 * se:+.3f}] R")

        rng = np.random.default_rng(42)
        draws = rng.choice(rs, size=(4000, n), replace=True)
        totals = draws.sum(axis=1)
        print(f"\n   bootstrap total R: median {np.median(totals):+.1f}, "
              f"5th pct {np.percentile(totals, 5):+.1f}, "
              f"95th pct {np.percentile(totals, 95):+.1f}")
        print(f"   probability of losing money over {n} trades: "
              f"{(totals < 0).mean() * 100:.1f}%")

        perms = np.array([rng.permutation(rs) for _ in range(2000)])
        equity = np.cumsum(perms, axis=1)
        padded = np.concatenate([np.zeros((2000, 1)), equity], axis=1)
        dds = (padded - np.maximum.accumulate(padded, axis=1)).min(axis=1)
        print("\n   worst drawdown if the same trades arrived in another order:")
        print(f"     median {np.median(dds):.1f} R, 5th pct {np.percentile(dds, 5):.1f} R"
              f"  (= {np.median(dds) * design.risk_pct:.1f}% / "
              f"{np.percentile(dds, 5) * design.risk_pct:.1f}% of account)")

    print("\n" + "=" * 128)
    print(f"VERDICT ({design.name}): out-of-sample expectancy "
          f"{out_summary.get('expectancy_R', 0):+.3f} R, "
          f"CAGR {out_summary.get('cagr_pct', 0):+.2f}%, "
          f"vs gold buy & hold {bh_cagr:+.2f}%")
    print("=" * 128 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
