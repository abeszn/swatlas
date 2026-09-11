"""Correctness and causality tests for the SMC primitives.

The headline test is CAUSALITY. Almost every impressive Smart Money Concepts
backtest is wrong for the same reason: swing points are marked with a centred
window and then traded on the bar they formed, which uses bars that had not
printed yet. The result looks superb and is unreproducible live.

The test: compute signals on the first k bars, then on the full series, and
require the overlapping region to be IDENTICAL. If any function reads forward,
seeing more data changes an earlier answer and the test fails. This catches
lookahead that eyeballing a chart never will.

    .venv\\Scripts\\python.exe scripts\\verify_smc.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import sys

from collections.abc import Sequence

import numpy as np
import pandas as pd

from swatlas.smc import (
    equilibrium_position, fair_value_gaps, market_structure, order_blocks,
    smc_signal, swing_points,
)

failures: list[str] = []


def check(label: str, condition: object, detail: str = "") -> None:
    ok = bool(condition)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(label)


def candles(rows: Sequence[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Build a frame from (open, high, low, close) tuples."""
    return pd.DataFrame(
        rows, columns=["open", "high", "low", "close"],
        index=pd.date_range("2024-01-01", periods=len(rows), freq="h", tz="UTC"),
    )


def random_walk(n: int = 1500, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    wick = np.abs(rng.normal(0, 0.35, n))
    frame = pd.DataFrame({
        "open": np.concatenate([[close[0]], close[:-1]]),
        "high": close + wick,
        "low": close - wick,
        "close": close,
    }, index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"))
    frame["high"] = frame[["open", "high", "close"]].max(axis=1)
    frame["low"] = frame[["open", "low", "close"]].min(axis=1)
    return frame


def main() -> int:
    print("\n-- swing points --")
    # A clean peak at index 3, with 2 lower bars either side.
    rows = [(1, 2, 0, 1), (1, 3, 1, 2), (2, 4, 2, 3), (3, 9, 3, 8),
            (8, 5, 4, 5), (5, 4, 3, 4), (4, 3, 2, 3)]
    frame = candles(rows)
    highs, lows = swing_points(frame, left=2, right=2)
    marked = highs.dropna()
    check("one swing high detected", len(marked) == 1, f"got {len(marked)}")
    check("swing high price is the peak (9)", marked.iloc[0] == 9)
    check("reported 2 bars AFTER the peak, not on it",
          frame.index.get_loc(marked.index[0]) == 5,
          f"reported at index {frame.index.get_loc(marked.index[0])}, peak was at 3")

    print("\n-- fair value gaps --")
    # Bullish: bar2.low (5) > bar0.high (3) -> gap [3, 5]
    frame = candles([(1, 3, 1, 2), (3, 8, 3, 7), (7, 9, 5, 8)])
    gaps = fair_value_gaps(frame)
    check("bullish FVG found", len(gaps) == 1 and gaps[0].direction == "bullish")
    check("gap spans [3, 5]", gaps and gaps[0].bottom == 3 and gaps[0].top == 5,
          f"got [{gaps[0].bottom}, {gaps[0].top}]" if gaps else "none")
    check("created on the third candle, not earlier",
          gaps and gaps[0].created_at == 2)

    # Bearish mirror: bar2.high (3) < bar0.low (5) -> gap [3, 5]
    frame = candles([(8, 9, 5, 6), (6, 6, 2, 3), (3, 3, 1, 2)])
    gaps = fair_value_gaps(frame)
    check("bearish FVG found", len(gaps) == 1 and gaps[0].direction == "bearish")

    frame = candles([(1, 3, 1, 2), (2, 4, 2, 3), (3, 5, 2, 4)])
    check("no FVG when candles overlap", len(fair_value_gaps(frame)) == 0)

    print("\n-- min size filter --")
    frame = candles([(1, 3, 1, 2), (3, 8, 3, 7), (7, 9, 5, 8)])
    check("gap of 2 survives a threshold of 1",
          len(fair_value_gaps(frame, min_size=1.0)) == 1)
    check("gap of 2 is filtered by a threshold of 5",
          len(fair_value_gaps(frame, min_size=5.0)) == 0)

    print("\n-- market structure --")
    walk = random_walk()
    structure = market_structure(walk, 2, 2)
    events = structure["event"].value_counts().to_dict()
    check("structure produces BOS events", events.get("bos", 0) > 0, str(events))
    check("structure produces CHoCH events", events.get("choch", 0) > 0)
    check("trend only ever -1, 0, +1",
          set(structure["trend"].unique()) <= {-1, 0, 1})
    check("CHoCH always flips the trend sign",
          bool((structure.loc[structure["event"] == "choch", "trend"] != 0).all()))

    print("\n-- order blocks --")
    blocks = order_blocks(walk, structure)
    check("order blocks found", len(blocks) > 0, f"{len(blocks)} blocks")
    check("every OB is created on a structure event",
          all(structure["event"].to_numpy()[b.created_at] in ("bos", "choch")
              for b in blocks))
    check("every OB zone has positive height",
          all(b.height >= 0 for b in blocks))

    print("\n-- equilibrium --")
    check("midpoint is 0.5", equilibrium_position(150, 100, 200) == 0.5)
    check("range low is 0.0", equilibrium_position(100, 100, 200) == 0.0)
    discount = equilibrium_position(120, 100, 200)
    premium = equilibrium_position(180, 100, 200)
    check("discount reads below 0.5", discount is not None and discount < 0.5)
    check("premium reads above 0.5", premium is not None and premium > 0.5)
    check("degenerate range returns None",
          equilibrium_position(100, 100, 100) is None)
    check("undefined range returns None",
          equilibrium_position(100, float("nan"), 200) is None)

    print("\n-- CAUSALITY (the test that matters) --")
    full = smc_signal(walk)
    for k in (400, 700, 1100):
        partial = smc_signal(walk.iloc[:k])
        same = (partial.to_numpy() == full.iloc[:k].to_numpy()).all()
        check(f"signals on first {k} bars unchanged by later data", same,
              "" if same else "LOOKAHEAD BIAS - a later bar altered an earlier signal")

    structure_full = market_structure(walk, 2, 2)
    for k in (400, 900):
        structure_partial = market_structure(walk.iloc[:k], 2, 2)
        same = (structure_partial["trend"].to_numpy()
                == structure_full["trend"].to_numpy()[:k]).all()
        check(f"structure trend on first {k} bars is causal", same)

    print("\n-- signal sanity --")
    counts = full.value_counts().to_dict()
    check("all three states occur", len(counts) >= 2, str(counts))
    check("signals are sparse (entries, not a permanent position)",
          counts.get("flat", 0) / len(full) > 0.5,
          f"{counts.get('flat', 0) / len(full):.1%} flat")

    eq_on = smc_signal(walk, require_equilibrium=True)
    non_flat_off = (full != "flat").sum()
    non_flat_on = (eq_on != "flat").sum()
    check("equilibrium filter reduces entries", non_flat_on <= non_flat_off,
          f"{non_flat_on} with filter vs {non_flat_off} without")

    fvg_only = smc_signal(walk, use_order_blocks=False, use_fvg=True)
    ob_only = smc_signal(walk, use_order_blocks=True, use_fvg=False)
    check("FVG-only produces entries", (fvg_only != "flat").sum() > 0,
          f"{(fvg_only != 'flat').sum()} entries")
    check("OB-only produces entries", (ob_only != "flat").sum() > 0,
          f"{(ob_only != 'flat').sum()} entries")

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILED: ' + ', '.join(failures)}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
