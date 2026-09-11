"""Understand XAUUSD before designing anything against it.

Answers four questions that decide the shape of any strategy:
  1. Which timeframes survive trading costs at all?
  2. How has volatility changed across the sample? (gold roughly tripled since 2018)
  3. Does gold trend or mean-revert, and at what horizon?
  4. Are there session or day-of-week effects worth exploiting or avoiding?
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from swatlas.indicators import atr

DATA_DIR = Path(__file__).resolve().parent / "data"

# Retail gold spreads run roughly $0.15-0.40 depending on broker and session.
# MetaQuotes-Demo reports 0, which is fiction, so everything here assumes this.
ASSUMED_SPREAD_USD = 0.25
CONTRACT_SIZE = 100.0  # oz per lot -> $1 move = $100 per lot

BARS_PER_YEAR = {"M15": 96 * 252, "H1": 24 * 252, "H4": 6 * 252, "D1": 252}


def load(label: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"XAUUSD_{label}.parquet")


def log_returns(frame: pd.DataFrame) -> pd.Series:
    """Bar-to-bar log returns.

    Built via to_numpy() because np.log() on a Series is typed as returning a
    bare ndarray, which then has no .diff()/.autocorr().
    """
    return pd.Series(np.log(frame["close"].to_numpy()), index=frame.index).diff()


def when(frame: pd.DataFrame) -> pd.DatetimeIndex:
    """The index as a DatetimeIndex, so .year/.hour/.dayofweek are available."""
    return pd.DatetimeIndex(frame.index)


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def cost_analysis() -> None:
    section("1. COST DRAG BY TIMEFRAME  (round-trip spread vs typical bar range)")
    print(f"assuming ${ASSUMED_SPREAD_USD:.2f}/oz spread, {CONTRACT_SIZE:.0f} oz/lot\n")
    print(f"{'TF':<5}{'bars':>8}{'median ATR':>13}{'ATR %':>9}"
          f"{'cost/ATR':>11}{'trades/yr @':>13}")
    print(f"{'':<5}{'':>8}{'($/oz)':>13}{'of px':>9}{'(round trip)':>11}{'20% drag':>13}")
    print("-" * 78)

    for label in ("M15", "H1", "H4", "D1"):
        frame = load(label)
        a = atr(frame, 14).dropna()
        median_atr = a.median()
        # Use recent price so the percentage reflects today's gold, not 2004's.
        atr_pct = (a / frame["close"]).median() * 100
        round_trip = ASSUMED_SPREAD_USD * 2
        cost_ratio = round_trip / median_atr
        # How many round trips per year before costs eat 20% of a 1R-per-trade budget?
        trades_for_20pct = 0.20 / cost_ratio if cost_ratio else np.inf
        print(f"{label:<5}{len(frame):>8}{median_atr:>13.2f}{atr_pct:>8.2f}%"
              f"{cost_ratio:>11.1%}{trades_for_20pct:>13.1f}")

    print("\nRead: 'cost/ATR' is what you pay in spread for one round trip, as a")
    print("fraction of one ATR. On M15 that is a large slice of the move you are")
    print("trying to capture; on H4/D1 it is noise. This alone rules out fast gold")
    print("scalping unless you have institutional spreads.")


def volatility_regimes() -> None:
    section("2. VOLATILITY BY YEAR  (H1 bars)")
    frame = load("H1")
    frame["atr"] = atr(frame, 14)
    frame["atr_pct"] = frame["atr"] / frame["close"] * 100

    yearly = frame.groupby(when(frame).year).agg(
        bars=("close", "size"),
        close_first=("close", "first"),
        close_last=("close", "last"),
        atr_usd=("atr", "median"),
        atr_pct=("atr_pct", "median"),
    )
    yearly["return_pct"] = (yearly["close_last"] / yearly["close_first"] - 1) * 100

    print(f"\n{'year':<7}{'bars':>7}{'close':>10}{'yr return':>12}"
          f"{'ATR $':>9}{'ATR %':>9}")
    print("-" * 78)
    for year, row in yearly.iterrows():
        print(f"{year:<7}{int(row['bars']):>7}{row['close_last']:>10.0f}"
              f"{row['return_pct']:>11.1f}%{row['atr_usd']:>9.2f}{row['atr_pct']:>8.3f}%")

    print("\nRead: gold roughly tripled over the sample, so ATR in DOLLARS is not")
    print("comparable across years - ATR as a PERCENT is. Any stop, target, or")
    print("position size must be volatility-scaled or the strategy silently changes")
    print("risk over time. Note how few down years there are: a long-biased trend")
    print("system will look brilliant on this sample for reasons that are not skill.")


def trend_vs_reversion() -> None:
    section("3. TREND OR MEAN REVERSION?  (autocorrelation of forward returns)")
    print("\nIf returns are positively autocorrelated the market trends and breakout")
    print("logic has an edge. Negative means it reverts and fading extremes pays.\n")

    print(f"{'TF':<5}{'lag 1':>9}{'lag 2':>9}{'lag 4':>9}"
          f"{'lag 8':>9}{'lag 24':>9}{'verdict':>16}")
    print("-" * 78)
    for label in ("H1", "H4", "D1"):
        frame = load(label)
        returns = log_returns(frame).dropna()
        lags = [1, 2, 4, 8, 24]
        acs = [returns.autocorr(lag=l) for l in lags]
        mean_short = np.mean(acs[:3])
        verdict = ("trending" if mean_short > 0.02
                   else "reverting" if mean_short < -0.02 else "~random")
        print(f"{label:<5}" + "".join(f"{a:>9.3f}" for a in acs) + f"{verdict:>16}")

    section("3b. VARIANCE RATIO  (H1) - >1 trends, <1 reverts, 1 = random walk")
    frame = load("H1")
    returns = log_returns(frame).dropna()
    print(f"\n{'horizon (bars)':<18}{'variance ratio':>16}")
    print("-" * 78)
    for q in (2, 4, 8, 24, 48, 120):
        var_1 = returns.var()
        var_q = returns.rolling(q).sum().dropna().var()
        vr = var_q / (q * var_1)
        print(f"{q:<18}{vr:>16.3f}")
    print("\nRead: values meaningfully above 1.0 at longer horizons say moves persist,")
    print("which favours trend following over a matching holding period.")


def session_effects() -> None:
    section("4. SESSION AND DAY EFFECTS  (H1, UTC)")
    frame = load("H1")
    frame["ret"] = log_returns(frame)
    frame["range_pct"] = (frame["high"] - frame["low"]) / frame["close"] * 100
    frame["abs_ret"] = frame["ret"].abs()

    hourly = frame.groupby(when(frame).hour).agg(
        range_pct=("range_pct", "mean"),
        mean_ret_bp=("ret", lambda s: s.mean() * 10_000),
        n=("ret", "size"),
    )
    peak = hourly["range_pct"].max()

    print("\nhour  avg range %   mean ret (bp)   activity")
    print("-" * 78)
    for raw_hour, row in hourly.iterrows():
        hour = cast(int, raw_hour)  # iterrows keys are typed Hashable, not int
        bar = "#" * int(round(row["range_pct"] / peak * 42))
        tag = ""
        if 7 <= hour <= 11:
            tag = " London"
        elif 13 <= hour <= 17:
            tag = " NY/overlap"
        elif hour <= 5:
            tag = " Asia"
        print(f"{hour:>4}{row['range_pct']:>13.3f}{row['mean_ret_bp']:>16.2f}   "
              f"{bar}{tag}")

    print("\nday-of-week (H1 aggregated):")
    dow = frame.groupby(when(frame).dayofweek).agg(
        range_pct=("range_pct", "mean"), mean_ret_bp=("ret", lambda s: s.mean() * 10_000))
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for raw_day, row in dow.iterrows():
        print(f"  {names[cast(int, raw_day)]}  range {row['range_pct']:.3f}%   "
              f"mean ret {row['mean_ret_bp']:+.2f} bp")

    print("\nRead: the Asian session is quiet - trading breakouts there mostly pays")
    print("spread for noise. London and the NY overlap carry the real movement.")


def main() -> int:
    if not (DATA_DIR / "XAUUSD_H1.parquet").exists():
        print("No cached data. Run research/fetch_data.py first.", file=sys.stderr)
        return 1

    cost_analysis()
    volatility_regimes()
    trend_vs_reversion()
    session_effects()
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
