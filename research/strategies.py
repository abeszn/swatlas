"""Candidate signal generators for research.

Every function takes an OHLC frame and returns a Series of 'long'/'short'/'flat'
aligned to it: the DESIRED POSITION on each bar, not an entry event. Warm-up bars
must be 'flat'.

`random_entry` is not filler. It is the null hypothesis: same sizing, same stops,
same costs, coin-flip direction. A strategy that cannot beat it has no directional
edge and is only expressing the sizing rules and the sample's drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from swatlas.indicators import atr, sma
from swatlas.signals import donchian, ma_cross   # canonical, shared with live
from swatlas.smc import smc_signal

FLAT, LONG, SHORT = "flat", "long", "short"


def _blank(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(FLAT, index=frame.index, dtype=object)


def _apply_warmup(signal: pd.Series, *guards: pd.Series) -> pd.Series:
    """Force 'flat' wherever any guard series is NaN."""
    invalid = pd.concat(guards, axis=1).isna().any(axis=1)
    return signal.mask(invalid, FLAT)


# --------------------------------------------------------------------- trend
#
# donchian and ma_cross are imported from swatlas.signals rather than defined
# here, so research and live trading provably run the same code.


def momentum(frame: pd.DataFrame, lookback: int = 60, threshold_pct: float = 0.0,
             long_only: bool = False) -> pd.Series:
    """Time-series momentum: hold in the direction of the trailing return."""
    roc = frame["close"].pct_change(lookback) * 100
    signal = _blank(frame)
    signal[roc > threshold_pct] = LONG
    if not long_only:
        signal[roc < -threshold_pct] = SHORT
    return _apply_warmup(signal, roc)


# ----------------------------------------------------------- mean reversion

def bollinger_fade(frame: pd.DataFrame, period: int = 20, sd: float = 2.0,
                   trend_filter: int = 0) -> pd.Series:
    """Fade stretched moves back toward the mean, optionally only with the trend."""
    mid = sma(frame["close"], period)
    std = frame["close"].rolling(period).std()
    upper, lower = mid + sd * std, mid - sd * std

    signal = _blank(frame)
    long_ok = frame["close"] < lower
    short_ok = frame["close"] > upper

    if trend_filter:
        trend = sma(frame["close"], trend_filter)
        long_ok &= frame["close"] > trend
        short_ok &= frame["close"] < trend
        signal[long_ok] = LONG
        signal[short_ok] = SHORT
        return _apply_warmup(signal, mid, std, trend)

    signal[long_ok] = LONG
    signal[short_ok] = SHORT
    return _apply_warmup(signal, mid, std)


def rsi_fade(frame: pd.DataFrame, period: int = 14, low: float = 30,
             high: float = 70, trend_filter: int = 0) -> pd.Series:
    delta = frame["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    signal = _blank(frame)
    long_ok, short_ok = rsi < low, rsi > high
    if trend_filter:
        trend = sma(frame["close"], trend_filter)
        long_ok &= frame["close"] > trend
        short_ok &= frame["close"] < trend
        signal[long_ok] = LONG
        signal[short_ok] = SHORT
        return _apply_warmup(signal, rsi, trend)

    signal[long_ok] = LONG
    signal[short_ok] = SHORT
    return _apply_warmup(signal, rsi)


# ------------------------------------------------------------------ controls

def buy_hold(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(LONG, index=frame.index, dtype=object)


def random_entry(frame: pd.DataFrame, prob: float = 0.02, seed: int = 0,
                 hold: int = 20) -> pd.Series:
    """Coin-flip direction, entered at roughly `prob` per bar and held `hold` bars.

    The null hypothesis. Any real strategy must clear this by a wide margin.
    """
    rng = np.random.default_rng(seed)
    state = np.full(len(frame), FLAT, dtype=object)
    remaining, current = 0, FLAT
    for i in range(len(frame)):
        if remaining > 0:
            remaining -= 1
        elif rng.random() < prob:
            current = LONG if rng.random() < 0.5 else SHORT
            remaining = hold - 1
        else:
            current = FLAT
        state[i] = current
    return pd.Series(state, index=frame.index, dtype=object)


# ------------------------------------------------------------------ registry

REGISTRY = {
    "donchian": donchian,
    "ma_cross": ma_cross,
    "momentum": momentum,
    "bollinger_fade": bollinger_fade,
    "rsi_fade": rsi_fade,
    "smc": smc_signal,          # canonical implementation lives in swatlas.smc
    "buy_hold": buy_hold,
    "random_entry": random_entry,
}


def build(frame: pd.DataFrame, name: str, atr_period: int = 14, **kwargs) -> pd.DataFrame:
    """Return a copy of `frame` with `signal` and `atr` columns attached."""
    if name not in REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; have {sorted(REGISTRY)}")
    out = frame.copy()
    out["atr"] = atr(out, atr_period)
    out["signal"] = REGISTRY[name](frame, **kwargs)
    out.loc[out["atr"].isna(), "signal"] = FLAT
    return out
