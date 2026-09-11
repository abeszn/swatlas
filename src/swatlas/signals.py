"""Canonical signal generators shared by the live engine and the research harness.

Every function returns a Series of 'long'/'short'/'flat' - the DESIRED POSITION
per bar, not an entry event. Warm-up bars must be 'flat'.

This module exists so that live trading and backtesting call the SAME code. A
strategy defined twice is a strategy that will eventually behave differently in
the two places, and you will find out with real money.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import atr, sma

FLAT, LONG, SHORT = "flat", "long", "short"


def ma_cross(frame: pd.DataFrame, fast: int = 20, slow: int = 50,
             buffer_atr: float = 0.0, atr_period: int = 14,
             long_only: bool = False) -> pd.Series:
    """Moving-average crossover with an ATR buffer.

    The buffer requires the close to clear the slow MA by `buffer_atr` ATRs
    before a side is taken, which suppresses the whipsaw that dominates a bare
    crossover when the averages sit on top of each other.
    """
    f, s = sma(frame["close"], fast), sma(frame["close"], slow)
    a = atr(frame, atr_period)
    buf = a * buffer_atr

    signal = pd.Series(FLAT, index=frame.index, dtype=object)
    signal[(f > s) & (frame["close"] > s + buf)] = LONG
    if not long_only:
        signal[(f < s) & (frame["close"] < s - buf)] = SHORT

    invalid = pd.concat([f, s, a], axis=1).isna().any(axis=1)
    return signal.mask(invalid, FLAT)


def donchian(frame: pd.DataFrame, entry: int = 20, exit_n: int = 10,
             long_only: bool = False, trend_filter: int = 0) -> pd.Series:
    """Channel breakout: enter on an `entry`-bar extreme, leave on an
    `exit_n`-bar extreme against the position.

    Every channel is shifted one bar. The breakout must be judged against a
    channel built from bars that CLOSED BEFORE the bar being acted on, or the
    breakout bar is part of its own channel and the test is circular.
    """
    high_n = frame["high"].rolling(entry).max().shift(1).to_numpy()
    low_n = frame["low"].rolling(entry).min().shift(1).to_numpy()
    exit_high = frame["high"].rolling(exit_n).max().shift(1).to_numpy()
    exit_low = frame["low"].rolling(exit_n).min().shift(1).to_numpy()
    closes = frame["close"].to_numpy()

    trend = sma(frame["close"], trend_filter).to_numpy() if trend_filter else None

    state = np.full(len(frame), FLAT, dtype=object)
    current = FLAT

    for i in range(len(frame)):
        if np.isnan(high_n[i]) or np.isnan(low_n[i]):
            state[i] = FLAT
            continue

        up_ok = down_ok = True
        if trend is not None:
            if np.isnan(trend[i]):
                state[i] = FLAT
                continue
            up_ok = closes[i] > trend[i]
            down_ok = closes[i] < trend[i]

        if current == FLAT:
            if closes[i] > high_n[i] and up_ok:
                current = LONG
            elif closes[i] < low_n[i] and down_ok and not long_only:
                current = SHORT
        elif current == LONG and closes[i] < exit_low[i]:
            current = FLAT
        elif current == SHORT and closes[i] > exit_high[i]:
            current = FLAT

        state[i] = current

    return pd.Series(state, index=frame.index, dtype=object)
