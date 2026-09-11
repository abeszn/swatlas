"""Pure indicator maths on OHLC frames. No MetaTrader dependency, so this is
importable and testable without a terminal."""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def atr(frame: pd.DataFrame, period: int) -> pd.Series:
    """Wilder's Average True Range."""
    high, low, close = frame["high"], frame["low"], frame["close"]
    prev_close = close.shift(1)

    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    # Wilder smoothing is an EMA with alpha = 1/period.
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
