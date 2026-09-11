"""Live signal generation, dispatched by `strategy.name` in config.yaml.

`generate_signals` is a pure function of an OHLC frame, and both the live engine
and the backtester call it - so what you test is what trades.

Available strategies and what the research says about each (research/RESEARCH.md):

  ma_cross  - moving-average crossover. The original baseline. Break-even at
              best; kept as the default only because it is the simplest thing
              that exercises the whole pipeline.
  donchian  - channel breakout. The strongest family found: a broad in-sample
              plateau (85% of the parameter grid profitable) and it beat random
              entries both in and out of sample. Still only break-even after
              retail costs out-of-sample.
  smc       - Smart Money Concepts: structure (BOS/CHoCH), order blocks, fair
              value gaps, optional premium/discount equilibrium. Looked good
              in-sample (+0.033 R) and INVERTED out-of-sample (-0.034 R), which
              is the signature of a fitted result. Available, not recommended.

None of these has a demonstrated edge after costs. Choosing one is choosing
which break-even system to forward-test, not which one makes money.
"""

from __future__ import annotations

from enum import Enum

import pandas as pd

from .config import Config
from .indicators import atr, sma
from .signals import donchian, ma_cross
from .smc import smc_signal


class Signal(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


def generate_signals(frame: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Return `frame` with an `atr` column, a `signal` column, and whatever
    diagnostic columns the chosen strategy exposes.

    `signal` is the desired position per bar, not an entry event. The engine
    diffs it against the live position, so LONG -> SHORT closes and reverses.
    """
    strat, risk_cfg = config.strategy, config.risk
    out = frame.copy()
    out["atr"] = atr(out, risk_cfg.atr_period)

    name = strat.name
    extra = dict(strat.params)

    try:
        if name == "ma_cross":
            out["fast"] = sma(out["close"], strat.fast_period)
            out["slow"] = sma(out["close"], strat.slow_period)
            out["signal"] = ma_cross(
                frame, fast=strat.fast_period, slow=strat.slow_period,
                buffer_atr=strat.entry_buffer_atr, atr_period=risk_cfg.atr_period,
                **extra)
        elif name == "donchian":
            out["signal"] = donchian(frame, **extra)
        elif name == "smc":
            extra.setdefault("atr_period", risk_cfg.atr_period)
            out["signal"] = smc_signal(frame, **extra)
        else:
            raise ValueError(
                f"unknown strategy {name!r}; expected one of "
                "'ma_cross', 'donchian', 'smc'")
    except TypeError as exc:
        # `params` is strategy-specific. Changing strategy.name in config.yaml
        # while leaving the previous strategy's params behind produces an
        # unhelpful TypeError deep in the signal function; say what to fix.
        raise ValueError(
            f"strategy {name!r} rejected strategy.params {extra}: {exc}. "
            f"These parameters belong to a different strategy - update or clear "
            f"strategy.params in config.yaml when you change strategy.name."
        ) from exc

    # A NaN ATR means no volatility estimate, so no position can be sized.
    out.loc[out["atr"].isna(), "signal"] = Signal.FLAT.value
    return out


def latest_signal(frame: pd.DataFrame, config: Config) -> tuple[Signal, float]:
    """Signal and ATR from the most recent completed bar."""
    enriched = generate_signals(frame, config)
    last = enriched.iloc[-1]
    return Signal(last["signal"]), float(last["atr"])
