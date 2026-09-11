"""Server-time to UTC conversion.

MT5 reports bar and tick timestamps in the BROKER'S SERVER TIME, packed into a
Unix-epoch integer. Reading that with `datetime.fromtimestamp(t, tz=utc)` is the
obvious thing to do and it is wrong: it silently relabels server time as UTC.

MetaQuotes-Demo runs at UTC+3, so every chart was three hours ahead of reality.
It went unnoticed until the dashboard grew a timezone picker and started showing
a "last closed bar" in the future.

Consequences of getting this wrong:

* Displayed times are off by the server offset.
* Any HOUR-OF-DAY analysis is shifted, so session attribution ("this is the
  London session") lands on the wrong bars.
* Anything comparing bar times against real-world UTC events - an economic
  calendar, for instance - misaligns by the same amount.

It does NOT affect strategies that only care about bar ordering: trend,
breakout and structure logic are unaffected, because a constant shift does not
change which bar follows which.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import pandas as pd

log = logging.getLogger("timeutil")

_cached: timedelta | None = None
_PROBE_SYMBOLS = ("EURUSD", "XAUUSD", "GBPUSD", "USDJPY")


def detect_server_offset(force: bool = False) -> timedelta:
    """Measure the broker's UTC offset from a live tick.

    Brokers use whole- or half-hour offsets, so the raw measurement is rounded
    to the nearest half hour - that removes tick latency without inventing an
    offset that no broker actually uses.

    Falls back to zero when no fresh tick is available (weekend, market closed),
    which is the safe direction: no shift rather than a wrong shift.
    """
    global _cached
    if _cached is not None and not force:
        return _cached

    now = datetime.now(timezone.utc)
    for symbol in _PROBE_SYMBOLS:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None or not tick.time:
            continue
        server_now = datetime.fromtimestamp(tick.time, tz=timezone.utc)
        raw_hours = (server_now - now).total_seconds() / 3600.0
        # A stale tick (closed market) would give a large bogus offset.
        if abs(raw_hours) > 14:
            continue
        rounded = round(raw_hours * 2) / 2
        _cached = timedelta(hours=rounded)
        log.info("Broker server time is UTC%+g (measured on %s)", rounded, symbol)
        return _cached

    log.warning("Could not measure the broker's server offset - no fresh tick. "
                "Treating bar times as UTC; displayed times may be shifted.")
    _cached = timedelta(0)
    return _cached


def reset_cache() -> None:
    global _cached
    _cached = None


def to_utc(frame: pd.DataFrame, offset: timedelta | None = None) -> pd.DataFrame:
    """Shift a bar frame's index from server time to true UTC."""
    shift = detect_server_offset() if offset is None else offset
    if not shift:
        return frame
    out = frame.copy()
    out.index = pd.DatetimeIndex(out.index) - shift
    return out


def rates_to_frame(rates, offset: timedelta | None = None) -> pd.DataFrame:
    """Standard MT5 rates -> DataFrame indexed by true UTC bar-open time."""
    frame = pd.DataFrame(rates)
    frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    return to_utc(frame.set_index("time"), offset)
