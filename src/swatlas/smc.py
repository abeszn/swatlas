"""Smart Money Concepts primitives: market structure, BOS/CHoCH, fair value gaps,
order blocks, and premium/discount equilibrium.

These ideas are usually taught discretionarily, which makes them impossible to
test. Everything here is a precise mechanical definition, stated in the
docstrings, so a backtest measures something reproducible. Where practitioners
disagree the choice is documented and exposed as a parameter.

LOOKAHEAD IS THE MAIN HAZARD HERE. A swing high is only *knowable* `right` bars
after it prints, because you must see the bars that failed to exceed it. Naive
SMC backtests mark swings with a centred window and then trade them on the bar
they formed, which quietly reads the future and makes any structure strategy
look extraordinary. Every function below returns values already lagged to the
bar on which the information genuinely existed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BULLISH, BEARISH = "bullish", "bearish"


@dataclass
class Zone:
    """A price region of interest - an order block or a fair value gap."""
    kind: str            # "ob" | "fvg"
    direction: str       # BULLISH (expect support) | BEARISH (expect resistance)
    top: float
    bottom: float
    created_at: int      # bar index where the zone became known
    mitigated_at: int | None = None

    @property
    def height(self) -> float:
        return self.top - self.bottom

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top

    def touched(self, low: float, high: float) -> bool:
        """True if a bar's range overlaps the zone at all."""
        return not (high < self.bottom or low > self.top)


# ---------------------------------------------------------------- swing points

def swing_points(frame: pd.DataFrame, left: int = 2, right: int = 2
                 ) -> tuple[pd.Series, pd.Series]:
    """Fractal swing highs and lows.

    A swing high at bar i requires high[i] to be strictly greater than the
    `left` highs before it and >= the `right` highs after it.

    Returns two Series aligned to the frame holding the swing PRICE at the bar
    where it became CONFIRMED (i.e. shifted forward by `right`), NaN elsewhere.
    Using the confirmation bar rather than the formation bar is what keeps this
    honest - at bar i you cannot yet know bar i is a swing.
    """
    high, low = frame["high"].to_numpy(), frame["low"].to_numpy()
    n = len(frame)
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)

    for i in range(left, n - right):
        window_left_h = high[i - left:i]
        window_right_h = high[i + 1:i + 1 + right]
        if high[i] > window_left_h.max() and high[i] >= window_right_h.max():
            swing_high[i + right] = high[i]      # known only `right` bars later

        window_left_l = low[i - left:i]
        window_right_l = low[i + 1:i + 1 + right]
        if low[i] < window_left_l.min() and low[i] <= window_right_l.min():
            swing_low[i + right] = low[i]

    return (pd.Series(swing_high, index=frame.index),
            pd.Series(swing_low, index=frame.index))


# ------------------------------------------------------------ market structure

def market_structure(frame: pd.DataFrame, left: int = 2, right: int = 2
                     ) -> pd.DataFrame:
    """Track trend via Break of Structure and Change of Character.

    * BOS   - close breaks the most recent confirmed swing in the SAME direction
              as the prevailing trend. Continuation.
    * CHoCH - close breaks the most recent confirmed swing AGAINST the
              prevailing trend. The first such break is the reversal signal.

    Closes are used rather than wicks: a wick through a level and back is the
    textbook liquidity sweep, not a structural break. This is the stricter and
    more common convention.

    Returns a frame with columns: trend, event, ref_high, ref_low, range_high,
    range_low - all valid as of that bar, no lookahead.
    """
    swing_high, swing_low = swing_points(frame, left, right)
    close = frame["close"].to_numpy()
    sh, sl = swing_high.to_numpy(), swing_low.to_numpy()
    n = len(frame)

    trend = np.zeros(n, dtype=int)          # +1 up, -1 down, 0 undecided
    event = np.array([""] * n, dtype=object)
    ref_high = np.full(n, np.nan)           # live level whose break = bullish BOS
    ref_low = np.full(n, np.nan)
    # The dealing range: the swing pair that defines premium/discount.
    range_high = np.full(n, np.nan)
    range_low = np.full(n, np.nan)

    current_trend = 0
    last_sh = np.nan
    last_sl = np.nan
    range_hi = np.nan
    range_lo = np.nan

    for i in range(n):
        # Absorb any swing confirmed on this bar BEFORE testing for a break.
        if not np.isnan(sh[i]):
            last_sh = sh[i]
        if not np.isnan(sl[i]):
            last_sl = sl[i]

        broke_up = not np.isnan(last_sh) and close[i] > last_sh
        broke_down = not np.isnan(last_sl) and close[i] < last_sl

        if broke_up:
            event[i] = "bos" if current_trend >= 0 else "choch"
            range_lo = last_sl if not np.isnan(last_sl) else range_lo
            range_hi = close[i]
            current_trend = 1
            last_sh = np.nan          # consumed; wait for the next swing high
        elif broke_down:
            event[i] = "bos" if current_trend <= 0 else "choch"
            range_hi = last_sh if not np.isnan(last_sh) else range_hi
            range_lo = close[i]
            current_trend = -1
            last_sl = np.nan
        else:
            # Extend the live range with confirmed swings so equilibrium tracks.
            if not np.isnan(last_sh):
                range_hi = np.nanmax([range_hi, last_sh])
            if not np.isnan(last_sl):
                range_lo = np.nanmin([range_lo, last_sl])

        trend[i] = current_trend
        ref_high[i] = last_sh
        ref_low[i] = last_sl
        range_high[i] = range_hi
        range_low[i] = range_lo

    return pd.DataFrame({
        "trend": trend, "event": event,
        "ref_high": ref_high, "ref_low": ref_low,
        "range_high": range_high, "range_low": range_low,
    }, index=frame.index)


# --------------------------------------------------------- fair value gaps

def fair_value_gaps(frame: pd.DataFrame, min_size: float | pd.Series = 0.0
                    ) -> list[Zone]:
    """Three-candle imbalances.

    Bullish FVG: low[i] > high[i-2]. The untraded band [high[i-2], low[i]].
    Bearish FVG: high[i] < low[i-2]. The untraded band [high[i], low[i-2]].

    The gap is only known once candle i closes, so `created_at` is i - no
    lookahead. `min_size` filters noise gaps; pass an ATR series to scale it
    with volatility, which matters on a sample where gold tripled.
    """
    high, low = frame["high"].to_numpy(), frame["low"].to_numpy()
    n = len(frame)
    floor = (min_size.to_numpy() if isinstance(min_size, pd.Series)
             else np.full(n, float(min_size)))

    zones: list[Zone] = []
    for i in range(2, n):
        threshold = floor[i] if not np.isnan(floor[i]) else 0.0
        if low[i] > high[i - 2] and (low[i] - high[i - 2]) >= threshold:
            zones.append(Zone("fvg", BULLISH, top=low[i], bottom=high[i - 2],
                              created_at=i))
        elif high[i] < low[i - 2] and (low[i - 2] - high[i]) >= threshold:
            zones.append(Zone("fvg", BEARISH, top=low[i - 2], bottom=high[i],
                              created_at=i))
    return zones


# ------------------------------------------------------------- order blocks

def order_blocks(frame: pd.DataFrame, structure: pd.DataFrame,
                 max_lookback: int = 20, body_only: bool = False) -> list[Zone]:
    """The last opposing candle before the move that broke structure.

    On a bullish BOS at bar i, walk back up to `max_lookback` bars to the most
    recent bearish candle (close < open); that candle is the bullish order
    block. Mirror for bearish.

    `body_only` uses [open, close] instead of [low, high]. Practitioners
    disagree; the wick-inclusive default is the more conservative choice because
    it produces a wider zone that price must reach, so it triggers less often.

    `created_at` is the BOS bar, not the order-block candle - the block is only
    identifiable once the break happens.
    """
    open_, close = frame["open"].to_numpy(), frame["close"].to_numpy()
    high, low = frame["high"].to_numpy(), frame["low"].to_numpy()
    events = structure["event"].to_numpy()
    n = len(frame)

    zones: list[Zone] = []
    for i in range(n):
        if events[i] not in ("bos", "choch"):
            continue
        bullish_break = structure["trend"].to_numpy()[i] > 0

        for j in range(i - 1, max(-1, i - max_lookback - 1), -1):
            is_opposing = (close[j] < open_[j]) if bullish_break else (close[j] > open_[j])
            if not is_opposing:
                continue
            if body_only:
                top, bottom = max(open_[j], close[j]), min(open_[j], close[j])
            else:
                top, bottom = high[j], low[j]
            zones.append(Zone("ob", BULLISH if bullish_break else BEARISH,
                              top=top, bottom=bottom, created_at=i))
            break
    return zones


# ---------------------------------------------------------------- equilibrium

def equilibrium_position(price: float, range_low: float, range_high: float
                         ) -> float | None:
    """Where price sits in the dealing range: 0.0 = range low, 1.0 = range high.

    Below 0.5 is discount (where longs are considered good value), above 0.5 is
    premium. Exactly 0.5 is equilibrium. Returns None if the range is undefined
    or degenerate.
    """
    if np.isnan(range_low) or np.isnan(range_high):
        return None
    span = range_high - range_low
    if span <= 0:
        return None
    return (price - range_low) / span


# ------------------------------------------------------------------- strategy

def smc_signal(frame: pd.DataFrame,
               swing: int | None = None,
               swing_left: int = 2,
               swing_right: int = 2,
               use_order_blocks: bool = True,
               use_fvg: bool = True,
               require_equilibrium: bool = False,
               equilibrium_tolerance: float = 0.0,
               min_fvg_atr: float = 0.25,
               max_zone_age: int = 50,
               atr_period: int = 14,
               body_only_ob: bool = False) -> pd.Series:
    """The standard SMC entry model, made mechanical.

    1. Structure sets the bias (BOS/CHoCH via `market_structure`).
    2. Price must retrace into a Point Of Interest aligned with that bias -
       an order block or an unmitigated fair value gap.
    3. Optionally the POI must sit on the right side of equilibrium: discount
       for longs, premium for shorts. This is the `require_equilibrium` switch -
       off by default so its contribution can be measured rather than assumed.

    Emits the bias only on bars where a POI is actually being tested; FLAT
    otherwise. The backtester holds an open position through FLAT, so this reads
    as "enter on the retracement, then manage by stop and target".

    A zone is consumed once touched, so one order block produces one entry, not
    an entry on every bar price spends inside it.
    """
    from .indicators import atr as _atr

    if swing is not None:          # convenience for sweeps: one knob, both sides
        swing_left = swing_right = swing

    structure = market_structure(frame, swing_left, swing_right)
    atr_series = _atr(frame, atr_period)

    zones: list[Zone] = []
    if use_fvg:
        zones += fair_value_gaps(frame, min_size=atr_series * min_fvg_atr)
    if use_order_blocks:
        zones += order_blocks(frame, structure, body_only=body_only_ob)

    # Bucket by creation bar so the sweep below stays linear.
    by_bar: dict[int, list[Zone]] = {}
    for zone in zones:
        by_bar.setdefault(zone.created_at, []).append(zone)

    trend = structure["trend"].to_numpy()
    range_high = structure["range_high"].to_numpy()
    range_low = structure["range_low"].to_numpy()
    high, low, close = (frame["high"].to_numpy(), frame["low"].to_numpy(),
                        frame["close"].to_numpy())
    n = len(frame)

    signal = np.array(["flat"] * n, dtype=object)
    active: list[Zone] = []

    for i in range(n):
        # Zones created on earlier bars only - a zone born on bar i is not
        # tradeable on bar i.
        active = [z for z in active
                  if z.mitigated_at is None and (i - z.created_at) <= max_zone_age]

        bias = trend[i]
        if bias != 0 and active:
            wanted = BULLISH if bias > 0 else BEARISH
            for zone in active:
                if zone.direction != wanted or not zone.touched(low[i], high[i]):
                    continue

                if require_equilibrium:
                    where = equilibrium_position(close[i], range_low[i], range_high[i])
                    if where is None:
                        continue
                    if wanted == BULLISH and where > 0.5 + equilibrium_tolerance:
                        continue          # too expensive; wait for discount
                    if wanted == BEARISH and where < 0.5 - equilibrium_tolerance:
                        continue          # too cheap to sell

                signal[i] = "long" if wanted == BULLISH else "short"
                zone.mitigated_at = i     # consumed
                break

        for zone in by_bar.get(i, ()):    # becomes available from bar i+1
            active.append(zone)

    warm = max(swing_left + swing_right, atr_period)
    signal[:warm] = "flat"
    return pd.Series(signal, index=frame.index, dtype=object)
