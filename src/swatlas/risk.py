"""Position sizing and the hard limits that stand between a bug and the balance.

All contract maths goes through `Contract`, whose tick value is verified against
the terminal rather than taken from the symbol info field - see contract.py for
why that distinction is not academic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import Config
from .contract import Contract
from .strategy import Signal

log = logging.getLogger("risk")


@dataclass(frozen=True)
class TradePlan:
    side: str          # "buy" or "sell"
    volume: float
    stop_loss: float
    take_profit: float
    risk_amount: float


def position_size(balance: float, stop_distance: float, contract: Contract,
                  config: Config) -> float:
    """Lots such that hitting the stop loses ~risk_per_trade_pct of balance.

    Returns 0.0 when the correctly-sized trade is smaller than the broker's
    minimum lot - taking the minimum anyway would silently exceed the risk budget.
    """
    risk_cfg = config.risk
    risk_amount = balance * risk_cfg.risk_per_trade_pct / 100.0

    if stop_distance <= 0 or contract.tick_size <= 0 or contract.tick_value <= 0:
        log.error(
            "Cannot size position: stop_distance=%s tick_size=%s tick_value=%s",
            stop_distance, contract.tick_size, contract.tick_value,
        )
        return 0.0

    loss_per_lot = contract.loss_per_lot(stop_distance)
    if loss_per_lot <= 0:
        return 0.0

    risk_based = contract.round_volume(risk_amount / loss_per_lot)

    # A fixed lot overrides risk-based sizing entirely: the user has asked to
    # trade an exact size, so risk_per_trade_pct becomes advisory. The size still
    # passes through the broker/config lot band below, and the real money risked
    # is logged - a fixed lot means each trade risks a different, uncontrolled %.
    desired = (contract.round_volume(risk_cfg.fixed_lot)
               if risk_cfg.fixed_lot > 0 else risk_based)

    # Clamp into the configured lot band. The ceiling protects the balance; the
    # floor (min_lot) guarantees a tradeable size even when risk maths comes out
    # tiny - which DELIBERATELY lets a trade risk more than risk_per_trade_pct on
    # a small stop or account. That is the point of a floor (see config.yaml).
    ceiling = min(risk_cfg.max_lot, contract.volume_max)
    floor = max(risk_cfg.min_lot, contract.volume_min)
    volume = contract.round_volume(min(max(desired, floor), ceiling))

    if volume < contract.volume_min:
        log.warning(
            "Skipping trade: the lot ceiling %.2f is below the broker minimum %.2f. "
            "Raise risk.max_lot to trade this symbol.",
            ceiling, contract.volume_min,
        )
        return 0.0

    forced_risk = volume * loss_per_lot
    if risk_cfg.fixed_lot > 0:
        log.warning(
            "Fixed lot %.2f in force (risk maths wanted %.4f). This trade risks "
            "%.2f (%.2f%% of balance); risk_per_trade_pct %.2f%% is ignored.",
            volume, risk_based, forced_risk, forced_risk / balance * 100,
            risk_cfg.risk_per_trade_pct,
        )
    elif volume > risk_based:
        log.warning(
            "Lot floor raised size to %.2f (risk maths wanted %.4f). This trade risks "
            "%.2f (%.2f%% of balance) versus your %.2f%% budget.",
            volume, risk_based, forced_risk, forced_risk / balance * 100,
            risk_cfg.risk_per_trade_pct,
        )

    log.debug("Sized %.2f lots (risk %.2f, loss/lot %.2f)", volume, risk_amount, loss_per_lot)
    return volume


def build_plan(signal: Signal, atr_value: float, price: float, balance: float,
               contract: Contract, config: Config) -> TradePlan | None:
    """Turn a signal into a concrete, sized order - or None if it should be skipped."""
    if signal is Signal.FLAT:
        return None
    if not atr_value or atr_value <= 0:
        log.warning("ATR is %s; refusing to size a trade without volatility data.", atr_value)
        return None

    risk_cfg = config.risk
    stop_distance = atr_value * risk_cfg.stop_loss_atr_mult
    target_distance = atr_value * risk_cfg.take_profit_atr_mult

    volume = position_size(balance, stop_distance, contract, config)
    if volume <= 0:
        return None

    if signal is Signal.LONG:
        side, sl, tp = "buy", price - stop_distance, price + target_distance
    else:
        side, sl, tp = "sell", price + stop_distance, price - target_distance

    return TradePlan(
        side=side,
        volume=volume,
        stop_loss=contract.round_price(sl),
        take_profit=contract.round_price(tp),
        risk_amount=contract.money(stop_distance, volume),
    )


def daily_loss_breached(realised_pnl: float, balance: float, config: Config) -> bool:
    """True once today's realised loss has eaten through the daily budget."""
    limit = balance * config.risk.max_daily_loss_pct / 100.0
    if realised_pnl <= -limit:
        log.error(
            "DAILY LOSS LIMIT HIT: realised %.2f today against a limit of -%.2f (%.1f%%). "
            "No new entries until tomorrow.",
            realised_pnl, limit, config.risk.max_daily_loss_pct,
        )
        return True
    return False
