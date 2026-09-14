"""Symbol scanning and trade proposal generation.

Lives in the library rather than in the CLI script so the command line and the
dashboard produce IDENTICAL proposals. A scanner defined twice is a scanner
that will eventually disagree with itself, and you would find out by taking a
trade the other half would have rejected.

For each symbol the questions are asked in order, stopping at the first no:

  1. Is the market open?   (a live tick exists)
  2. Is there a signal?    (the configured strategy on the last closed bar)
  3. Can it be sized?      (min lot must fit inside the risk budget)
  4. Is it clear of news?  (no high-impact release in the blackout window)

Survivors then go through correlation-aware portfolio selection, which weighs
them against each other AND against positions already open.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import MetaTrader5 as mt5
import pandas as pd

from .config import Config
from .contract import Contract, resolve_symbol
from .news import NewsCalendar
from .portfolio import (
    Exposure, PortfolioLimits, Selection, correlation_matrix, select,
)
from .risk import TradePlan, build_plan
from .strategy import Signal, latest_signal
from .timeutil import rates_to_frame

log = logging.getLogger("scanner")

DEFAULT_BASKET = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
    "EURJPY", "GBPJPY", "EURGBP", "XAUUSD", "XAGUSD",
]


@dataclass
class Scan:
    symbol: str
    status: str
    signal: Signal | None = None
    atr: float | None = None
    spread_points: int | None = None
    plan: TradePlan | None = None
    held: str | None = None
    min_lot_risk_pct: float | None = None
    news: str | None = None
    digits: int = 5
    price: float | None = None
    closes: pd.Series | None = None

    @property
    def actionable(self) -> bool:
        return self.plan is not None and self.held is None and self.news is None

    def to_dict(self) -> dict:
        """JSON-safe view. `closes` is deliberately dropped - it is a price
        series used for correlation, not something the UI needs."""
        return {
            "symbol": self.symbol, "status": self.status,
            "signal": self.signal.value if self.signal else None,
            "atr": self.atr, "spread_points": self.spread_points,
            "held": self.held, "news": self.news, "digits": self.digits,
            "price": self.price,
            "min_lot_risk_pct": (round(self.min_lot_risk_pct, 2)
                                 if self.min_lot_risk_pct is not None else None),
            "actionable": self.actionable,
            "plan": None if self.plan is None else {
                "side": self.plan.side, "volume": self.plan.volume,
                "stop_loss": self.plan.stop_loss,
                "take_profit": self.plan.take_profit,
                "risk_amount": round(self.plan.risk_amount, 2),
            },
        }


@dataclass
class ScanResult:
    scans: list[Scan] = field(default_factory=list)
    proposals: list[Scan] = field(default_factory=list)
    rejected: list[tuple[str, str, str]] = field(default_factory=list)  # symbol, side, reason
    existing: list[Exposure] = field(default_factory=list)
    selection: Selection | None = None
    balance: float = 0.0

    def to_dict(self) -> dict:
        s = self.selection
        return {
            "scans": [x.to_dict() for x in self.scans],
            "proposals": [x.to_dict() for x in self.proposals],
            "rejected": [{"symbol": a, "side": b, "reason": c}
                         for a, b, c in self.rejected],
            "existing": [{"symbol": e.symbol,
                          "side": "LONG" if e.direction > 0 else "SHORT",
                          "risk": round(e.risk, 2)} for e in self.existing],
            "portfolio": None if s is None else {
                "naive_risk": round(s.naive_risk, 2),
                "combined_risk": round(s.combined_risk, 2),
                "naive_pct": round(s.naive_risk / self.balance * 100, 2)
                             if self.balance else 0.0,
                "combined_pct": round(s.combined_risk / self.balance * 100, 2)
                                if self.balance else 0.0,
                "diversification": round(s.diversification_ratio, 2),
                "currency_risk": {k: round(v, 2) for k, v in s.currency_risk.items()
                                  if abs(v) > 0.01},
            },
        }


def scan_symbol(symbol: str, config: Config,
                calendar: NewsCalendar | None = None) -> Scan:
    # DEFAULT_BASKET is written in plain pair names; brokers routinely wrap
    # them (EURUSDm, EURUSD.a, ...), so resolve to whatever this account's
    # broker actually calls it before doing anything else. Everything below
    # - and what gets sent back on Approve - uses the resolved name.
    resolved = resolve_symbol(symbol, mt5)
    if resolved is None:
        return Scan(symbol, "not offered by this broker")
    symbol = resolved

    info = mt5.symbol_info(symbol)
    if info is None:
        return Scan(symbol, "not offered by this broker")
    if not info.visible:
        mt5.symbol_select(symbol, True)
        info = mt5.symbol_info(symbol)

    tick = mt5.symbol_info_tick(symbol)
    if tick is None or tick.ask <= 0:
        return Scan(symbol, "market closed", digits=info.digits)

    contract = Contract.verified(info, mt5)
    spread_points = round((tick.ask - tick.bid) / contract.point)

    rates = mt5.copy_rates_from_pos(symbol, config.timeframe, 0,
                                    config.bars_needed + 1)
    if rates is None or len(rates) < config.bars_needed:
        return Scan(symbol, "insufficient history", spread_points=spread_points,
                    digits=contract.digits)

    frame = rates_to_frame(rates).iloc[:-1]
    closes = frame["close"]

    symbol_config = dataclasses.replace(config, symbol=symbol)
    signal, atr_value = latest_signal(frame, symbol_config)
    atr_value = float(atr_value)

    existing = [p for p in (mt5.positions_get(symbol=symbol) or ())
                if p.magic == config.magic]
    held = None
    if existing:
        held = "LONG" if existing[0].type == mt5.POSITION_TYPE_BUY else "SHORT"

    account = mt5.account_info()
    balance = account.balance if account else 0.0
    stop_distance = atr_value * config.risk.stop_loss_atr_mult
    min_lot_risk_pct = (contract.volume_min * contract.loss_per_lot(stop_distance)
                        / balance * 100) if stop_distance > 0 and balance else None

    common = dict(spread_points=spread_points, held=held,
                  min_lot_risk_pct=min_lot_risk_pct, digits=contract.digits,
                  closes=closes)

    if signal is Signal.FLAT:
        return Scan(symbol, "no signal", signal, atr_value, **common)

    price = tick.ask if signal is Signal.LONG else tick.bid
    plan = build_plan(signal, atr_value, price, balance, contract, symbol_config)

    if plan is None:
        return Scan(symbol, "too big to size", signal, atr_value,
                    price=price, **common)
    if held:
        return Scan(symbol, f"already {held}", signal, atr_value, plan=plan,
                    price=price, **common)

    if calendar is not None:
        event = calendar.blackout(
            symbol, datetime.now(timezone.utc), config.news.before_minutes,
            config.news.after_minutes, config.news.min_impact)
        if event is not None:
            return Scan(symbol, "news blackout", signal, atr_value, plan=plan,
                        news=f"{event.currency} {event.title}"[:40],
                        price=price, **common)

    return Scan(symbol, "TRADEABLE", signal, atr_value, plan=plan,
                price=price, **common)


def open_exposures(config: Config) -> list[Exposure]:
    """Positions already open, expressed as portfolio exposures."""
    result = []
    for p in (mt5.positions_get() or ()):
        direction = 1 if p.type == mt5.POSITION_TYPE_BUY else -1
        stop_distance = abs(p.price_open - p.sl) if p.sl else 0.0
        info = mt5.symbol_info(p.symbol)
        risk = (Contract.verified(info, mt5).money(stop_distance, p.volume)
                if info is not None and stop_distance > 0 else 0.0)
        result.append(Exposure(p.symbol, direction, risk, "open"))
    return result


def run_scan(config: Config, symbols: list[str] | None = None,
             calendar: NewsCalendar | None = None,
             limits: PortfolioLimits | None = None) -> ScanResult:
    """Scan the basket and return correlation-filtered proposals.

    DEFAULT_BASKET holds broker-neutral names; the caller's explicit `symbols`
    (if given) is assumed to already be broker-resolved, since a caller passing
    its own list has presumably already decided what to ask the terminal for.
    """
    symbols = symbols or [s + config.broker_suffix for s in DEFAULT_BASKET]
    account = mt5.account_info()
    balance = account.balance if account else 0.0

    scans: list[Scan] = []
    for symbol in symbols:
        try:
            scans.append(scan_symbol(symbol, config, calendar))
        except Exception as exc:                     # one bad symbol must not
            log.warning("scan failed for %s: %s", symbol, exc)   # kill the scan
            scans.append(Scan(symbol, f"error: {exc}"))

    result = ScanResult(scans=scans, balance=balance)
    actionable = [s for s in scans if s.actionable]
    if not actionable:
        return result

    closes = {s.symbol: s.closes for s in scans if s.closes is not None}
    corr = correlation_matrix(closes)
    result.existing = open_exposures(config)

    candidates = [
        Exposure(s.symbol, 1 if s.plan.side == "buy" else -1, s.plan.risk_amount)
        for s in actionable if s.plan is not None
    ]
    selection = select(candidates, result.existing, corr, balance,
                       limits or PortfolioLimits())
    result.selection = selection
    chosen = {c.symbol for c in selection.chosen}
    result.proposals = [s for s in actionable if s.symbol in chosen]
    result.rejected = [
        (e.symbol, "LONG" if e.direction > 0 else "SHORT", reason)
        for e, reason in selection.rejected
    ]
    return result
