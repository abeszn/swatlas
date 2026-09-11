"""Thin, defensive wrapper around the MetaTrader5 terminal API.

Everything that touches the terminal goes through here so the safety checks
(demo-account guard, filling-mode negotiation, stop-distance clamping) cannot be
bypassed by accident from strategy code.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import pandas as pd

from .config import Config
from .contract import Contract, resolve_symbol
from .timeutil import rates_to_frame

log = logging.getLogger("mt5")


class MT5Error(RuntimeError):
    """Raised when the terminal rejects a call or is in an unusable state."""


@dataclass(frozen=True)
class OrderResult:
    ok: bool
    retcode: int
    comment: str
    order: int = 0
    volume: float = 0.0
    price: float = 0.0


class MT5Client:
    """Owns the terminal connection for the lifetime of a `with` block."""

    def __init__(self, config: Config):
        self.config = config
        self._symbol_info = None
        self._contract: Contract | None = None

    # ------------------------------------------------------------------ setup

    def __enter__(self) -> "MT5Client":
        self.connect()
        return self

    def __exit__(self, *exc_info) -> None:
        mt5.shutdown()
        log.info("Terminal connection closed")

    def connect(self) -> None:
        creds = self.config.credentials
        kwargs = {}
        if creds.path:
            kwargs["path"] = creds.path
        if creds.login and creds.password and creds.server:
            kwargs.update(login=creds.login, password=creds.password, server=creds.server)

        if not mt5.initialize(**kwargs):
            code, msg = mt5.last_error()
            raise MT5Error(
                f"Could not connect to the MT5 terminal ({code}: {msg}). "
                "Check that the terminal is installed and running, that 'Algo Trading' "
                "is enabled, and that MT5_PATH/credentials in .env are correct."
            )

        self._guard_account()
        self._select_symbol()

    def _guard_account(self) -> None:
        """Refuse to touch a real-money account unless explicitly permitted."""
        info = mt5.account_info()
        if info is None:
            raise MT5Error("Connected, but the terminal reports no logged-in account.")

        modes = {
            mt5.ACCOUNT_TRADE_MODE_DEMO: "DEMO",
            mt5.ACCOUNT_TRADE_MODE_CONTEST: "CONTEST",
            mt5.ACCOUNT_TRADE_MODE_REAL: "REAL",
        }
        mode = modes.get(info.trade_mode, f"UNKNOWN({info.trade_mode})")

        log.info(
            "Account %s on %s | %s | balance %.2f %s | leverage 1:%d",
            info.login, info.server, mode, info.balance, info.currency, info.leverage,
        )

        if mode == "REAL" and not self.config.execution.allow_live_account:
            raise MT5Error(
                "This is a REAL-money account and execution.allow_live_account is false. "
                "Refusing to start. Log the terminal into a demo account, or flip that "
                "flag deliberately once you have proven the strategy."
            )
        if not info.trade_allowed:
            log.warning(
                "Terminal reports trading is NOT allowed for this account "
                "(check the 'Algo Trading' toggle in MT5)."
            )

    def _select_symbol(self) -> None:
        symbol = resolve_symbol(self.config.symbol, mt5)
        if symbol is None:
            raise MT5Error(
                f"Symbol {self.config.symbol!r} does not exist on this broker, even after "
                "searching for prefixed/suffixed variants (EURUSD, EURUSD.a, EURUSDm, ...). "
                "Check Market Watch for the exact name and update config.yaml."
            )
        if symbol != self.config.symbol:
            log.info("Configured symbol %r resolved to broker's %r",
                     self.config.symbol, symbol)
            self.config = dataclasses.replace(self.config, symbol=symbol)

        info = mt5.symbol_info(symbol)
        if not info.visible and not mt5.symbol_select(symbol, True):
            raise MT5Error(f"Could not add {symbol!r} to Market Watch.")

        self._symbol_info = mt5.symbol_info(symbol)
        # Never trust the reported tick value - it is wrong on some symbols and
        # feeds position sizing directly. See contract.py.
        self._contract = Contract.verified(self._symbol_info, mt5)

        log.info(
            "Symbol %s | digits %d | lots %.2f-%.2f step %.2f | spread %d pts | stops level %d pts",
            symbol, self._symbol_info.digits, self._symbol_info.volume_min,
            self._symbol_info.volume_max, self._symbol_info.volume_step,
            self._symbol_info.spread, self._symbol_info.trade_stops_level,
        )
        log.info("Contract %s", self._contract.sanity_report())

    def use_symbol(self, symbol: str) -> None:
        """Point this client at a different symbol on the same open connection.

        Lets a multi-symbol scan reuse one connection instead of opening and
        tearing down a terminal link per instrument.
        """
        self.config = dataclasses.replace(self.config, symbol=symbol)
        self._select_symbol()

    @property
    def symbol_info(self):
        if self._symbol_info is None:
            raise MT5Error("Symbol info unavailable - connect() has not run.")
        return self._symbol_info

    @property
    def contract(self) -> Contract:
        if self._contract is None:
            raise MT5Error("Contract unavailable - connect() has not run.")
        return self._contract

    def refresh_symbol_info(self) -> None:
        """Re-read symbol metadata; spread and stops level move during the session."""
        self._symbol_info = mt5.symbol_info(self.config.symbol)

    # ------------------------------------------------------------------- data

    def account_balance(self) -> float:
        info = mt5.account_info()
        if info is None:
            raise MT5Error("Lost the account connection.")
        return float(info.balance)

    def bars(self, count: int | None = None) -> pd.DataFrame:
        """Most recent completed bars, oldest first, indexed by bar open time (UTC)."""
        count = count or self.config.bars_needed
        # +1 because index 0 is the still-forming bar, which we drop below.
        rates = mt5.copy_rates_from_pos(
            self.config.symbol, self.config.timeframe, 0, count + 1
        )
        if rates is None or len(rates) < 2:
            code, msg = mt5.last_error()
            raise MT5Error(f"No price history for {self.config.symbol} ({code}: {msg}).")

        # rates_to_frame converts server time to true UTC - see timeutil.py.
        frame = rates_to_frame(rates)
        return frame.iloc[:-1]  # drop the incomplete current bar

    def open_positions(self) -> list:
        """Positions on our symbol opened by this bot (matched on magic number)."""
        positions = mt5.positions_get(symbol=self.config.symbol) or ()
        return [p for p in positions if p.magic == self.config.magic]

    def realised_pnl_today(self) -> float:
        """Sum of profit+commission+swap on this bot's deals closed since UTC midnight."""
        now = datetime.now(timezone.utc)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        # Widen the window by a day either side; the terminal filters on server time,
        # which is not UTC, so a tight window silently drops deals.
        deals = mt5.history_deals_get(start - timedelta(days=1), now + timedelta(days=1))
        if deals is None:
            return 0.0

        total = 0.0
        for deal in deals:
            if deal.magic != self.config.magic or deal.symbol != self.config.symbol:
                continue
            if datetime.fromtimestamp(deal.time, tz=timezone.utc) < start:
                continue
            total += deal.profit + deal.commission + deal.swap
        return total

    # -------------------------------------------------------------- execution

    def _filling_mode(self) -> int:
        """Pick a filling mode the broker actually accepts for this symbol."""
        allowed = self.symbol_info.filling_mode
        if allowed & 1:  # SYMBOL_FILLING_FOK
            return mt5.ORDER_FILLING_FOK
        if allowed & 2:  # SYMBOL_FILLING_IOC
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN

    def _clamp_stops(self, side: str, price: float, sl: float, tp: float) -> tuple[float, float]:
        """Push SL/TP out to the broker's minimum stop distance if they are too tight."""
        info = self.symbol_info
        min_distance = info.trade_stops_level * info.point
        if min_distance <= 0:
            return round(sl, info.digits), round(tp, info.digits)

        if side == "buy":
            sl = min(sl, price - min_distance)
            tp = max(tp, price + min_distance)
        else:
            sl = max(sl, price + min_distance)
            tp = min(tp, price - min_distance)
        return round(sl, info.digits), round(tp, info.digits)

    def market_order(self, side: str, volume: float, sl: float, tp: float,
                     comment: str = "swatlas") -> OrderResult:
        """Send a market buy or sell. Honours execution.dry_run."""
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

        self.refresh_symbol_info()
        tick = mt5.symbol_info_tick(self.config.symbol)
        if tick is None:
            raise MT5Error(f"No live tick for {self.config.symbol}; market may be closed.")

        price = tick.ask if side == "buy" else tick.bid
        sl, tp = self._clamp_stops(side, price, sl, tp)

        if self.config.execution.dry_run:
            log.info(
                "[DRY RUN] would %s %.2f lots %s @ %.5f sl=%.5f tp=%.5f",
                side.upper(), volume, self.config.symbol, price, sl, tp,
            )
            return OrderResult(True, 0, "dry-run", volume=volume, price=price)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.config.symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.config.execution.deviation_points,
            "magic": self.config.magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(),
        }

        result = mt5.order_send(request)
        if result is None:
            code, msg = mt5.last_error()
            raise MT5Error(f"order_send returned nothing ({code}: {msg}).")

        ok = result.retcode == mt5.TRADE_RETCODE_DONE
        level = log.info if ok else log.error
        level(
            "%s %s %.2f lots @ %.5f sl=%.5f tp=%.5f -> retcode %d (%s)",
            "FILLED" if ok else "REJECTED", side.upper(), volume, price, sl, tp,
            result.retcode, result.comment,
        )
        return OrderResult(ok, result.retcode, result.comment,
                           result.order, result.volume, result.price)

    def close_position(self, position, comment: str = "swatlas-close") -> OrderResult:
        """Close an open position at market."""
        self.refresh_symbol_info()
        tick = mt5.symbol_info_tick(position.symbol)
        if tick is None:
            raise MT5Error(f"No live tick for {position.symbol}; cannot close.")

        is_long = position.type == mt5.POSITION_TYPE_BUY
        close_type = mt5.ORDER_TYPE_SELL if is_long else mt5.ORDER_TYPE_BUY
        price = tick.bid if is_long else tick.ask

        if self.config.execution.dry_run:
            log.info("[DRY RUN] would close #%d (%.2f lots) @ %.5f",
                     position.ticket, position.volume, price)
            return OrderResult(True, 0, "dry-run", volume=position.volume, price=price)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": close_type,
            "position": position.ticket,
            "price": price,
            "deviation": self.config.execution.deviation_points,
            "magic": self.config.magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(),
        }

        result = mt5.order_send(request)
        if result is None:
            code, msg = mt5.last_error()
            raise MT5Error(f"order_send returned nothing ({code}: {msg}).")

        ok = result.retcode == mt5.TRADE_RETCODE_DONE
        level = log.info if ok else log.error
        level("%s close #%d -> retcode %d (%s)",
              "OK" if ok else "FAILED", position.ticket, result.retcode, result.comment)
        return OrderResult(ok, result.retcode, result.comment,
                           result.order, result.volume, result.price)
