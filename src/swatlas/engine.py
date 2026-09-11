"""The live trading loop.

Design notes worth knowing before you change anything here:

* Decisions are made once per *completed* bar, never on ticks. The forming bar is
  dropped in `MT5Client.bars()`, so a signal cannot flicker and re-fire.
* FLAT means "no fresh edge", not "get out". An open position is left to its stop
  loss or take profit; only an opposite signal closes it early. Exiting on FLAT
  churns badly inside the entry buffer.
* Every entry passes the risk guards. There is no code path that opens a position
  without going through `risk.build_plan`.
"""

from __future__ import annotations

import logging
import time

import MetaTrader5 as mt5

from datetime import datetime, timezone

from . import risk
from .config import Config
from .mt5_client import MT5Client, MT5Error
from .news import NewsCalendar
from .strategy import Signal, latest_signal

log = logging.getLogger("engine")


class Engine:
    def __init__(self, client: MT5Client, config: Config):
        self.client = client
        self.config = config
        self._last_bar_time = None
        self._halted_for_day = False
        self._calendar: NewsCalendar | None = None
        if config.news.enabled:
            self._calendar = NewsCalendar.load(config.news.max_cache_age_hours)
            upcoming = self._calendar.upcoming(
                config.symbol, datetime.now(timezone.utc),
                within_minutes=24 * 60, min_impact=config.news.min_impact)
            log.info("News filter on: %d %s-impact events for %s in the next 24h",
                     len(upcoming), config.news.min_impact, config.symbol)
            for event in upcoming[:5]:
                log.info("  %s  %s  %s", event.when.strftime("%Y-%m-%d %H:%M UTC"),
                         event.currency, event.title)

    # ----------------------------------------------------------------- helpers

    def _current_side(self) -> Signal:
        """What this bot is currently holding, per the terminal."""
        positions = self.client.open_positions()
        if not positions:
            return Signal.FLAT
        if len(positions) > 1:
            log.warning("%d positions open on this magic; acting on the first.", len(positions))
        return Signal.LONG if positions[0].type == mt5.POSITION_TYPE_BUY else Signal.SHORT

    def _entry_blocked(self) -> str | None:
        """Reason a new entry must not be taken, or None if clear."""
        balance = self.client.account_balance()

        if risk.daily_loss_breached(self.client.realised_pnl_today(), balance, self.config):
            self._halted_for_day = True
            return "daily loss limit"

        if len(self.client.open_positions()) >= self.config.risk.max_open_positions:
            return "max open positions reached"

        account = mt5.account_info()
        if account is not None and not account.trade_allowed:
            return "terminal reports trading not allowed (check Algo Trading toggle)"

        # News blocks NEW entries only. Open positions keep their broker-side
        # stops - yanking them mid-release is usually worse than sitting still.
        if self._calendar is not None:
            news_cfg = self.config.news
            event = self._calendar.blackout(
                self.config.symbol, datetime.now(timezone.utc),
                news_cfg.before_minutes, news_cfg.after_minutes, news_cfg.min_impact)
            if event is not None:
                return (f"news blackout: {event.currency} {event.title} at "
                        f"{event.when:%H:%M UTC} ({event.impact} impact)")

        return None

    def _open(self, signal: Signal, atr_value: float) -> None:
        blocked = self._entry_blocked()
        if blocked:
            log.info("Entry skipped: %s", blocked)
            return

        tick = mt5.symbol_info_tick(self.config.symbol)
        if tick is None:
            log.info("Entry skipped: no live tick (market likely closed).")
            return

        self.client.refresh_symbol_info()
        price = tick.ask if signal is Signal.LONG else tick.bid
        plan = risk.build_plan(
            signal, atr_value, price, self.client.account_balance(),
            self.client.contract, self.config,
        )
        if plan is None:
            return

        log.info(
            "Entry %s %.2f lots | sl %.5f tp %.5f | risking %.2f (%.2f%%)",
            plan.side.upper(), plan.volume, plan.stop_loss, plan.take_profit,
            plan.risk_amount, self.config.risk.risk_per_trade_pct,
        )
        self.client.market_order(plan.side, plan.volume, plan.stop_loss, plan.take_profit)

    def _close_all(self, reason: str) -> None:
        for position in self.client.open_positions():
            log.info("Closing #%d (%s)", position.ticket, reason)
            self.client.close_position(position)

    # -------------------------------------------------------------- decisioning

    def on_new_bar(self) -> None:
        frame = self.client.bars()
        signal, atr_value = latest_signal(frame, self.config)
        held = self._current_side()
        close = float(frame["close"].iloc[-1])

        log.info(
            "Bar %s close %.5f | signal %s | holding %s | ATR %.5f",
            frame.index[-1].strftime("%Y-%m-%d %H:%M"), close,
            signal.value.upper(), held.value.upper(), atr_value,
        )

        if self._halted_for_day:
            log.info("Halted for the day; managing existing positions only.")
            return

        if held is Signal.FLAT:
            if signal is not Signal.FLAT:
                self._open(signal, atr_value)
            return

        # Already in the market.
        if signal is held or signal is Signal.FLAT:
            return  # let the stop or target do its job

        log.info("Signal flipped %s -> %s; reversing.", held.value.upper(), signal.value.upper())
        self._close_all("signal reversal")
        self._open(signal, atr_value)

    # --------------------------------------------------------------- main loop

    def run(self) -> None:
        cfg = self.config
        log.info(
            "Starting on %s %s | magic %d | dry_run=%s | risk %.2f%%/trade, %.1f%% daily cap",
            cfg.symbol, cfg.timeframe_name, cfg.magic, cfg.execution.dry_run,
            cfg.risk.risk_per_trade_pct, cfg.risk.max_daily_loss_pct,
        )
        if cfg.execution.dry_run:
            log.info("DRY RUN: signals and sizing are real, no orders will be sent.")

        try:
            while True:
                try:
                    latest = self.client.bars().index[-1]
                    if latest != self._last_bar_time:
                        self._last_bar_time = latest
                        self.on_new_bar()
                except MT5Error as exc:
                    # Weekend gaps and brief terminal hiccups land here. Keep going.
                    log.warning("Skipping cycle: %s", exc)

                time.sleep(cfg.execution.poll_seconds)

        except KeyboardInterrupt:
            log.info("Interrupted. Open positions are LEFT OPEN and still carry their "
                     "stop loss and take profit - close them in the terminal if you "
                     "do not want them running unattended.")
