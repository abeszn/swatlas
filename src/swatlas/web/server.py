"""Local dashboard for the Swatlas bot.

Serves a single-page UI plus a small JSON API over the live MT5 terminal.

Two deliberate constraints:

* Binds to 127.0.0.1 only. This exposes account balances and can close
  positions - it must never be reachable from the network.
* All terminal calls are serialised behind one lock. The MetaTrader5 package
  wraps a single IPC channel to the terminal and is not safe to call
  concurrently from FastAPI's thread pool.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import MetaTrader5 as mt5
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ..config import TIMEFRAMES, Config, load_config
from ..contract import Contract
from ..mt5_client import MT5Client
from ..news import NewsCalendar
from ..portfolio import PortfolioLimits
from ..scanner import run_scan, scan_symbol
from ..strategy import latest_signal
from ..timeutil import detect_server_offset, rates_to_frame

log = logging.getLogger("web")

STATIC = Path(__file__).parent / "static"
_lock = threading.Lock()

TRADE_MODES = {0: "DEMO", 1: "CONTEST", 2: "REAL"}


class TradeRequest(BaseModel):
    """Body for POST /api/trade.

    Must live at MODULE scope. This file uses `from __future__ import
    annotations`, so annotations are strings resolved against module globals -
    a class defined inside create_app() is invisible to that lookup, and FastAPI
    silently degrades the parameter to a query field (422 "Field required").
    """
    symbol: str
    side: str                      # must still match the live signal
    risk_pct: float | None = None
    min_lot: float | None = None
    max_lot: float | None = None
    fixed_lot: float | None = None
    dry_run: bool | None = None


class AutoRunRequest(BaseModel):
    """Body for POST /api/autorun - scan the basket and place every survivor.

    Same lot/risk knobs as a scan, plus the portfolio limits, executed in one
    locked call. dry_run defaults to config.yaml; the UI never sends live orders
    without the account and allow_live_account guards below agreeing.
    """
    risk_pct: float | None = None
    min_lot: float | None = None
    max_lot: float | None = None
    fixed_lot: float | None = None
    risk_budget: float | None = None
    max_corr: float | None = None
    max_currency_risk: float | None = None
    max_trades: int = 4
    dry_run: bool | None = None


def apply_risk_overrides(
    base: Config, *, risk_pct: float | None = None, min_lot: float | None = None,
    max_lot: float | None = None, fixed_lot: float | None = None,
) -> Config:
    """Return `base` with the dashboard's risk/lot knobs applied and validated.

    dataclasses.replace bypasses load_config's validation, so a UI that sends
    min_lot > max_lot would otherwise produce silently broken sizing. Re-check
    the same invariants here and reject bad combinations with a 400.
    """
    changes: dict = {}
    if risk_pct is not None:
        changes["risk_per_trade_pct"] = risk_pct
    if min_lot is not None:
        changes["min_lot"] = min_lot
    if max_lot is not None:
        changes["max_lot"] = max_lot
    if fixed_lot is not None:
        changes["fixed_lot"] = fixed_lot
    if not changes:
        return base

    merged = dataclasses.replace(base.risk, **changes)
    if not 0 < merged.risk_per_trade_pct <= 5:
        raise HTTPException(400, "risk_pct must be in (0, 5].")
    if merged.max_lot <= 0:
        raise HTTPException(400, "max_lot must be positive.")
    if merged.min_lot < 0 or merged.fixed_lot < 0:
        raise HTTPException(400, "lot sizes must not be negative.")
    if merged.min_lot > merged.max_lot:
        raise HTTPException(400, "min_lot must not exceed max_lot.")
    if merged.fixed_lot > merged.max_lot:
        raise HTTPException(400, "fixed_lot must not exceed max_lot.")
    return dataclasses.replace(base, risk=merged)


def create_app(config: Config | None = None) -> FastAPI:
    config = config or load_config()
    app = FastAPI(title="Swatlas", docs_url=None, redoc_url=None)
    # Contracts are cached per symbol: verification costs an order_calc_profit
    # round trip and the values do not change during a session.
    state: dict = {"contracts": {}, "connected": False}

    def contract_for(symbol: str) -> Contract:
        cached = state["contracts"].get(symbol)
        if cached is not None:
            return cached
        info = mt5.symbol_info(symbol)
        if info is None:
            raise HTTPException(404, f"Symbol {symbol!r} not found on this broker.")
        if not info.visible:
            mt5.symbol_select(symbol, True)
            info = mt5.symbol_info(symbol)
        verified = Contract.verified(info, mt5)
        state["contracts"][symbol] = verified
        return verified

    def resolve_timeframe(name: str | None) -> tuple[str, int]:
        label = (name or config.timeframe_name).upper()
        if label not in TIMEFRAMES:
            raise HTTPException(
                400, f"Unknown timeframe {label!r}; expected one of "
                     f"{', '.join(TIMEFRAMES)}")
        return label, TIMEFRAMES[label]

    def view_config(symbol: str | None, timeframe: str | None) -> Config:
        """A config pointed at whatever the UI is currently looking at."""
        label, value = resolve_timeframe(timeframe)
        return dataclasses.replace(
            config, symbol=symbol or config.symbol,
            timeframe_name=label, timeframe=value)

    def _initialize() -> bool:
        creds = config.credentials
        kwargs = {}
        if creds.path:
            kwargs["path"] = creds.path
        if creds.login and creds.password and creds.server:
            kwargs.update(login=creds.login, password=creds.password, server=creds.server)
        return bool(mt5.initialize(**kwargs))

    def connect() -> None:
        """Ensure a live terminal link, reconnecting only when actually needed.

        Calling mt5.initialize() on every request looks harmless but is not: the
        terminal exposes ONE IPC channel, and hammering it while the dashboard
        polls (plus any CLI script running alongside) produces intermittent
        'Authorization failed' errors. That surfaced as a flood of 503s and a
        Close button that appeared to hang.

        So: probe cheaply, reconnect with a short retry only on real failure.
        """
        if state["connected"] and mt5.terminal_info() is not None:
            pass
        else:
            state["connected"] = False
            for attempt in range(3):
                if _initialize() and mt5.terminal_info() is not None:
                    state["connected"] = True
                    break
                time.sleep(0.25 * (attempt + 1))
            else:
                code, msg = mt5.last_error()
                raise HTTPException(
                    503, f"MT5 terminal unreachable ({code}: {msg}). "
                         "Is the terminal running and logged in?")

        # Warm the configured symbol, best-effort. Different accounts/brokers
        # offer different symbol names (or none at all) - a missing symbol
        # here must not take down every endpoint (balance, positions, scan),
        # since none of those actually depend on config.yaml's single symbol.
        try:
            contract_for(config.symbol)
        except HTTPException as exc:
            log.warning("configured symbol %r unavailable on this account: %s",
                       config.symbol, exc.detail)

    @app.get("/")
    def index() -> FileResponse:
        # no-store: the browser otherwise serves a cached page after an edit,
        # which looks exactly like "my fix did not work".
        return FileResponse(STATIC / "index.html", headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        })

    @app.get("/api/symbols")
    def api_symbols(all: bool = False) -> JSONResponse:
        """Symbols for the chart picker.

        Defaults to Market Watch only. This broker lists 12,540 instruments but
        just 24 are in Market Watch - dumping all of them into a dropdown makes
        it unusable, and the rest are stocks and CFDs nobody is charting here.
        Pass ?all=true to get the full list.
        """
        with _lock:
            connect()
            everything = mt5.symbols_get() or ()
            chosen = everything if all else [s for s in everything if s.visible]
            # Always include whatever the bot is configured to trade, even if
            # someone removed it from Market Watch.
            if config.symbol not in {s.name for s in chosen}:
                chosen = list(chosen) + [s for s in everything
                                         if s.name == config.symbol]

            rows = [{"name": s.name,
                     "group": (s.path.split("\\")[0] if s.path else "Other"),
                     "digits": s.digits, "visible": bool(s.visible)}
                    for s in chosen]
            rows.sort(key=lambda r: (r["group"], r["name"]))
            return JSONResponse({
                "symbols": rows,
                "total_available": len(everything),
                "showing_all": bool(all),
                "timeframes": list(TIMEFRAMES),
                "configured": config.symbol,
                "configured_timeframe": config.timeframe_name,
            })

    @app.get("/api/state")
    def api_state(symbol: str | None = None,
                  timeframe: str | None = None) -> JSONResponse:
        with _lock:
            connect()
            view = view_config(symbol, timeframe)
            account = mt5.account_info()
            if account is None:
                raise HTTPException(503, "Connected, but the terminal reports no logged-in account.")
            terminal = mt5.terminal_info()

            # The viewed symbol may not exist on whichever account is logged
            # into the terminal right now - that must not block account
            # balance, positions or trade proposals, none of which need it.
            info = mt5.symbol_info(view.symbol)
            symbol_available = info is not None
            tick = mt5.symbol_info_tick(view.symbol) if symbol_available else None
            contract = None
            if symbol_available:
                try:
                    contract = contract_for(view.symbol)
                except HTTPException:
                    symbol_available = False

            positions = []
            for p in (mt5.positions_get() or ()):
                # Price precision is PER SYMBOL. Using the configured symbol's
                # digits for every row renders FX pairs as "1.39" when the
                # dashboard is pointed at gold - correct rounding, useless table.
                p_info = mt5.symbol_info(p.symbol)
                positions.append({
                    "ticket": p.ticket, "symbol": p.symbol,
                    "side": "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
                    "volume": p.volume, "open": p.price_open, "current": p.price_current,
                    "sl": p.sl, "tp": p.tp, "profit": round(p.profit, 2),
                    "swap": round(p.swap, 2), "magic": p.magic,
                    "digits": p_info.digits if p_info is not None
                             else (contract.digits if contract else 5),
                    "ours": p.magic == config.magic,
                    "opened": datetime.fromtimestamp(p.time, tz=timezone.utc).isoformat(),
                })

            signal_name, atr_value, last_bar = None, None, None
            if symbol_available:
                try:
                    rates = mt5.copy_rates_from_pos(
                        view.symbol, view.timeframe, 0, view.bars_needed + 1)
                    if rates is not None and len(rates) > 2:
                        frame = rates_to_frame(rates).iloc[:-1]
                        sig, a = latest_signal(frame, view)
                        signal_name, atr_value = sig.value, round(float(a), contract.digits + 2)
                        last_bar = pd.DatetimeIndex(frame.index)[-1].isoformat()
                except Exception as exc:
                    log.debug("signal unavailable: %s", exc)

            mode = TRADE_MODES.get(account.trade_mode, "UNKNOWN")
            return JSONResponse({
                "account": {
                    "login": account.login, "server": account.server, "mode": mode,
                    "currency": account.currency, "balance": round(account.balance, 2),
                    "equity": round(account.equity, 2), "margin": round(account.margin, 2),
                    "free_margin": round(account.margin_free, 2),
                    "margin_level": round(account.margin_level, 1),
                    "leverage": account.leverage,
                    "profit": round(account.profit, 2),
                    "trade_allowed": bool(account.trade_allowed),
                },
                "terminal": {
                    "connected": bool(terminal.connected),
                    "algo_trading": bool(terminal.trade_allowed),
                    "build": mt5.version()[1] if mt5.version() else None,
                    "server_utc_offset": detect_server_offset().total_seconds() / 3600,
                },
                "symbol": {
                    "name": view.symbol, "timeframe": view.timeframe_name,
                    "configured": config.symbol,
                    "configured_timeframe": config.timeframe_name,
                    "available": symbol_available,
                    "error": None if symbol_available
                             else f"{view.symbol!r} is not offered by this account's broker.",
                    "bid": tick.bid if tick else None, "ask": tick.ask if tick else None,
                    "spread_points": info.spread if info else None,
                    "digits": contract.digits if contract else 5,
                    "tick_value": contract.tick_value if contract else None,
                    "tick_value_source": contract.tick_value_source if contract else None,
                    "market_open": bool(tick and tick.ask > 0),
                },
                "strategy": {
                    "signal": signal_name, "atr": atr_value, "last_bar": last_bar,
                    "magic": config.magic,
                    "fast": config.strategy.fast_period,
                    "slow": config.strategy.slow_period,
                    "risk_pct": config.risk.risk_per_trade_pct,
                    "min_lot": config.risk.min_lot,
                    "max_lot": config.risk.max_lot,
                    "fixed_lot": config.risk.fixed_lot,
                    "dry_run": config.execution.dry_run,
                },
                "positions": positions,
                "server_time": datetime.now(timezone.utc).isoformat(),
            })

    @app.get("/api/candles")
    def api_candles(bars: int = 200, symbol: str | None = None,
                    timeframe: str | None = None) -> JSONResponse:
        bars = max(50, min(bars, 1000))
        with _lock:
            connect()
            view = view_config(symbol, timeframe)
            contract_for(view.symbol)
            # Pull enough extra bars for the indicators to warm up, or the first
            # part of every chart would be blank.
            warmup = max(view.strategy.slow_period, view.risk.atr_period,
                         view.strategy.min_history) + 5
            rates = mt5.copy_rates_from_pos(
                view.symbol, view.timeframe, 0, bars + warmup)
            if rates is None or len(rates) < 10:
                raise HTTPException(
                    503, f"No price history for {view.symbol} {view.timeframe_name}.")

            frame = rates_to_frame(rates).iloc[:-1]

            from ..strategy import generate_signals
            enriched = generate_signals(frame, view).tail(bars)

            # Only ma_cross exposes fast/slow. donchian and smc do not, so the
            # overlays must be discovered rather than assumed - hardcoding them
            # turned every chart request into a 500 the moment the strategy
            # changed.
            overlays = [c for c in ("fast", "slow") if c in enriched.columns]

            def value(row, column):
                raw = row.get(column)
                return None if raw is None or pd.isna(raw) else float(raw)

            return JSONResponse({
                "symbol": view.symbol, "timeframe": view.timeframe_name,
                "strategy": view.strategy.name,
                "digits": contract_for(view.symbol).digits,
                "overlays": overlays,
                "candles": [
                    # iterrows() types its key as Hashable; this frame's index is
                    # always a DatetimeIndex, so the cast states what is true.
                    {"t": cast(pd.Timestamp, idx).isoformat(),
                     "o": float(r["open"]), "h": float(r["high"]),
                     "l": float(r["low"]), "c": float(r["close"]),
                     "signal": r["signal"],
                     **{c: value(r, c) for c in overlays}}
                    for idx, r in enriched.iterrows()
                ],
            })

    @app.get("/api/deals")
    def api_deals(days: int = 30) -> JSONResponse:
        with _lock:
            connect()
            now = datetime.now(timezone.utc)
            deals = mt5.history_deals_get(now - timedelta(days=days),
                                          now + timedelta(days=1)) or ()
            rows = []
            for d in deals:
                if d.entry != mt5.DEAL_ENTRY_OUT:   # only closing deals realise P&L
                    continue
                d_info = mt5.symbol_info(d.symbol)
                rows.append({
                    "ticket": d.ticket, "symbol": d.symbol, "volume": d.volume,
                    "price": d.price,
                    "digits": d_info.digits if d_info is not None else 5,
                    "profit": round(d.profit + d.commission + d.swap, 2),
                    "magic": d.magic, "ours": d.magic == config.magic,
                    "time": datetime.fromtimestamp(d.time, tz=timezone.utc).isoformat(),
                })
            rows.sort(key=lambda r: r["time"], reverse=True)

            realised = sum(r["profit"] for r in rows if r["ours"])
            wins = [r for r in rows if r["ours"] and r["profit"] > 0]
            ours = [r for r in rows if r["ours"]]
            return JSONResponse({
                "deals": rows[:100],
                "stats": {
                    "closed": len(ours),
                    "realised": round(realised, 2),
                    "win_rate": round(len(wins) / len(ours) * 100, 1) if ours else None,
                },
            })

    def _close_one(position) -> dict:
        """Close a single position at market. Assumes the lock is already held."""
        info = mt5.symbol_info(position.symbol)
        allowed = info.filling_mode
        filling = (mt5.ORDER_FILLING_FOK if allowed & 1
                   else mt5.ORDER_FILLING_IOC if allowed & 2
                   else mt5.ORDER_FILLING_RETURN)
        tick = mt5.symbol_info_tick(position.symbol)
        if tick is None or tick.ask <= 0:
            return {"ticket": position.ticket, "symbol": position.symbol,
                    "ok": False, "comment": "no live tick - market closed"}

        is_long = position.type == mt5.POSITION_TYPE_BUY
        result = mt5.order_send({
            "action": mt5.TRADE_ACTION_DEAL, "symbol": position.symbol,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if is_long else mt5.ORDER_TYPE_BUY,
            "position": position.ticket,
            "price": tick.bid if is_long else tick.ask,
            "deviation": config.execution.deviation_points,
            "magic": config.magic, "comment": "swatlas-ui-close",
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": filling,
        })
        if result is None:
            code, msg = mt5.last_error()
            return {"ticket": position.ticket, "symbol": position.symbol,
                    "ok": False, "comment": f"order_send failed ({code}: {msg})"}

        ok = result.retcode == mt5.TRADE_RETCODE_DONE
        log.info("UI close #%s %s -> %s (%s)", position.ticket, position.symbol,
                 result.retcode, result.comment)
        return {"ticket": position.ticket, "symbol": position.symbol, "ok": ok,
                "retcode": result.retcode, "comment": result.comment,
                "price": result.price}

    @app.post("/api/scan")
    def api_scan(risk_pct: float | None = None,
                 min_lot: float | None = None,
                 max_lot: float | None = None,
                 fixed_lot: float | None = None,
                 risk_budget: float | None = None,
                 max_corr: float | None = None,
                 max_currency_risk: float | None = None,
                 max_trades: int = 4) -> JSONResponse:
        with _lock:
            connect()
            scan_config = apply_risk_overrides(
                config, risk_pct=risk_pct, min_lot=min_lot,
                max_lot=max_lot, fixed_lot=fixed_lot)

            calendar = None
            if scan_config.news.enabled:
                calendar = NewsCalendar.load(scan_config.news.max_cache_age_hours)

            defaults = PortfolioLimits()
            limits = PortfolioLimits(
                max_total_risk_pct=(risk_budget if risk_budget is not None
                                    else defaults.max_total_risk_pct),
                max_pairwise_corr=(max_corr if max_corr is not None
                                   else defaults.max_pairwise_corr),
                max_currency_risk_pct=(max_currency_risk
                                       if max_currency_risk is not None
                                       else defaults.max_currency_risk_pct),
                max_trades=max_trades)

            result = run_scan(scan_config, calendar=calendar, limits=limits)
            payload = result.to_dict()
            payload["limits"] = {
                "risk_pct": scan_config.risk.risk_per_trade_pct,
                "min_lot": scan_config.risk.min_lot,
                "max_lot": scan_config.risk.max_lot,
                "fixed_lot": scan_config.risk.fixed_lot,
                "risk_budget": limits.max_total_risk_pct,
                "max_corr": limits.max_pairwise_corr,
                "max_currency_risk": limits.max_currency_risk_pct,
                "max_trades": limits.max_trades,
            }
            payload["dry_run"] = config.execution.dry_run
            return JSONResponse(payload)

    @app.post("/api/trade")
    def api_trade(request: TradeRequest) -> JSONResponse:
        """Execute an approved proposal.

        Everything is RE-DERIVED here. The browser sends a symbol and a
        direction, never a lot size or a stop - those come from the risk engine
        at execution time, against current price and volatility. A stale tab
        therefore cannot place an oversized or reversed trade.
        """
        with _lock:
            connect()
            account = mt5.account_info()
            if account is None:
                raise HTTPException(503, "No account information.")
            if account.trade_mode == mt5.ACCOUNT_TRADE_MODE_REAL and \
                    not config.execution.allow_live_account:
                raise HTTPException(
                    403, "REAL-money account and allow_live_account is false.")

            if request.side not in ("buy", "sell"):
                raise HTTPException(400, f"bad side {request.side!r}")

            trade_config = apply_risk_overrides(
                config, risk_pct=request.risk_pct, min_lot=request.min_lot,
                max_lot=request.max_lot, fixed_lot=request.fixed_lot)

            calendar = (NewsCalendar.load(trade_config.news.max_cache_age_hours)
                        if trade_config.news.enabled else None)
            fresh = scan_symbol(request.symbol, trade_config, calendar)

            if fresh.plan is None or not fresh.actionable:
                return JSONResponse(
                    {"ok": False, "symbol": request.symbol,
                     "comment": f"no longer tradeable: {fresh.status}"
                                + (f" ({fresh.news})" if fresh.news else "")},
                    status_code=409)

            if fresh.plan.side != request.side:
                # The signal flipped between the scan and the click.
                return JSONResponse(
                    {"ok": False, "symbol": request.symbol,
                     "comment": f"signal changed to {fresh.plan.side.upper()} "
                                f"since the scan - rescan before trading"},
                    status_code=409)

            dry = config.execution.dry_run if request.dry_run is None else request.dry_run
            client = MT5Client(dataclasses.replace(
                trade_config,
                execution=dataclasses.replace(trade_config.execution, dry_run=dry)))
            client.use_symbol(request.symbol)      # reuses the audited order path

            outcome = client.market_order(
                fresh.plan.side, fresh.plan.volume,
                fresh.plan.stop_loss, fresh.plan.take_profit,
                comment="swatlas-ui")

            return JSONResponse({
                "ok": outcome.ok, "dry_run": dry, "symbol": request.symbol,
                "side": fresh.plan.side, "volume": outcome.volume or fresh.plan.volume,
                "price": outcome.price, "ticket": outcome.order,
                "stop_loss": fresh.plan.stop_loss,
                "take_profit": fresh.plan.take_profit,
                "risk_amount": round(fresh.plan.risk_amount, 2),
                "comment": outcome.comment,
            }, status_code=200 if outcome.ok else 400)

    @app.post("/api/autorun")
    def api_autorun(request: AutoRunRequest) -> JSONResponse:
        """Scan the basket and place EVERY surviving proposal in one call.

        This is the dashboard's "Find & Run" button. The scan and the orders
        happen inside a single held lock, so the proposals are executed at the
        instant they were sized - no stale-tab window like /api/trade guards
        against, because nothing else can interleave. dry_run is still the final
        gate: with it true (the config default) this only logs intended orders.
        """
        with _lock:
            connect()
            account = mt5.account_info()
            if account is None:
                raise HTTPException(503, "No account information.")
            if account.trade_mode == mt5.ACCOUNT_TRADE_MODE_REAL and \
                    not config.execution.allow_live_account:
                raise HTTPException(
                    403, "REAL-money account and allow_live_account is false.")

            run_config = apply_risk_overrides(
                config, risk_pct=request.risk_pct, min_lot=request.min_lot,
                max_lot=request.max_lot, fixed_lot=request.fixed_lot)

            calendar = None
            if run_config.news.enabled:
                calendar = NewsCalendar.load(run_config.news.max_cache_age_hours)

            defaults = PortfolioLimits()
            limits = PortfolioLimits(
                max_total_risk_pct=(request.risk_budget
                                    if request.risk_budget is not None
                                    else defaults.max_total_risk_pct),
                max_pairwise_corr=(request.max_corr if request.max_corr is not None
                                   else defaults.max_pairwise_corr),
                max_currency_risk_pct=(request.max_currency_risk
                                       if request.max_currency_risk is not None
                                       else defaults.max_currency_risk_pct),
                max_trades=request.max_trades)

            result = run_scan(run_config, calendar=calendar, limits=limits)

            dry = (config.execution.dry_run if request.dry_run is None
                   else request.dry_run)
            client = MT5Client(dataclasses.replace(
                run_config,
                execution=dataclasses.replace(run_config.execution, dry_run=dry)))

            executed = []
            for prop in result.proposals:
                if prop.plan is None:
                    continue
                try:
                    client.use_symbol(prop.symbol)
                    outcome = client.market_order(
                        prop.plan.side, prop.plan.volume,
                        prop.plan.stop_loss, prop.plan.take_profit,
                        comment="swatlas-autorun")
                    executed.append({
                        "ok": outcome.ok, "symbol": prop.symbol,
                        "side": prop.plan.side,
                        "volume": outcome.volume or prop.plan.volume,
                        "price": outcome.price, "ticket": outcome.order,
                        "stop_loss": prop.plan.stop_loss,
                        "take_profit": prop.plan.take_profit,
                        "risk_amount": round(prop.plan.risk_amount, 2),
                        "comment": outcome.comment,
                    })
                except Exception as exc:                 # one failed order must
                    log.warning("autorun %s failed: %s", prop.symbol, exc)  # not
                    executed.append({"ok": False, "symbol": prop.symbol,  # abort
                                     "comment": str(exc)})                 # the rest

            payload = result.to_dict()
            payload["limits"] = {
                "risk_pct": run_config.risk.risk_per_trade_pct,
                "min_lot": run_config.risk.min_lot,
                "max_lot": run_config.risk.max_lot,
                "fixed_lot": run_config.risk.fixed_lot,
                "risk_budget": limits.max_total_risk_pct,
                "max_corr": limits.max_pairwise_corr,
                "max_currency_risk": limits.max_currency_risk_pct,
                "max_trades": limits.max_trades,
            }
            payload["dry_run"] = dry
            payload["executed"] = executed
            payload["placed"] = sum(1 for e in executed if e["ok"])
            return JSONResponse(payload)

    @app.post("/api/positions/{ticket}/close")
    def api_close(ticket: int) -> JSONResponse:
        with _lock:
            connect()
            matches = mt5.positions_get(ticket=ticket)
            if not matches:
                # Already gone - a stop or target may have fired first. That is
                # a success from the user's point of view, not an error.
                return JSONResponse({"ok": True, "already_closed": True,
                                     "comment": "position already closed"})
            outcome = _close_one(matches[0])
            return JSONResponse(outcome, status_code=200 if outcome["ok"] else 400)

    @app.post("/api/positions/close-all")
    def api_close_all(only_ours: bool = True) -> JSONResponse:
        with _lock:
            connect()
            positions = mt5.positions_get() or ()
            if only_ours:
                positions = [p for p in positions if p.magic == config.magic]
            if not positions:
                return JSONResponse({"ok": True, "closed": 0, "results": []})

            results = [_close_one(p) for p in positions]
            closed = sum(1 for r in results if r["ok"])
            return JSONResponse({"ok": closed == len(results), "closed": closed,
                                 "attempted": len(results), "results": results})

    return app
