"""Signal-driven, cost-aware bar-by-bar backtester.

Takes a frame that already carries `signal` and `atr` columns, so any strategy -
the live one in strategy.py or a research candidate - runs through identical
execution and accounting code.

Conservative by construction:

* Entries fill at the NEXT bar's open. Filling on the signal bar's close is
  lookahead bias and is the usual reason a backtest flatters a strategy.
* When a bar's range covers both stop and target, the STOP is taken. Bar data
  cannot say which came first, so it assumes the bad one.
* Spread and slippage are charged on every fill; swap is charged per night held;
  commission is charged per round turn.

Still not modelled: variable/news spread widening, requotes, partial fills,
weekend gaps through a stop (the stop is assumed to fill at its price, which is
optimistic), and the fact that your fill is not the whole market's fill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .contract import Contract
from .costs import Costs


@dataclass(frozen=True)
class BacktestParams:
    risk_per_trade_pct: float = 0.5
    stop_atr_mult: float = 2.0
    target_atr_mult: float = 3.0
    trail_atr_mult: float | None = None   # None disables trailing
    max_hold_bars: int | None = None
    exit_on_flat: bool = False            # True: leave when signal goes FLAT
    min_lot: float = 0.0                  # floor: size up to this even if risk maths says less
    max_lot: float = 100.0
    starting_balance: float = 10_000.0
    compound: bool = True                 # size off current balance vs starting
    bars_per_year: int = 252 * 6          # H4 default; used for CAGR/Sharpe


@dataclass
class Trade:
    entry_time: pd.Timestamp
    side: str
    entry: float
    stop: float
    target: float | None      # None when the strategy runs without a fixed target
    volume: float
    risk_amount: float
    exit_time: pd.Timestamp | None = None
    exit: float | None = None
    reason: str = ""
    bars_held: int = 0
    gross: float = 0.0
    costs: float = 0.0
    pnl: float = 0.0

    @property
    def r_multiple(self) -> float:
        """P&L in units of the risk taken. The only comparable measure of a trade."""
        return self.pnl / self.risk_amount if self.risk_amount else 0.0


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity: pd.Series = field(default_factory=pd.Series)
    exposure: pd.Series = field(default_factory=pd.Series)
    params: BacktestParams = field(default_factory=BacktestParams)
    benchmark_return_pct: float = 0.0

    @property
    def closed(self) -> list[Trade]:
        return [t for t in self.trades if t.exit is not None and t.exit_time is not None]

    def summary(self) -> dict:
        trades = self.closed
        start = self.params.starting_balance
        if not trades or len(self.equity) < 2:
            return {"trades": len(trades), "note": "insufficient data"}

        pnls = pd.Series([t.pnl for t in trades])
        rs = pd.Series([t.r_multiple for t in trades])
        wins, losses = pnls[pnls > 0], pnls[pnls <= 0]
        gross_win, gross_loss = wins.sum(), abs(losses.sum())

        final = float(self.equity.iloc[-1])
        years = len(self.equity) / self.params.bars_per_year
        cagr = ((final / start) ** (1 / years) - 1) * 100 if years > 0 and final > 0 else -100.0

        bar_returns = self.equity.pct_change().dropna()
        sharpe = 0.0
        if bar_returns.std() > 0:
            sharpe = (bar_returns.mean() / bar_returns.std()
                      * np.sqrt(self.params.bars_per_year))

        peak = self.equity.cummax()
        drawdown = ((self.equity - peak) / peak).min() * 100

        return {
            "trades": len(trades),
            "win_rate_pct": round(len(wins) / len(trades) * 100, 1),
            "expectancy_R": round(rs.mean(), 3),
            "total_R": round(rs.sum(), 1),
            "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else float("inf"),
            "return_pct": round((final / start - 1) * 100, 1),
            "cagr_pct": round(cagr, 2),
            "sharpe": round(sharpe, 2),
            "max_dd_pct": round(drawdown, 1),
            "calmar": round(cagr / abs(drawdown), 2) if drawdown else 0.0,
            "avg_bars_held": round(np.mean([t.bars_held for t in trades]), 1),
            "exposure_pct": round(self.exposure.mean() * 100, 1),
            "costs_paid": round(sum(t.costs for t in trades), 0),
            "cost_drag_pct_of_gross": (
                round(sum(t.costs for t in trades) / abs(sum(t.gross for t in trades)) * 100, 1)
                if sum(t.gross for t in trades) else 0.0),
            "final_balance": round(final, 0),
            "buy_hold_return_pct": round(self.benchmark_return_pct, 1),
        }

    def yearly(self) -> pd.DataFrame:
        """Per-year P&L and trade count - the fastest way to spot a strategy that
        only worked in one regime."""
        if not self.closed:
            return pd.DataFrame()
        rows = pd.DataFrame([
            {"year": t.exit_time.year, "pnl": t.pnl, "R": t.r_multiple}
            for t in self.closed if t.exit_time is not None
        ])
        if rows.empty:
            return pd.DataFrame()
        return rows.groupby("year").agg(
            trades=("pnl", "size"), pnl=("pnl", "sum"), total_R=("R", "sum"),
            win_rate=("pnl", lambda s: round((s > 0).mean() * 100, 1)),
        ).round(2)


def run_backtest(frame: pd.DataFrame, contract: Contract, costs: Costs,
                 params: BacktestParams) -> BacktestResult:
    """Replay `frame` bar by bar.

    `frame` must have open/high/low/close plus `signal` ('long'/'short'/'flat')
    and `atr`. `contract` carries the VERIFIED tick value - see contract.py.
    """
    vol_min = contract.volume_min
    vol_max = contract.volume_max

    entry_cost = costs.spread + costs.slippage  # price units, charged once per round trip

    def money(price_move: float, volume: float) -> float:
        return contract.money(price_move, volume)

    balance = params.starting_balance
    equity_points: list[float] = []
    in_market: list[int] = []
    trades: list[Trade] = []
    open_trade: Trade | None = None
    entry_index = 0

    # itertuples yields dynamically-generated namedtuples; a checker can only see
    # them as plain tuples, so attribute access (bar.close, bar.atr) reads as an
    # error. Annotating Any states the truth rather than suppressing diagnostics.
    rows: list[Any] = list(frame.itertuples())
    n = len(rows)

    def close_trade(trade: Trade, price: float, when: pd.Timestamp,
                    reason: str, bar_i: int) -> float:
        move = (price - trade.entry) if trade.side == "buy" else (trade.entry - price)
        trade.gross = money(move, trade.volume)

        # Financing accrues on notional, not on a flat per-lot figure - gold's
        # price moved 10x across this sample, so a flat charge is meaningless.
        nights = max(0, (when.normalize() - trade.entry_time.normalize()).days)
        annual_pct = (costs.swap_long_annual_pct if trade.side == "buy"
                      else costs.swap_short_annual_pct)
        notional = contract.notional(trade.volume, trade.entry)
        swap = notional * (annual_pct / 100.0) * (nights / 365.0)
        commission = costs.commission_per_lot_roundturn * trade.volume
        spread_cost = money(entry_cost, trade.volume)

        trade.costs = spread_cost + commission - swap  # positive number = paid
        trade.pnl = trade.gross - spread_cost - commission + swap
        trade.exit, trade.exit_time, trade.reason = price, when, reason
        trade.bars_held = bar_i - entry_index
        return trade.pnl

    for i, bar in enumerate(rows):
        # ---- manage an open position against this bar's range -----------------
        if open_trade is not None:
            long = open_trade.side == "buy"

            if long:
                hit_stop = bar.low <= open_trade.stop
                hit_target = (open_trade.target is not None
                              and bar.high >= open_trade.target)
            else:
                hit_stop = bar.high >= open_trade.stop
                hit_target = (open_trade.target is not None
                              and bar.low <= open_trade.target)

            if hit_stop:  # ambiguity always resolves against us
                balance += close_trade(open_trade, open_trade.stop, bar.Index, "stop", i)
                open_trade = None
            elif hit_target and open_trade.target is not None:
                balance += close_trade(open_trade, open_trade.target, bar.Index, "target", i)
                open_trade = None
            else:
                # Trail the stop on favourable closes, never loosening it.
                if params.trail_atr_mult and not np.isnan(bar.atr):
                    slack = bar.atr * params.trail_atr_mult
                    if long:
                        open_trade.stop = max(open_trade.stop, bar.close - slack)
                    else:
                        open_trade.stop = min(open_trade.stop, bar.close + slack)

                if params.max_hold_bars and (i - entry_index) >= params.max_hold_bars:
                    balance += close_trade(open_trade, bar.close, bar.Index, "time", i)
                    open_trade = None

        # ---- act on this bar's signal, filling at the NEXT bar's open ---------
        next_bar = rows[i + 1] if i + 1 < n else None
        signal = getattr(bar, "signal", "flat")
        wanted = {"long": "buy", "short": "sell"}.get(signal)

        if next_bar is not None:
            held = open_trade.side if open_trade else None

            needs_exit = (
                open_trade is not None
                and ((wanted is not None and wanted != held)
                     or (params.exit_on_flat and wanted is None))
            )
            if open_trade is not None and needs_exit:
                reason = "reversal" if wanted is not None else "flat"
                balance += close_trade(open_trade, next_bar.open, next_bar.Index, reason, i + 1)
                open_trade = None
                held = None

            if wanted is not None and held is None and not np.isnan(bar.atr) and bar.atr > 0:
                fill = next_bar.open + (entry_cost if wanted == "buy" else -entry_cost)
                stop_distance = bar.atr * params.stop_atr_mult

                sizing_base = balance if params.compound else params.starting_balance
                risk_amount = sizing_base * params.risk_per_trade_pct / 100.0
                loss_per_lot = contract.loss_per_lot(stop_distance)
                risk_based = contract.round_volume(risk_amount / loss_per_lot)
                # Clamp into [min_lot, max_lot], matching the live risk engine
                # (see risk.position_size). The floor deliberately lets a trade
                # risk more than risk_per_trade_pct so the backtest models the
                # same over-risking the live bot will do.
                ceiling = min(params.max_lot, vol_max)
                floor = max(params.min_lot, vol_min)
                volume = contract.round_volume(min(max(risk_based, floor), ceiling))

                if volume >= vol_min and balance > 0:
                    if wanted == "buy":
                        stop = fill - stop_distance
                        target = (fill + bar.atr * params.target_atr_mult
                                  if params.target_atr_mult else None)
                    else:
                        stop = fill + stop_distance
                        target = (fill - bar.atr * params.target_atr_mult
                                  if params.target_atr_mult else None)

                    open_trade = Trade(
                        entry_time=next_bar.Index, side=wanted, entry=fill,
                        stop=contract.round_price(stop),
                        target=contract.round_price(target) if target else None,
                        volume=volume,
                        risk_amount=money(stop_distance, volume),
                    )
                    entry_index = i + 1
                    trades.append(open_trade)

        equity_points.append(balance)
        in_market.append(1 if open_trade is not None else 0)

    if open_trade is not None:
        balance += close_trade(open_trade, rows[-1].close, rows[-1].Index, "end-of-data", n - 1)
        equity_points[-1] = balance

    benchmark = (frame["close"].iloc[-1] / frame["close"].iloc[0] - 1) * 100

    return BacktestResult(
        trades=trades,
        equity=pd.Series(equity_points, index=frame.index),
        exposure=pd.Series(in_market, index=frame.index),
        params=params,
        benchmark_return_pct=float(benchmark),
    )
