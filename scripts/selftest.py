"""Offline self-test: exercises contract maths, signals, sizing, and the
backtester on synthetic data. Needs no MetaTrader terminal and places no orders.

    .venv\\Scripts\\python.exe scripts\\selftest.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import dataclasses
import sys

import numpy as np
import pandas as pd

from swatlas.backtest import BacktestParams, run_backtest
from swatlas.config import load_config
from swatlas.contract import Contract
from swatlas.costs import Costs
from swatlas.indicators import atr, sma
from swatlas.portfolio import (
    Exposure, PortfolioLimits, combined_risk, currency_exposure,
    effective_correlation, legs, select,
)
from swatlas.risk import build_plan, position_size
from swatlas.strategy import Signal, generate_signals

# EURUSD-like: 100k contract, 5 digits, $1 per 0.00001 tick per lot.
FX = Contract(symbol="EURUSD", digits=5, point=0.00001, contract_size=100_000.0,
              tick_size=0.00001, tick_value=1.0, volume_min=0.01,
              volume_max=100.0, volume_step=0.01)

# XAUUSD as the broker REPORTS it - tick_value 0.1 contradicts 0.01 x 100 = 1.0.
GOLD_REPORTED = Contract(symbol="XAUUSD", digits=2, point=0.01, contract_size=100.0,
                         tick_size=0.01, tick_value=0.1, volume_min=0.01,
                         volume_max=100.0, volume_step=0.01)
# XAUUSD as the terminal's own order_calc_profit proves it to be.
GOLD_TRUE = Contract(symbol="XAUUSD", digits=2, point=0.01, contract_size=100.0,
                     tick_size=0.01, tick_value=1.0, volume_min=0.01,
                     volume_max=100.0, volume_step=0.01)

# USDJPY: base currency IS the account currency, so notional must NOT be
# multiplied by the price. tick_value 0.627 is the verified live value.
JPY = Contract(symbol="USDJPY", digits=3, point=0.001, contract_size=100_000.0,
               tick_size=0.001, tick_value=0.627, volume_min=0.01,
               volume_max=100.0, volume_step=0.01)


def synthetic_bars(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    """Random walk with drifting trends, so a crossover system has something to find."""
    rng = np.random.default_rng(seed)
    trend = np.sin(np.linspace(0, 12 * np.pi, n)) * 0.00004
    close = 1.10 + np.cumsum(rng.normal(0, 0.00035, n) + trend)

    wick = np.abs(rng.normal(0, 0.0004, n))
    frame = pd.DataFrame({
        "open": np.concatenate([[close[0]], close[:-1]]),
        "high": close + wick,
        "low": close - wick,
        "close": close,
        "tick_volume": rng.integers(50, 500, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"))
    frame["high"] = frame[["open", "high", "close"]].max(axis=1)
    frame["low"] = frame[["open", "low", "close"]].min(axis=1)
    return frame


failures: list[str] = []


def check(label: str, condition: object, detail: str = "") -> None:
    """`condition` is typed `object` on purpose: pandas and numpy comparisons
    return numpy.bool_, not builtins.bool, and coercing at every call site
    would be noise. Truthiness is what we actually want here."""
    ok = bool(condition)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(label)


def main() -> int:
    config = load_config()
    frame = synthetic_bars()

    print("\n-- contract maths --")
    check("FX: 100 pips on 1 lot = $1000", abs(FX.money(0.01, 1.0) - 1000.0) < 1e-6,
          f"got {FX.money(0.01, 1.0):.2f}")
    check("GOLD (true): $1/oz on 1 lot = $100",
          abs(GOLD_TRUE.money(1.0, 1.0) - 100.0) < 1e-6,
          f"got {GOLD_TRUE.money(1.0, 1.0):.2f}")
    check("GOLD (reported) is wrong by 10x - the bug this guards against",
          abs(GOLD_REPORTED.money(1.0, 1.0) - 10.0) < 1e-6,
          f"got {GOLD_REPORTED.money(1.0, 1.0):.2f}, should be 100.00")
    check("notional: 0.1 lot gold @ $4000 = $40,000",
          abs(GOLD_TRUE.notional(0.1, 4000.0) - 40_000.0) < 1e-6)
    check("notional: 1 lot EURUSD @ 1.155 = $115,500",
          abs(FX.notional(1.0, 1.155) - 115_500.0) < 1.0,
          f"got ${FX.notional(1.0, 1.155):,.0f}")
    # USDJPY: 1 lot is 100,000 USD, NOT 100,000 * 158. The naive
    # volume*contract_size*price formula overstates this by ~158x.
    check("notional: 1 lot USDJPY @ 158 = ~$100,000 (not $15.8M)",
          abs(JPY.notional(1.0, 158.0) - 100_000.0) < 2_000.0,
          f"got ${JPY.notional(1.0, 158.0):,.0f}")
    check("volume rounds DOWN to the step", GOLD_TRUE.round_volume(0.1749) == 0.17,
          f"got {GOLD_TRUE.round_volume(0.1749)}")
    check("sub-step volume rounds to 0", GOLD_TRUE.round_volume(0.004) == 0.0)

    print("\n-- indicators --")
    fast = sma(frame["close"], 20)
    atr_series = atr(frame, 14)
    check("SMA warm-up is NaN then populated",
          fast.iloc[:19].isna().all() and fast.iloc[19:].notna().all())
    check("SMA matches a manual mean",
          abs(fast.iloc[50] - frame["close"].iloc[31:51].mean()) < 1e-12)
    check("ATR is positive and finite",
          bool((atr_series.dropna() > 0).all()) and np.isfinite(atr_series.iloc[-1]),
          f"last={atr_series.iloc[-1]:.6f}")

    print("\n-- signals --")
    # Pin the strategy rather than inheriting config.yaml: these assertions are
    # about ma_cross specifically, and must not break when the live config is
    # pointed at a different strategy.
    ma_config = dataclasses.replace(
        config, strategy=dataclasses.replace(config.strategy,
                                             name="ma_cross", params={}))
    signals = generate_signals(frame, ma_config)
    warmup = max(ma_config.strategy.slow_period, ma_config.risk.atr_period)
    check("no signal during warm-up",
          (signals["signal"].iloc[:warmup - 1] == Signal.FLAT.value).all())
    check("long signals sit above the slow MA",
          bool((signals.loc[signals["signal"] == "long", "close"] >
                signals.loc[signals["signal"] == "long", "slow"]).all()))
    check("short signals sit below the slow MA",
          bool((signals.loc[signals["signal"] == "short", "close"] <
                signals.loc[signals["signal"] == "short", "slow"]).all()))

    print("\n-- sizing --")
    # The pure risk-sizing assertions below must not be perturbed by a lot floor,
    # so run them against a floor-disabled config. The floor is exercised on its
    # own further down.
    no_floor = dataclasses.replace(
        config, risk=dataclasses.replace(config.risk, min_lot=0.0))
    balance, stop_distance = 10_000.0, 0.0020
    volume = position_size(balance, stop_distance, FX, no_floor)
    realised = FX.money(stop_distance, volume)
    budget = balance * no_floor.risk.risk_per_trade_pct / 100
    check("sized loss stays within the risk budget", realised <= budget + 1e-9,
          f"${realised:.2f} <= ${budget:.2f} at {volume} lots")
    check("sized loss is close to the budget", realised > budget * 0.9,
          f"{realised / budget:.1%} of budget")
    check("max_lot ceiling is enforced",
          position_size(10_000_000.0, stop_distance, FX, no_floor) <= no_floor.risk.max_lot)

    gold_vol = position_size(balance, 20.0, GOLD_TRUE, no_floor)   # $20 stop
    gold_risk = GOLD_TRUE.money(20.0, gold_vol)
    check("gold sizing respects the budget with the TRUE tick value",
          gold_risk <= budget + 1e-9, f"${gold_risk:.2f} at {gold_vol} lots")
    bad_vol = position_size(balance, 20.0, GOLD_REPORTED, no_floor)
    check("the reported tick value would have sized 10x too large",
          bad_vol > gold_vol * 5, f"{bad_vol} vs correct {gold_vol} lots")

    # The floor: a trade whose risk-based size is below min_lot is sized UP to it,
    # deliberately exceeding the per-trade risk budget rather than being skipped.
    floored_cfg = dataclasses.replace(
        config, risk=dataclasses.replace(config.risk, min_lot=0.10, max_lot=0.50))
    floored = position_size(balance, 20.0, GOLD_TRUE, floored_cfg)  # risk maths ~0.02 lots
    check("lot floor sizes up to min_lot", abs(floored - 0.10) < 1e-9,
          f"got {floored} lots")
    check("floored trade knowingly exceeds the risk budget",
          GOLD_TRUE.money(20.0, floored) > budget,
          f"${GOLD_TRUE.money(20.0, floored):.2f} > ${budget:.2f}")
    check("lot ceiling still caps a huge account",
          position_size(10_000_000.0, 20.0, GOLD_TRUE, floored_cfg) <= floored_cfg.risk.max_lot)

    # Fixed lot: overrides risk-based sizing entirely, still clamped to the band.
    fixed_cfg = dataclasses.replace(
        config, risk=dataclasses.replace(config.risk, fixed_lot=0.30,
                                         min_lot=0.10, max_lot=0.50))
    check("fixed lot trades the exact size regardless of risk maths",
          abs(position_size(balance, 20.0, GOLD_TRUE, fixed_cfg) - 0.30) < 1e-9,
          f"got {position_size(balance, 20.0, GOLD_TRUE, fixed_cfg)} lots")
    check("fixed lot is still capped by max_lot",
          position_size(balance, 20.0, GOLD_TRUE,
                        dataclasses.replace(config, risk=dataclasses.replace(
                            config.risk, fixed_lot=0.90, max_lot=0.50))) <= 0.50)

    print("\n-- trade plan --")
    plan = build_plan(Signal.LONG, 0.0010, 1.10000, balance, FX, config)
    check("long plan is a buy", plan is not None and plan.side == "buy")
    assert plan is not None, "build_plan returned None for a valid LONG setup"
    check("long stop below entry, target above",
          plan.stop_loss < 1.10000 < plan.take_profit,
          f"sl={plan.stop_loss} tp={plan.take_profit}")
    check("reward:risk matches the config",
          abs((plan.take_profit - 1.10) / (1.10 - plan.stop_loss)
              - config.risk.take_profit_atr_mult / config.risk.stop_loss_atr_mult) < 1e-6)
    short_plan = build_plan(Signal.SHORT, 0.0010, 1.10, balance, FX, config)
    check("short plan mirrors the long",
          short_plan is not None and short_plan.stop_loss > 1.10)
    check("FLAT produces no plan",
          build_plan(Signal.FLAT, 0.0010, 1.10, balance, FX, config) is None)
    check("zero ATR produces no plan",
          build_plan(Signal.LONG, 0.0, 1.10, balance, FX, config) is None)

    print("\n-- strategy dispatch --")
    for name, params in (("ma_cross", {}),
                         ("donchian", {"entry": 20, "exit_n": 10}),
                         ("smc", {"swing": 3, "min_fvg_atr": 0.25})):
        cfg = dataclasses.replace(
            config, strategy=dataclasses.replace(config.strategy,
                                                 name=name, params=params))
        out = generate_signals(frame, cfg)
        states = set(out["signal"].unique())
        check(f"{name} dispatches and emits valid states",
              states <= {"long", "short", "flat"} and "signal" in out,
              f"states={sorted(states)}")
    try:
        generate_signals(frame, dataclasses.replace(
            config, strategy=dataclasses.replace(config.strategy,
                                                 name="nonsense", params={})))
        check("unknown strategy name is rejected", False)
    except ValueError:
        check("unknown strategy name is rejected", True)

    # Mismatched params must produce an actionable message, not a raw TypeError.
    try:
        generate_signals(frame, dataclasses.replace(
            config, strategy=dataclasses.replace(
                config.strategy, name="ma_cross", params={"entry": 20})))
        check("mismatched strategy.params is rejected clearly", False)
    except ValueError as exc:
        check("mismatched strategy.params is rejected clearly",
              "strategy.params" in str(exc))

    print("\n-- portfolio correlation --")
    check("legs splits an FX pair", legs("EURUSD") == ("EUR", "USD"))
    check("legs strips a broker suffix", legs("EURUSD.a") == ("EUR", "USD"))
    check("legs handles metals", legs("XAUUSD") == ("XAU", "USD"))
    check("legs rejects nonsense", legs("XX") is None)

    # Shorting a negatively-correlated instrument is concentration, not a hedge.
    usdcad_short = Exposure("USDCAD", -1, 5.0)
    audusd_long = Exposure("AUDUSD", +1, 5.0)
    check("effective corr flips sign with direction",
          abs(effective_correlation(-0.65, usdcad_short, audusd_long) - 0.65) < 1e-9,
          f"price -0.65 -> position {effective_correlation(-0.65, usdcad_short, audusd_long):+.2f}")

    # Three short-JPY legs are one big JPY bet however different they look.
    jpy_book = [Exposure("USDJPY", 1, 5.0), Exposure("EURJPY", 1, 5.0),
                Exposure("GBPJPY", 1, 5.0)]
    net = currency_exposure(jpy_book)
    check("currency netting accumulates a shared leg",
          abs(net["JPY"] + 15.0) < 1e-9, f"JPY {net['JPY']:+.2f}")
    check("opposite legs cancel",
          abs(currency_exposure([Exposure("EURUSD", 1, 5.0),
                                 Exposure("EURGBP", -1, 5.0)])["EUR"]) < 1e-9)

    independent = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]], index=["EURUSD", "USDJPY"],
        columns=["EURUSD", "USDJPY"])
    two = [Exposure("EURUSD", 1, 5.0), Exposure("USDJPY", 1, 5.0)]
    quadrature = combined_risk(two, independent)
    check("independent trades add in quadrature",
          abs(quadrature - np.sqrt(50.0)) < 1e-6,
          f"${quadrature:.2f} vs naive $10.00")

    identical = pd.DataFrame(
        [[1.0, 1.0], [1.0, 1.0]], index=["EURUSD", "USDJPY"],
        columns=["EURUSD", "USDJPY"])
    check("perfectly correlated trades add linearly",
          abs(combined_risk(two, identical) - 10.0) < 1e-6,
          f"${combined_risk(two, identical):.2f}")

    # The exact case that motivated this: USDCAD short + AUDUSD long.
    measured = pd.DataFrame(
        [[1.0, -0.65], [-0.65, 1.0]], index=["USDCAD", "AUDUSD"],
        columns=["USDCAD", "AUDUSD"])
    picked = select([audusd_long], [usdcad_short], measured, 1000.0,
                    PortfolioLimits())          # the shipped defaults
    check("the real correlated pair is rejected by the DEFAULT limits",
          len(picked.chosen) == 0 and len(picked.rejected) == 1,
          picked.rejected[0][1] if picked.rejected else "accepted!")
    check("a 0.70 limit would NOT have caught it - why the default is 0.60",
          len(select([audusd_long], [usdcad_short], measured, 1000.0,
                     PortfolioLimits(max_pairwise_corr=0.70)).chosen) == 1)
    relaxed = select([audusd_long], [usdcad_short], measured, 1000.0,
                     PortfolioLimits(max_pairwise_corr=0.95))
    check("...but allowed when the limit is relaxed", len(relaxed.chosen) == 1)

    capped = select([Exposure("EURJPY", 1, 5.0), Exposure("GBPJPY", 1, 5.0)],
                    [Exposure("USDJPY", 1, 5.0)], pd.DataFrame(), 1000.0,
                    PortfolioLimits(max_currency_risk_pct=1.0, max_pairwise_corr=1.0))
    check("currency cap blocks stacking a third JPY leg",
          len(capped.chosen) <= 1,
          f"chose {[c.symbol for c in capped.chosen]}")

    spread_out = select(
        [Exposure("EURUSD", 1, 5.0), Exposure("USDJPY", 1, 5.0)], [],
        independent, 1000.0, PortfolioLimits(max_pairwise_corr=0.7))
    check("independent candidates are both accepted",
          len(spread_out.chosen) == 2)
    check("diversification ratio exceeds 1 for independent trades",
          spread_out.diversification_ratio > 1.3,
          f"{spread_out.diversification_ratio:.2f}x")

    print("\n-- backtester --")
    signalled = generate_signals(frame, ma_config)
    params = BacktestParams(risk_per_trade_pct=0.5, stop_atr_mult=2.0,
                            target_atr_mult=3.0, starting_balance=10_000.0,
                            bars_per_year=96 * 252)
    free = run_backtest(signalled, FX, Costs.free(), params)
    costed = run_backtest(signalled, FX, Costs(spread=0.0002, slippage=0.0), params)

    check("trades were taken", len(free.closed) > 0, f"{len(free.closed)} trades")
    check("every trade is closed", all(t.exit is not None for t in free.trades))
    check("no entry fills before its signal bar",
          all(t.entry_time > frame.index[warmup - 1] for t in free.trades))
    check("equity curve covers every bar", len(free.equity) == len(frame))
    check("costs reduce the result",
          costed.summary()["return_pct"] < free.summary()["return_pct"],
          f"{costed.summary()['return_pct']}% vs {free.summary()['return_pct']}% free")
    check("cost accounting is non-negative",
          all(t.costs >= 0 for t in costed.closed))
    check("no single loss blows far past the risk budget",
          min((t.pnl for t in free.trades), default=0) > -budget * 3,
          f"worst ${min((t.pnl for t in free.trades), default=0):.2f}")
    check("expectancy in R is finite",
          np.isfinite(free.summary()["expectancy_R"]))

    print("\n  free-cost backtest (synthetic data - meaningless as a forecast):")
    for key, value in free.summary().items():
        print(f"    {key.replace('_', ' '):<26} {value}")

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILED: ' + ', '.join(failures)}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
