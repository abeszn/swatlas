"""Place ONE properly-sized market order, to prove the execution path end to end.

This is a plumbing test, not a strategy. It exercises the real code path -
verified contract maths, ATR sizing, risk budget, stop/target placement,
broker filling-mode negotiation - so that when the bot trades, nothing about
execution is untested.

    # show the plan, send nothing
    .venv\\Scripts\\python.exe scripts\\place_test_trade.py --side buy

    # actually send it
    .venv\\Scripts\\python.exe scripts\\place_test_trade.py --side buy --confirm

    # close what it opened
    .venv\\Scripts\\python.exe scripts\\place_test_trade.py --close --confirm

Refuses to run on a real-money account regardless of flags.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import dataclasses
import logging
import sys

import MetaTrader5 as mt5

from swatlas.config import load_config
from swatlas.indicators import atr
from swatlas.logsetup import setup_logging
from swatlas.mt5_client import MT5Client, MT5Error
from swatlas.risk import build_plan
from swatlas.strategy import Signal


def main() -> int:
    parser = argparse.ArgumentParser(description="Place one sized test trade.")
    parser.add_argument("--side", choices=("buy", "sell"), default="buy")
    parser.add_argument("--symbol", default=None, help="override config.yaml")
    parser.add_argument("--risk-pct", type=float, default=None)
    parser.add_argument("--close", action="store_true",
                        help="close this bot's open positions instead of opening")
    parser.add_argument("--confirm", action="store_true",
                        help="required to actually send anything")
    args = parser.parse_args()

    setup_logging(logging.INFO)
    config = load_config()
    if args.symbol:
        config = dataclasses.replace(config, symbol=args.symbol)
    if args.risk_pct:
        config = dataclasses.replace(
            config, risk=dataclasses.replace(config.risk, risk_per_trade_pct=args.risk_pct))
    # Dry-run unless explicitly confirmed on the command line.
    config = dataclasses.replace(
        config, execution=dataclasses.replace(config.execution, dry_run=not args.confirm))

    try:
        with MT5Client(config) as client:
            account = mt5.account_info()
            if account.trade_mode == mt5.ACCOUNT_TRADE_MODE_REAL:
                print("REAL-money account. This script is for demo only.", file=sys.stderr)
                return 1

            if args.close:
                positions = client.open_positions()
                if not positions:
                    print("\nNo open positions with this bot's magic number.\n")
                    return 0
                for position in positions:
                    print(f"\nClosing #{position.ticket}: {position.volume} lots, "
                          f"P&L {position.profit:+.2f}")
                    client.close_position(position)
                if not args.confirm:
                    print("\n(dry run - pass --confirm to actually close)\n")
                return 0

            contract = client.contract
            frame = client.bars()
            atr_value = float(atr(frame, config.risk.atr_period).iloc[-1])

            tick = mt5.symbol_info_tick(config.symbol)
            if tick is None or tick.ask <= 0:
                print(f"No live tick for {config.symbol}; the market is closed.",
                      file=sys.stderr)
                return 1

            signal = Signal.LONG if args.side == "buy" else Signal.SHORT
            price = tick.ask if args.side == "buy" else tick.bid
            balance = client.account_balance()
            plan = build_plan(signal, atr_value, price, balance, contract, config)

            print(f"\n{'=' * 66}")
            print(f"  account   {account.login} ({account.server})  balance ${balance:,.2f}")
            print(f"  symbol    {config.symbol}   {contract.sanity_report()}")
            print(f"  price     bid {tick.bid} / ask {tick.ask}")
            print(f"  ATR({config.risk.atr_period})    {atr_value:.5f}   "
                  f"stop {config.risk.stop_loss_atr_mult}xATR = "
                  f"{atr_value * config.risk.stop_loss_atr_mult:.5f}")

            if plan is None:
                min_risk = contract.volume_min * contract.loss_per_lot(
                    atr_value * config.risk.stop_loss_atr_mult)
                if config.risk.min_lot > 0:
                    # With a lot floor set, a too-small trade is sized UP, not
                    # skipped - so reaching here means no valid size exists at all.
                    print(f"\n  NO TRADE: the lot ceiling (max_lot {config.risk.max_lot}) "
                          f"is below the broker minimum ({contract.volume_min} lots), "
                          f"so no valid size exists. Raise risk.max_lot to trade "
                          f"{config.symbol}.")
                    print(f"  At the broker minimum this trade would risk "
                          f"${min_risk:,.2f} = {min_risk / balance * 100:.1f}% of balance.")
                else:
                    print(f"\n  NO TRADE: this symbol cannot be sized within a "
                          f"{config.risk.risk_per_trade_pct}% risk budget on a "
                          f"${balance:,.2f} account.")
                    print(f"  The broker minimum ({contract.volume_min} lots) would risk "
                          f"${min_risk:,.2f} = {min_risk / balance * 100:.1f}% of balance.")
                    print(f"  You would need ~${min_risk / (config.risk.risk_per_trade_pct / 100):,.0f} "
                          f"to trade it at {config.risk.risk_per_trade_pct}% risk. "
                          f"Or set risk.min_lot to trade at a fixed floor.")
                print(f"{'=' * 66}\n")
                return 0

            print(f"  PLAN      {plan.side.upper()} {plan.volume} lots")
            print(f"            entry ~{price}  SL {plan.stop_loss}  TP {plan.take_profit}")
            print(f"            risking ${plan.risk_amount:.2f} "
                  f"({plan.risk_amount / balance * 100:.2f}% of balance)")
            print(f"            reward if TP hits: "
                  f"${contract.money(abs(plan.take_profit - price), plan.volume):.2f}")
            print(f"            margin required: "
                  f"${contract.notional(plan.volume, price) / account.leverage:,.2f}")
            print(f"{'=' * 66}\n")

            result = client.market_order(plan.side, plan.volume,
                                         plan.stop_loss, plan.take_profit,
                                         comment="swatlas-test")
            if not args.confirm:
                print("\n(dry run - pass --confirm to actually send this order)\n")
                return 0

            if result.ok:
                print(f"\nFILLED: ticket {result.order}, {result.volume} lots "
                      f"@ {result.price}\n")
                for position in client.open_positions():
                    print(f"  open #{position.ticket}: {position.symbol} "
                          f"{'BUY' if position.type == 0 else 'SELL'} {position.volume} "
                          f"@ {position.price_open}  SL {position.sl}  TP {position.tp}  "
                          f"P&L {position.profit:+.2f}")
                print()
                return 0

            print(f"\nREJECTED: retcode {result.retcode} - {result.comment}\n",
                  file=sys.stderr)
            return 1

    except MT5Error as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
