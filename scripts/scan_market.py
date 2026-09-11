"""Scan a basket of symbols for tradeable setups, and optionally act on them.

    .venv\\Scripts\\python.exe scripts\\scan_market.py              # report only
    .venv\\Scripts\\python.exe scripts\\scan_market.py --execute    # place orders

The scanning and selection logic lives in swatlas.scanner, shared with the
dashboard, so the command line and the UI cannot drift apart and propose
different trades.

Selection is correlation-aware: candidates are weighed against each other AND
against positions already open, so a scan cannot hand you five positions that
are really one bet. See swatlas/portfolio.py.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import dataclasses
import logging

from swatlas.config import load_config
from swatlas.logsetup import setup_logging
from swatlas.mt5_client import MT5Client, MT5Error
from swatlas.news import NewsCalendar
from swatlas.portfolio import PortfolioLimits
from swatlas.scanner import DEFAULT_BASKET, run_scan


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan symbols for setups.")
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_BASKET)
    parser.add_argument("--execute", action="store_true", help="place the orders")
    parser.add_argument("--max-trades", type=int, default=4)
    parser.add_argument("--risk-budget", type=float, default=2.0,
                        help="max CORRELATION-ADJUSTED %% of balance across the "
                             "whole book, open positions included")
    parser.add_argument("--max-corr", type=float, default=0.60,
                        help="reject a candidate whose effective correlation with "
                             "anything already held exceeds this")
    parser.add_argument("--max-currency-risk", type=float, default=1.0,
                        help="max net %% of balance exposed to any one currency")
    parser.add_argument("--no-news", action="store_true",
                        help="skip the economic-calendar filter")
    parser.add_argument("--risk-pct", type=float, default=None,
                        help="override risk per trade from config.yaml")
    args = parser.parse_args()

    setup_logging(logging.WARNING)   # the table below is the output
    config = load_config()
    if args.risk_pct is not None:
        config = dataclasses.replace(
            config, risk=dataclasses.replace(config.risk,
                                             risk_per_trade_pct=args.risk_pct))

    try:
        with MT5Client(config) as client:
            import MetaTrader5 as mt5
            account = mt5.account_info()
            if account.trade_mode == mt5.ACCOUNT_TRADE_MODE_REAL:
                print("REAL-money account - this script is demo only.")
                return 1

            print(f"\n{account.login} ({account.server})  balance "
                  f"${account.balance:,.2f}  equity ${account.equity:,.2f}")
            if config.strategy.name == "ma_cross":
                detail = f"MA {config.strategy.fast_period}/{config.strategy.slow_period}"
            else:
                detail = f"{config.strategy.name} {config.strategy.params or ''}".strip()
            print(f"strategy: {detail} on {config.timeframe_name}, "
                  f"{config.risk.risk_per_trade_pct}% risk, "
                  f"stop {config.risk.stop_loss_atr_mult}xATR")

            calendar = None
            if config.news.enabled and not args.no_news:
                calendar = NewsCalendar.load(config.news.max_cache_age_hours)
                print(f"news: {len(calendar.events)} events loaded, blocking "
                      f"{config.news.min_impact}-impact from -"
                      f"{config.news.before_minutes}min to +"
                      f"{config.news.after_minutes}min")
            print(f"portfolio: max {args.risk_budget}% correlation-adjusted risk, "
                  f"pairwise corr <= {args.max_corr}, "
                  f"max {args.max_currency_risk}% per currency\n")

            limits = PortfolioLimits(
                max_total_risk_pct=args.risk_budget,
                max_pairwise_corr=args.max_corr,
                max_currency_risk_pct=args.max_currency_risk,
                max_trades=args.max_trades)
            result = run_scan(config, args.symbols, calendar, limits)

            header = (f"{'symbol':<9}{'status':<20}{'signal':>7}{'spread':>8}"
                      f"{'min-lot risk':>14}{'plan':>22}")
            print(header)
            print("-" * len(header))
            for scan in result.scans:
                sig = scan.signal.value.upper() if scan.signal else "-"
                spread = (f"{scan.spread_points} pts"
                          if scan.spread_points is not None else "-")
                minrisk = (f"{scan.min_lot_risk_pct:.1f}% of bal"
                           if scan.min_lot_risk_pct is not None else "-")
                plan = (f"{scan.plan.side.upper()} {scan.plan.volume:g} lots"
                        if scan.plan else "-")
                print(f"{scan.symbol:<9}{scan.status:<20}{sig:>7}{spread:>8}"
                      f"{minrisk:>14}{plan:>22}")
                if scan.news:
                    print(f"{'':<9}  blocked by: {scan.news}")

            actionable = [s for s in result.scans if s.actionable]
            print(f"\n{len(actionable)} actionable of {len(result.scans)} scanned.")
            if not actionable:
                print("Nothing to do.\n")
                return 0

            if result.existing:
                print("\nAlready open ({}): ".format(len(result.existing)) + ", ".join(
                    f"{e.symbol} {'LONG' if e.direction > 0 else 'SHORT'} "
                    f"(${e.risk:.2f} at risk)" for e in result.existing))
            for symbol, side, reason in result.rejected:
                print(f"  rejected {symbol} {side}: {reason}")

            if not result.proposals:
                print("\nNothing passes the portfolio limits.\n")
                return 0

            print(f"\nSelected {len(result.proposals)} trade(s):")
            for scan in result.proposals:
                assert scan.plan is not None
                print(f"  {scan.symbol:<9}{scan.plan.side.upper():<5}"
                      f"{scan.plan.volume:g} lots  SL {scan.plan.stop_loss}  "
                      f"TP {scan.plan.take_profit}  risk ${scan.plan.risk_amount:.2f}")

            s = result.selection
            if s is not None:
                print(f"\nPortfolio risk (including {len(result.existing)} already open):")
                print(f"  naive sum            ${s.naive_risk:.2f} "
                      f"({s.naive_risk / account.balance * 100:.2f}% of balance)")
                print(f"  correlation-adjusted ${s.combined_risk:.2f} "
                      f"({s.combined_risk / account.balance * 100:.2f}% of balance)")
                print(f"  diversification ratio {s.diversification_ratio:.2f}x "
                      f"(1.00 = one concentrated bet)")
                net = {c: v for c, v in s.currency_risk.items() if abs(v) > 0.01}
                if net:
                    print("  net currency exposure: " + ", ".join(
                        f"{c} {v:+.2f}" for c, v in
                        sorted(net.items(), key=lambda kv: -abs(kv[1]))))

            if not args.execute:
                print("\n(scan only - pass --execute to place these orders)\n")
                return 0

            print()
            client.config = dataclasses.replace(
                client.config,
                execution=dataclasses.replace(config.execution, dry_run=False))
            for scan in result.proposals:
                assert scan.plan is not None
                client.use_symbol(scan.symbol)
                outcome = client.market_order(
                    scan.plan.side, scan.plan.volume, scan.plan.stop_loss,
                    scan.plan.take_profit, comment="swatlas-scan")
                state = "FILLED " if outcome.ok else "REJECTED"
                print(f"  {state} {scan.symbol} {scan.plan.side.upper()} "
                      f"{scan.plan.volume:g} @ {outcome.price} "
                      f"{'#' + str(outcome.order) if outcome.ok else outcome.comment}")
            print()
            return 0

    except MT5Error as exc:
        print(f"FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
