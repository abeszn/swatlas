"""Pre-flight check. Run this FIRST, and any time something stops working.

    .venv\\Scripts\\python.exe scripts\\check_connection.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import sys

import MetaTrader5 as mt5

from swatlas.config import load_config
from swatlas.logsetup import setup_logging
from swatlas.mt5_client import MT5Client, MT5Error
from swatlas.strategy import latest_signal


def main() -> int:
    setup_logging()
    config = load_config()

    print(f"\nConfig: {config.symbol} {config.timeframe_name}, magic {config.magic}")
    print(f"Dry run: {config.execution.dry_run}\n")

    try:
        with MT5Client(config) as client:
            terminal = mt5.terminal_info()
            print(f"Terminal   : {terminal.name} build {mt5.version()[1]}")
            print(f"Connected  : {terminal.connected}")
            print(f"Algo trade : {terminal.trade_allowed}"
                  f"{'' if terminal.trade_allowed else '   <-- enable the Algo Trading button'}")

            info = client.symbol_info
            tick = mt5.symbol_info_tick(config.symbol)
            if tick:
                print(f"\nTick       : bid {tick.bid} / ask {tick.ask} "
                      f"(spread {round((tick.ask - tick.bid) / info.point)} pts)")
            else:
                print("\nTick       : none - market is probably closed")

            frame = client.bars()
            print(f"History    : {len(frame)} completed bars, "
                  f"{frame.index[0]:%Y-%m-%d %H:%M} -> {frame.index[-1]:%Y-%m-%d %H:%M} UTC")

            signal, atr_value = latest_signal(frame, config)
            print(f"Signal now : {signal.value.upper()}  (ATR {atr_value:.5f})")

            positions = client.open_positions()
            print(f"Our trades : {len(positions)} open, "
                  f"realised today {client.realised_pnl_today():+.2f}")

            print("\nAll checks passed.\n")
            return 0

    except MT5Error as exc:
        print(f"\nFAILED: {exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
