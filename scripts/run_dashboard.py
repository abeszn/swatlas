"""Start the local dashboard and open it in your browser.

Easiest: double-click `dashboard.bat` in the project root, or the desktop
shortcut. From a terminal:

    .venv\\Scripts\\python.exe scripts\\run_dashboard.py

Binds to localhost only. The dashboard shows account balances and can place and
close trades, so it must not be exposed to the network - there is no auth on it.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import logging
import socket
import threading
import webbrowser

import uvicorn

from swatlas.config import load_config
from swatlas.logsetup import setup_logging
from swatlas.web import create_app


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.4)
        return probe.connect_ex((host, port)) == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Swatlas dashboard.")
    parser.add_argument("--port", type=int, default=8760)
    parser.add_argument("--symbol", default=None, help="override config.yaml")
    parser.add_argument("--no-open", action="store_true",
                        help="do not open a browser window")
    args = parser.parse_args()

    setup_logging(logging.INFO)
    log = logging.getLogger("dashboard")
    url = f"http://127.0.0.1:{args.port}"

    # Starting a second copy would fail with a confusing bind error, and the
    # user's actual intent is almost always "show me the dashboard".
    if port_in_use(args.port):
        log.info("Dashboard is already running at %s - opening it.", url)
        if not args.no_open:
            webbrowser.open(url)
        return 0

    config = load_config()
    if args.symbol:
        import dataclasses
        config = dataclasses.replace(config, symbol=args.symbol)

    print()
    print(f"   Swatlas dashboard   {url}")
    print(f"   watching {config.symbol} {config.timeframe_name} "
          f"({config.strategy.name})")
    print("   Close this window or press Ctrl+C to stop.")
    print()

    if not args.no_open:
        # Fire once the server is actually accepting connections, so the first
        # page load does not race the startup and show a connection error.
        def open_when_ready() -> None:
            for _ in range(40):
                if port_in_use(args.port):
                    webbrowser.open(url)
                    return
                threading.Event().wait(0.25)
        threading.Thread(target=open_when_ready, daemon=True).start()

    # Deliberately NO `workers` argument. Passing it - even workers=1 - makes
    # uvicorn start a multiprocess supervisor that spawns a separate worker,
    # so the dashboard showed up as two python processes and the MT5 connection
    # lived in the child. Omitting it runs the server in THIS process, which is
    # what an app holding a single terminal IPC channel wants.
    uvicorn.run(create_app(config), host="127.0.0.1", port=args.port,
                log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
