"""Run the live loop.

    .venv\\Scripts\\python.exe scripts\\run_bot.py            # dry run per config.yaml
    .venv\\Scripts\\python.exe scripts\\run_bot.py --execute  # actually send orders

--execute only lifts the dry-run flag. The demo-account guard is separate and
still applies: a REAL account is refused unless execution.allow_live_account
is set to true in config.yaml.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import dataclasses
import logging
import sys

from swatlas.config import load_config
from swatlas.engine import Engine
from swatlas.logsetup import setup_logging
from swatlas.mt5_client import MT5Client, MT5Error


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Swatlas MT5 bot.")
    parser.add_argument("--execute", action="store_true",
                        help="send real orders instead of dry-running")
    parser.add_argument("--config", default=None, help="path to a config.yaml")
    parser.add_argument("--debug", action="store_true", help="verbose logging")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.debug else logging.INFO)
    config = load_config(args.config)

    if args.execute:
        config = dataclasses.replace(
            config,
            execution=dataclasses.replace(config.execution, dry_run=False),
        )

    try:
        with MT5Client(config) as client:
            Engine(client, config).run()
        return 0
    except MT5Error as exc:
        logging.getLogger("main").error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
