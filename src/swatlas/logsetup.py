"""Logging to both the console and a daily rotating file under logs/."""

from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler

from .config import PROJECT_ROOT

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-18s %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    formatter = logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Keep two weeks of history so a bad session can be reconstructed after the fact.
    file_handler = TimedRotatingFileHandler(
        log_dir / "swatlas.log", when="midnight", backupCount=14, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
