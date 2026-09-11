"""Pull XAUUSD history from the terminal and cache it to research/data/.

Run once (and again whenever you want fresher bars):

    .venv\\Scripts\\python.exe research\\fetch_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import MetaTrader5 as mt5
import pandas as pd

from swatlas.contract import Contract
from swatlas.timeutil import detect_server_offset, rates_to_frame

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_SYMBOL = "XAUUSD"

TIMEFRAMES = {
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}
MAX_BARS = 200_000


def fetch(symbol: str, label: str, timeframe: int) -> pd.DataFrame | None:
    """Grab the deepest history the terminal will serve for this timeframe."""
    for count in (MAX_BARS, 100_000, 50_000, 20_000, 10_000, 5_000, 1_000):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is not None and len(rates) > 0:
            break
    else:
        return None

    # Server time -> true UTC. Without this, hour-of-day analysis is shifted by
    # the broker's offset (UTC+3 here) and session labels land on wrong bars.
    frame = rates_to_frame(rates).sort_index()
    # Drop the still-forming final bar so cached data is all completed bars.
    return frame.iloc[:-1]


def main() -> int:
    symbols = sys.argv[1:] or [DEFAULT_SYMBOL]
    if not mt5.initialize():
        print(f"Could not connect to the terminal: {mt5.last_error()}", file=sys.stderr)
        return 1
    offset = detect_server_offset()
    print(f"broker server time is UTC{offset.total_seconds() / 3600:+g}; "
          f"bar times will be converted to true UTC")
    for symbol in symbols:
        if fetch_symbol(symbol) != 0:
            mt5.shutdown()
            return 1
    mt5.shutdown()
    return 0


def fetch_symbol(SYMBOL: str) -> int:
    info = mt5.symbol_info(SYMBOL)
    if info is None:
        print(f"{SYMBOL} not found on this broker.", file=sys.stderr)
        return 1
    if not info.visible:
        mt5.symbol_select(SYMBOL, True)
        info = mt5.symbol_info(SYMBOL)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Persist the VERIFIED contract alongside the bars. Research and live must
    # agree on contract maths, and the broker's reported tick value cannot be
    # trusted on every symbol - see swatlas/contract.py.
    contract = Contract.verified(info, mt5)
    print(contract.sanity_report())
    if contract.tick_value_source != "reported":
        print(f"  (broker reported tick_value={info.trade_tick_value}, "
              f"using verified {contract.tick_value})")

    spec_path = DATA_DIR / f"{SYMBOL}_contract.json"
    pd.Series(contract.to_dict()).to_json(spec_path)
    print(f"contract -> {spec_path.name}")

    for label, timeframe in TIMEFRAMES.items():
        frame = fetch(SYMBOL, label, timeframe)
        if frame is None:
            print(f"  {label:<4} no data")
            continue
        path = DATA_DIR / f"{SYMBOL}_{label}.parquet"
        frame.to_parquet(path)
        print(f"  {label:<4} {len(frame):>7} bars  "
              f"{frame.index[0]:%Y-%m-%d} -> {frame.index[-1]:%Y-%m-%d}  -> {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
