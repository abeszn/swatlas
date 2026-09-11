"""Minimal type stub for the MetaTrader5 package.

MetaTrader5 is a compiled C extension that ships no type information, so a type
checker sees an empty module and reports every single call - `mt5.initialize`,
`mt5.order_send`, `mt5.ORDER_TYPE_BUY` - as an unknown attribute. That was ~25
false errors across this project, which is enough noise to hide a real one.

The catch-all `__getattr__` below tells the checker "this module has attributes
we cannot describe", which is the truth. It silences the false positives without
suppressing diagnostics globally.

The handful of names declared explicitly are the ones where a wrong type would
actually cause a bug, so they are worth pinning down.
"""

from typing import Any

def __getattr__(name: str) -> Any: ...

# Pinned because a wrong assumption here would be a real defect, not noise.
def initialize(*args: Any, **kwargs: Any) -> bool: ...
def shutdown() -> None: ...
def last_error() -> tuple[int, str]: ...
def symbol_select(symbol: str, enable: bool = ...) -> bool: ...
def order_calc_profit(
    action: int, symbol: str, volume: float,
    price_open: float, price_close: float,
) -> float | None: ...
