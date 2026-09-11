"""Verified contract specification.

MetaQuotes-Demo reports `trade_tick_value = 0.1` for XAUUSD while also reporting
`trade_tick_size = 0.01` and `trade_contract_size = 100`. Those cannot all be
true: 0.01 x 100 = $1.00 per tick, and the terminal's own `order_calc_profit`
confirms $100 for a $1/oz move on one lot. The reported field is wrong by 10x.

Position sizing divides by tick value, so trusting that field would have sized
gold positions TEN TIMES too large. Everything that needs contract maths goes
through this class, which verifies the value against the terminal instead of
taking the field at face value.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass

log = logging.getLogger("contract")


def resolve_symbol(base: str, mt5_module) -> str | None:
    """Map a canonical instrument name to whatever this broker actually calls it.

    Broker feeds routinely wrap the plain name in a prefix or suffix -
    EURUSDm, EURUSD.a, EURUSDc, XAUUSD247 - so a symbol list or config.yaml
    written against one broker silently finds nothing the moment you log the
    terminal into a different account. Try the exact name first (cheap, and
    right most of the time), then search the full symbol universe for the
    shortest name that starts with it - the closest thing to "the same pair,
    this broker's spelling". Returns None if nothing matches at all.
    """
    if mt5_module.symbol_info(base) is not None:
        return base
    candidates = mt5_module.symbols_get(f"*{base}*") or ()
    if not candidates:
        return None
    prefixed = [c.name for c in candidates if c.name.upper().startswith(base.upper())]
    pool = prefixed or [c.name for c in candidates]
    return min(pool, key=len)


@dataclass(frozen=True)
class Contract:
    symbol: str
    digits: int
    point: float
    contract_size: float
    tick_size: float
    tick_value: float          # verified, in account currency, per 1.0 lot
    volume_min: float
    volume_max: float
    volume_step: float
    stops_level: int = 0
    tick_value_source: str = "reported"

    # ------------------------------------------------------------ construction

    @classmethod
    def from_symbol_info(cls, info, tick_value: float | None = None,
                         source: str = "reported") -> "Contract":
        return cls(
            symbol=info.name,
            digits=info.digits,
            point=info.point,
            contract_size=info.trade_contract_size,
            tick_size=info.trade_tick_size or info.point,
            tick_value=tick_value if tick_value is not None else info.trade_tick_value,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
            stops_level=info.trade_stops_level,
            tick_value_source=source,
        )

    @classmethod
    def verified(cls, info, mt5_module) -> "Contract":
        """Build a contract whose tick value is proven, not merely reported.

        Order of trust:
          1. `order_calc_profit` - the terminal's own P&L maths, authoritative.
          2. tick_size * contract_size - the definitional identity, valid when
             the profit currency is the account currency.
          3. The reported field, with a loud warning.
        """
        reported = info.trade_tick_value
        tick_size = info.trade_tick_size or info.point
        identity = tick_size * info.trade_contract_size

        calculated = None
        try:
            tick = mt5_module.symbol_info_tick(info.name)
            reference = tick.ask if tick and tick.ask > 0 else None
            if reference:
                move = tick_size * 1000  # big enough to dodge rounding
                profit = mt5_module.order_calc_profit(
                    mt5_module.ORDER_TYPE_BUY, info.name, 1.0, reference, reference + move
                )
                if profit:
                    calculated = profit / (move / tick_size)
        except Exception as exc:  # terminal quirks must not block startup
            log.debug("order_calc_profit unavailable for %s: %s", info.name, exc)

        if calculated and calculated > 0:
            # order_calc_profit round-trips through floats and returns things like
            # 1.0000000000000002. When it agrees with the exact identity, prefer
            # the identity's clean value - same number, no noise in the UI.
            if identity > 0 and math.isclose(calculated, identity, rel_tol=0.01):
                value, source = identity, "order_calc_profit"
            else:
                value, source = round(calculated, 8), "order_calc_profit"
        elif identity > 0:
            value, source = identity, "tick_size*contract_size"
        else:
            value, source = reported, "reported"

        if reported > 0 and not math.isclose(value, reported, rel_tol=0.01):
            log.warning(
                "%s tick value: broker reports %.6f but %s gives %.6f (%.1fx). "
                "Using the verified value - the reported one would have sized "
                "positions %.1fx off.",
                info.name, reported, source, value, value / reported, value / reported,
            )

        return cls.from_symbol_info(info, tick_value=value, source=source)

    @classmethod
    def from_dict(cls, data: dict) -> "Contract":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in fields})

    def to_dict(self) -> dict:
        return asdict(self)

    # ------------------------------------------------------------------- maths

    def money(self, price_move: float, volume: float) -> float:
        """Account-currency P&L for a price move on `volume` lots."""
        return (price_move / self.tick_size) * self.tick_value * volume

    def loss_per_lot(self, distance: float) -> float:
        """What one lot loses if price travels `distance` against the position."""
        return (distance / self.tick_size) * self.tick_value

    def notional(self, volume: float, price: float) -> float:
        """Face value of the position IN ACCOUNT CURRENCY - the base for financing.

        The obvious formula, `volume * contract_size * price`, is wrong whenever
        the contract is not denominated in the quote currency:

          EURUSD  1 lot = 100,000 EUR priced in USD -> 100,000 * 1.155 = $115,500  correct
          USDJPY  1 lot = 100,000 USD priced in JPY -> 100,000 * 158  = $15,800,000
                                                       ...which is 158x too large.

        Deriving it from tick value instead is exact for every case, because
        tick value is already expressed in the account currency. A move equal to
        the entire price IS the notional, so this reduces to `money(price, ...)`.

          USDJPY: (158 / 0.001) * 0.627 * 1 lot = $99,066   ~ $100k  correct
          EURJPY: (183 / 0.001) * 0.631 * 1 lot = $115,473  ~ EUR100k in USD, correct
          XAUUSD: (4345 / 0.01) * 1.00  * 1 lot = $434,500  correct

        Getting this wrong charged 158x the true financing cost on USD-base
        pairs, which turned a mediocre USDJPY backtest into a -96% catastrophe.
        """
        return self.money(price, volume)

    def round_volume(self, volume: float) -> float:
        """Round DOWN to the broker's step - never round risk upward."""
        if self.volume_step <= 0:
            return volume
        steps = math.floor(round(volume / self.volume_step, 8))
        decimals = (max(0, -math.floor(math.log10(self.volume_step)))
                    if self.volume_step < 1 else 0)
        return round(steps * self.volume_step, decimals)

    def round_price(self, price: float) -> float:
        return round(price, self.digits)

    def sanity_report(self) -> str:
        identity = self.tick_size * self.contract_size
        return (f"{self.symbol}: 1.0 lot = {self.contract_size:g} units, "
                f"$1 move = {self.money(1.0, 1.0):,.2f} | tick_value "
                f"{self.tick_value:g} via {self.tick_value_source} "
                f"(identity check {identity:g})")
