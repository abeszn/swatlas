"""Trading cost model.

MetaQuotes-Demo reports a zero spread, which is fiction. Every backtest in this
project charges costs explicitly through this object so a result can never
silently assume free trading.

Swap is modelled as an ANNUAL PERCENTAGE OF NOTIONAL, not a flat dollar amount
per lot. This matters on a 2004-2026 gold sample: the terminal quotes a flat
-$12.60/lot/night, but one lot was ~$40k of gold in 2004 and ~$435k in 2026.
Applying today's dollar figure across the whole sample charges early years an
absurd 11%/yr financing rate and quietly destroys any long-held position.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Costs:
    """Monetary values in account currency (USD); `spread` in price units.

    `spread` is charged ONCE per round trip: MT5 bars are bid prices, so a buy
    fills at bid+spread and exits at bid, a sell fills at bid and exits at
    bid+spread.

    Swap rates are percent of notional per year, negative when you pay.
    """

    spread: float = 0.25
    slippage: float = 0.0
    commission_per_lot_roundturn: float = 0.0
    swap_long_annual_pct: float = -5.0
    swap_short_annual_pct: float = -1.0

    @classmethod
    def xauusd_retail(cls) -> "Costs":
        """A deliberately unflattering retail gold profile.

        -5%/yr on longs is roughly USD funding plus a broker markup. Shorts are
        charged -1% rather than credited: in practice the markup eats most of the
        credit, and assuming you get paid to be short is the optimistic error.
        """
        return cls(
            spread=0.25,
            slippage=0.05,
            commission_per_lot_roundturn=0.0,
            swap_long_annual_pct=-5.0,
            swap_short_annual_pct=-1.0,
        )

    @classmethod
    def xauusd_tight(cls) -> "Costs":
        """A good ECN broker: tighter spread, explicit commission."""
        return cls(
            spread=0.12,
            slippage=0.03,
            commission_per_lot_roundturn=7.0,
            swap_long_annual_pct=-4.0,
            swap_short_annual_pct=-0.5,
        )

    @classmethod
    def fx_major_retail(cls) -> "Costs":
        """A retail FX major: ~1.2 pip effective spread including slippage.

        Financing on FX is a rate differential, so it can be positive on one
        side. -2%/-2% is a deliberately pessimistic symmetric assumption: it
        charges you on both sides rather than assuming you collect carry.
        """
        return cls(
            spread=0.00010,
            slippage=0.00002,
            commission_per_lot_roundturn=0.0,
            swap_long_annual_pct=-2.0,
            swap_short_annual_pct=-2.0,
        )

    @classmethod
    def fx_jpy_retail(cls) -> "Costs":
        """JPY pairs quote to 3 decimals, so the same pip is 100x the number."""
        return cls(
            spread=0.010,
            slippage=0.002,
            swap_long_annual_pct=-2.0,
            swap_short_annual_pct=-2.0,
        )

    @classmethod
    def free(cls) -> "Costs":
        """Zero costs - only for isolating whether a signal has any raw edge."""
        return cls(spread=0.0, slippage=0.0,
                   swap_long_annual_pct=0.0, swap_short_annual_pct=0.0)
