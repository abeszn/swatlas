"""Correlation-aware portfolio selection.

A scanner that treats symbols as independent will happily hand you five
positions that are really one bet. Measured on H4 returns, USDCAD and AUDUSD
have a price correlation of -0.65; a USDCAD SHORT next to an AUDUSD LONG is
therefore +0.65 correlated AS POSITIONS - they win and lose together. Summing
their nominal risk understates the real exposure badly.

This module fixes that three ways:

1. **Effective correlation.** Position correlation is price correlation times
   the product of the two directions. Two negatively-correlated instruments
   traded in opposite directions are a concentrated bet, not a hedge.

2. **Combined risk.** Instead of adding risk naively, the portfolio risk is
   `sqrt(r' C r)` over the effective correlation matrix. Independent trades add
   in quadrature (five 0.5% trades = 1.1%, not 2.5%); perfectly correlated
   trades add linearly. The budget is applied to THIS number.

3. **Currency netting.** Every FX pair is a bet on two currencies. Three
   short-JPY legs are one large JPY position however different the symbols
   look, so net per-currency exposure is capped directly.

Existing open positions are included in all of it - a new trade must diversify
against what you already hold, not merely against the rest of the same scan.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd

log = logging.getLogger("portfolio")


@dataclass(frozen=True)
class Exposure:
    """One position or candidate, reduced to what matters for risk."""
    symbol: str
    direction: int          # +1 long, -1 short
    risk: float             # account currency at risk if the stop is hit
    label: str = ""

    @property
    def signed_risk(self) -> float:
        return self.direction * self.risk


@dataclass(frozen=True)
class PortfolioLimits:
    max_total_risk_pct: float = 2.0     # applied to CORRELATION-ADJUSTED risk
    # 0.60, not 0.70. Measured H4 correlations among FX majors run 0.4-0.85
    # (EURUSD/GBPUSD 0.83, AUDUSD/NZDUSD 0.85, USDCAD/AUDUSD -0.65), so a 0.70
    # limit blocks almost nothing. A USDCAD SHORT beside an AUDUSD LONG scores
    # +0.65 as positions and sailed through the looser setting.
    max_pairwise_corr: float = 0.60
    max_currency_risk_pct: float = 1.0  # net exposure to any one currency
    max_trades: int = 4


@dataclass
class Selection:
    chosen: list[Exposure] = field(default_factory=list)
    rejected: list[tuple[Exposure, str]] = field(default_factory=list)
    combined_risk: float = 0.0
    naive_risk: float = 0.0
    currency_risk: dict[str, float] = field(default_factory=dict)

    @property
    def diversification_ratio(self) -> float:
        """naive / combined. 1.0 means no diversification at all; higher is
        better. Five independent trades score ~2.2."""
        return self.naive_risk / self.combined_risk if self.combined_risk else 1.0


def legs(symbol: str) -> tuple[str, str] | None:
    """Split a symbol into (base, quote). EURUSD -> (EUR, USD). XAUUSD -> (XAU, USD).

    Broker suffixes are stripped, so EURUSD.a and EURUSDm behave like EURUSD.
    Returns None for anything that is not a recognisable 6-letter pair.
    """
    name = re.sub(r"[^A-Za-z]", "", symbol).upper()
    if len(name) < 6:
        return None
    return name[:3], name[3:6]


def currency_exposure(exposures: list[Exposure]) -> dict[str, float]:
    """Net signed risk per currency.

    Long EURUSD is +EUR and -USD. Magnitudes are the trade's risk, so a currency
    appearing on the same side of several trades accumulates.
    """
    net: dict[str, float] = {}
    for exposure in exposures:
        pair = legs(exposure.symbol)
        if pair is None:
            continue
        base, quote = pair
        net[base] = net.get(base, 0.0) + exposure.signed_risk
        net[quote] = net.get(quote, 0.0) - exposure.signed_risk
    return net


def effective_correlation(price_corr: float, a: Exposure, b: Exposure) -> float:
    """Correlation between two POSITIONS, not two instruments.

    Shorting a negatively-correlated instrument converts diversification into
    concentration, and this sign flip is exactly what a naive scanner misses.
    """
    return price_corr * a.direction * b.direction


def correlation_matrix(closes: dict[str, pd.Series], min_bars: int = 120
                       ) -> pd.DataFrame:
    """Pairwise log-return correlation across symbols with overlapping history."""
    usable = {s: c for s, c in closes.items() if c is not None and len(c) >= min_bars}
    if len(usable) < 2:
        return pd.DataFrame()
    frame = pd.DataFrame(usable).dropna()
    if len(frame) < min_bars:
        return pd.DataFrame()
    # np.log on a DataFrame is typed as a bare ndarray, which then has no .diff()
    log_prices = pd.DataFrame(np.log(frame.to_numpy()),
                              index=frame.index, columns=frame.columns)
    return log_prices.diff().dropna().corr()


def _pair_corr(corr: pd.DataFrame, a: Exposure, b: Exposure) -> float:
    """Effective correlation, defaulting to a cautious 0.5 when unknown.

    Assuming independence for an unmeasured pair is the optimistic error, and
    the optimistic error is the one that concentrates risk.
    """
    if corr.empty or a.symbol not in corr.index or b.symbol not in corr.columns:
        return 0.5 * a.direction * b.direction if a.symbol != b.symbol else 1.0
    value = cast(float, corr.loc[a.symbol, b.symbol])
    if pd.isna(value):
        return 0.5 * a.direction * b.direction
    return effective_correlation(float(value), a, b)


def combined_risk(exposures: list[Exposure], corr: pd.DataFrame) -> float:
    """Portfolio risk as sqrt(r' C r) using effective correlations.

    Independent trades add in quadrature; correlated ones add closer to linearly.
    """
    if not exposures:
        return 0.0
    n = len(exposures)
    risks = np.array([e.risk for e in exposures])
    matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            c = _pair_corr(corr, exposures[i], exposures[j])
            matrix[i, j] = matrix[j, i] = c
    variance = float(risks @ matrix @ risks)
    return float(np.sqrt(max(variance, 0.0)))


def select(candidates: list[Exposure], existing: list[Exposure],
           corr: pd.DataFrame, balance: float,
           limits: PortfolioLimits | None = None) -> Selection:
    """Greedily pick the candidates that genuinely diversify the book.

    Candidates are considered most-diversifying first, so when the budget binds
    it is spent on independent bets rather than on whichever symbol happened to
    be scanned earliest.
    """
    limits = limits or PortfolioLimits()
    result = Selection()
    accepted = list(existing)

    max_total = balance * limits.max_total_risk_pct / 100.0
    max_currency = balance * limits.max_currency_risk_pct / 100.0

    def worst_corr(candidate: Exposure, against: list[Exposure]) -> float:
        return max((_pair_corr(corr, candidate, other) for other in against),
                   default=0.0)

    remaining = list(candidates)
    while remaining:
        remaining.sort(key=lambda c: worst_corr(c, accepted))
        candidate = remaining.pop(0)

        if len(result.chosen) >= limits.max_trades:
            result.rejected.append((candidate, f"max_trades ({limits.max_trades}) reached"))
            continue

        peak = worst_corr(candidate, accepted)
        if peak > limits.max_pairwise_corr:
            partner = max(accepted, key=lambda o: _pair_corr(corr, candidate, o))
            result.rejected.append((
                candidate,
                f"correlated {peak:+.2f} with {partner.symbol} "
                f"{'LONG' if partner.direction > 0 else 'SHORT'} "
                f"(limit {limits.max_pairwise_corr:.2f})"))
            continue

        trial = accepted + [candidate]
        trial_combined = combined_risk(trial, corr)
        if trial_combined > max_total:
            result.rejected.append((
                candidate,
                f"combined risk ${trial_combined:.2f} would exceed the "
                f"${max_total:.2f} budget ({limits.max_total_risk_pct}%)"))
            continue

        net = currency_exposure(trial)
        breached = [(c, v) for c, v in net.items() if abs(v) > max_currency]
        if breached:
            currency, value = max(breached, key=lambda kv: abs(kv[1]))
            result.rejected.append((
                candidate,
                f"net {currency} exposure ${value:+.2f} exceeds "
                f"${max_currency:.2f} ({limits.max_currency_risk_pct}%)"))
            continue

        accepted.append(candidate)
        result.chosen.append(candidate)

    result.combined_risk = combined_risk(accepted, corr)
    result.naive_risk = sum(e.risk for e in accepted)
    result.currency_risk = currency_exposure(accepted)
    return result
