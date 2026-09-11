# XAUUSD strategy research — findings

Research record for the gold strategy design, 2026-08-07. Reproduce with the
scripts in this directory, in order: `fetch_data.py`, `characterize.py`,
`screen.py`, `sweep.py`, `validate.py`.

**Headline: no price-only technical strategy tested here survives realistic
trading costs on gold. The best candidate is statistically indistinguishable
from zero out-of-sample.** The detail below is what makes that conclusion
trustworthy rather than a shrug.

---

## Data

| Timeframe | Bars | Range |
|---|---|---|
| M15 | 49,999 | 2024-06 → 2026-08 |
| H1 | 49,999 | 2018-02 → 2026-08 |
| H4 | 33,940 | 2004-06 → 2026-08 |
| D1 | 5,692 | 2004-06 → 2026-08 |

H4 was chosen as the research substrate: 22 years spans the 2004-2011 bull, the
2011-2015 bear, the 2016-2018 range, and the 2019-2026 bull, so a result cannot
come from one regime.

**Split:** in-sample 2004-2016 for all design and parameter selection;
out-of-sample 2017-2026 looked at exactly once, after freezing.

## A contract-spec bug found before any of this was trustworthy

The broker reports `trade_tick_value = 0.1` for XAUUSD while also reporting
`tick_size = 0.01` and `contract_size = 100`. Those are mutually inconsistent —
`0.01 × 100 = 1.0` — and the terminal's own `order_calc_profit` confirms a $1/oz
move on one lot pays **$100, not $10**.

That field feeds position sizing directly. Left alone it would have sized every
gold position **10× too large live**, and it made the first round of backtests
meaningless (P&L computed at 1/10 scale while financing accrued at full scale).

`swatlas/contract.py` now verifies tick value against `order_calc_profit` rather
than trusting the field, and `scripts/selftest.py` carries a regression test
pinning the correct value.

## 1. Cost structure rules out fast trading

Round-trip spread as a fraction of one ATR, assuming $0.25/oz:

| TF | median ATR | ATR % of price | cost / ATR |
|---|---|---|---|
| M15 | $5.15 | 0.15% | **9.7%** |
| H1 | $4.66 | 0.24% | **10.7%** |
| H4 | $7.08 | 0.52% | 7.1% |
| D1 | $19.37 | 1.40% | 2.6% |

On M15 and H1 you surrender ~10% of a typical bar's range on every round trip.
Intraday gold scalping is not viable at retail spreads — not because the signals
are bad but because of arithmetic.

## 2. Gold is close to a random walk

Autocorrelation of returns is ~0 at every horizon tested (H1/H4/D1, lags 1-24).
Variance ratios on H1: 0.982 (2 bars), 0.972 (8), 1.015 (24), 1.018 (48), 0.964
(120) — all within noise of 1.0.

There is no strong serial dependence to exploit. Any price-only strategy is
working against this.

## 3. Volatility regimes shifted enormously

| Year | Close | Return | ATR % |
|---|---|---|---|
| 2018 | 1280 | −2.7% | 0.175% |
| 2020 | 1894 | +24.6% | 0.297% |
| 2023 | 2063 | +12.8% | 0.212% |
| 2025 | 4319 | **+64.6%** | 0.292% |
| 2026 | 4348 | −0.0% | 0.455% |

Gold roughly tripled across the sample. Two consequences: ATR in *dollars* is
meaningless across years (everything must be volatility-scaled, which the design
does), and **a long-biased system will look brilliant on this sample for reasons
that are not skill**. Hence the buy-and-hold and random-entry controls below.

> **Correction — session hours in §3 are shifted.** MT5 reports bar times in the
> broker's SERVER time, not UTC, and this server runs at **UTC+3**. The cached
> data used for §3 was labelled UTC without converting, so every hour bucket is
> **3 hours late**: the "15-17 UTC" volatility peak is really **12-14 UTC**,
> which is the pre-NY / NY-open window rather than the NY/London overlap as
> labelled. The *shape* of the finding stands — volatility does concentrate in
> one window roughly twice as active as the quiet hours — but the session
> attribution was wrong.
>
> Fixed in `swatlas/timeutil.py`; `research/fetch_data.py` now converts on
> download. Re-run `fetch_data.py` then `characterize.py` for corrected hours.
>
> **No strategy result is affected.** Trend, breakout and structure logic depend
> only on bar ORDERING, and a constant shift does not reorder anything. §4-§11
> stand as written.

## 4. Screening: trend beats random, mean-reversion does not

In-sample, retail costs, expectancy in R per trade:

| Family | best net expectancy | vs random control |
|---|---|---|
| Donchian breakout | **+0.047 R** | clearly better |
| MA crossover | +0.030 R | better |
| Momentum | +0.017 R | marginally better |
| Bollinger fade | −0.099 R | worse |
| RSI fade | −0.088 R | worse |
| *random entries* | −0.085 to −0.138 R | — |

Trend-following contains a genuine directional signal: every trend variant beat
all three random-entry seeds, consistently. Mean-reversion is worse than random —
fading gold loses money reliably.

## 5. Robustness: Donchian is a plateau, momentum is noise

Parameter sweep, in-sample, retail costs:

- **Donchian** (8 entry × 5 exit lengths): **85% of the grid positive**, median
  +0.037 R, smooth surface with one interpretable dead zone (exit ≥ 30 bars gives
  back too much). This is a plateau — the effect tolerates being slightly wrong.
- **Momentum** (7 lookbacks × 4 thresholds): 54% positive, median +0.005 R, no
  coherent structure. Mostly noise.

Donchian was carried forward. Parameters were taken from the **middle** of the
plateau (entry 20 / exit 10), not its peak (entry 100 / exit 20, +0.074 R).
Picking the peak is how you fit noise.

## 6. Out-of-sample: the edge does not survive costs

Frozen design: Donchian 20/10 on H4, stop 3×ATR, trailing 3×ATR, no fixed target,
exit on channel flip, 0.5% risk per trade.

| | trades | exp R | PF | CAGR | Sharpe | maxDD |
|---|---|---|---|---|---|---|
| In-sample 2004-2016 | 604 | +0.047 | 1.13 | +0.92% | 0.31 | −6.1% |
| **Out-of-sample 2017-2026** | **428** | **+0.004** | **1.00** | **+0.03%** | **0.02** | **−7.5%** |
| random seed 0 | 389 | −0.058 | 0.90 | −1.01% | −0.31 | −15.4% |
| random seed 1 | 416 | −0.069 | 0.87 | −1.42% | −0.43 | −18.7% |
| random seed 2 | 443 | −0.023 | 0.96 | −0.42% | −0.11 | −9.7% |
| **gold buy & hold** | — | — | — | **+14.51%** | — | — |

**Statistical verdict on the out-of-sample result:**

- expectancy +0.0040 R, standard error 0.0531 R
- **t-statistic +0.07** (|t| > 2 would be weak evidence of anything)
- 95% confidence interval **[−0.100, +0.108] R** — straddles zero comprehensively
- bootstrap: **48.1% probability of losing money** over 428 trades

Profitable in only **4 of 9** out-of-sample years, and the single good year (2020,
+11.8 R) carries the whole result.

## 7. Why it fails: costs eat the entire edge

Out-of-sample, same signal, varying only cost:

| Cost assumption | exp R | PF | CAGR |
|---|---|---|---|
| Free (no costs) | +0.065 | 1.17 | +0.98% |
| Tight ECN ($0.12 + $7 comm) | +0.023 | 1.06 | +0.37% |
| **Retail ($0.25 + $0.05 slip)** | **+0.004** | **1.00** | **+0.03%** |
| Wide ($0.50) | −0.017 | 0.94 | −0.40% |
| Punitive ($1.00) | −0.076 | 0.82 | −1.25% |

The raw signal is worth **+0.065 R**. Retail costs consume **94% of it**. The
strategy is not wrong about direction — it is right by slightly less than the
broker charges.

This is also the one actionable lever: at tight-ECN pricing the edge is ~6x
larger than at retail. Execution quality matters more here than any signal tweak.

## 8. Risk scaling does not rescue it

| Risk/trade | CAGR | maxDD | Sharpe |
|---|---|---|---|
| 0.5% | +0.03% | −7.5% | 0.02 |
| 1.0% | +0.92% | −16.5% | 0.17 |
| 2.0% | +2.24% | −33.1% | 0.22 |
| 3.0% | +2.57% | −46.3% | 0.22 |

Leverage scales return and drawdown together; Sharpe is flat at ~0.2. There is no
setting at which this becomes attractive. A 33% drawdown to earn 2.2%/yr, against
a 14.5%/yr buy-and-hold, is not a trade worth making.

---

## 9. Smart Money Concepts (BOS / FVG / order blocks / equilibrium)

Implemented mechanically in `src/swatlas/smc.py` with the definitions stated in
the docstrings, then run through the identical harness. `scripts/verify_smc.py`
carries 30 checks including a **causality test**: signals computed on the first
k bars must be byte-identical to the full-series signals over the same region.
If any function reads forward, seeing more data changes an earlier answer. This
is the single most important test here — most impressive SMC backtests fail it,
because centred swing detection marks a swing on the bar it forms rather than
`right` bars later when it could actually be known.

**In-sample screen (H4, 2004-2016, retail costs):**

| Variant | trades | zero-cost CAGR | retail CAGR | exp R | PF |
|---|---|---|---|---|---|
| SMC OB+FVG, trailing exit | 1052 | +4.82% | **+1.32%** | +0.035 | 1.08 |
| SMC OB+FVG, stop/target | 1370 | +3.10% | −0.91% | −0.021 | 0.96 |
| SMC FVG only | 1068 | +2.00% | −1.09% | −0.026 | 0.95 |
| SMC OB only | 904 | +1.51% | −1.55% | −0.050 | 0.90 |
| **SMC + equilibrium filter** | 596 | −1.23% | **−3.06%** | −0.146 | 0.73 |
| *(donchian 20/10, for reference)* | 604 | +2.26% | +0.92% | +0.047 | 1.13 |
| *(random entries)* | ~630 | −1.3 to +0.4% | −2.1 to −3.0% | −0.09 to −0.14 | 0.78-0.84 |

At zero cost SMC carried MORE raw signal than Donchian (+4.82% vs +2.26% CAGR)
and comfortably beat random. Parameter sweep: 71% of the grid positive, with an
interpretable boundary — swing lengths 2-5 work, 6+ fail because structure
updates too late to be actionable.

**Out-of-sample (2017-2026), frozen at swing 3 / min FVG 0.25 ATR:**

| | trades | exp R | PF | CAGR | Sharpe | maxDD |
|---|---|---|---|---|---|---|
| In-sample | 952 | **+0.033** | 1.07 | +1.09% | 0.22 | −12.2% |
| **Out-of-sample** | 659 | **−0.034** | 0.91 | −1.19% | −0.23 | −21.6% |
| random controls | ~415 | −0.023 to −0.069 | 0.87-0.96 | negative | — | — |

t-statistic **−0.61**, 95% CI **[−0.141, +0.074] R**, **74.2% chance of losing
money** over 659 trades. Profitable in 4 of 9 years.

**The sign inverted between in-sample and out-of-sample.** That is the textbook
signature of a fitted result, and out-of-sample SMC is indistinguishable from
random entries. Even at zero cost it only managed +0.041 R out-of-sample.

### The equilibrium finding

This is the most interesting negative result. Requiring price to be in discount
(for longs) or premium (for shorts) cut trades from 1370 to 596 and made
performance dramatically **worse** — −0.146 R, worse than random entries.

The mechanism is structural, not a bug. After a break of structure, price sits
at the top of its dealing range by construction. Requiring a pullback past the
50% level before buying selects specifically for retracements deep enough that
the break has usually already failed. The filter systematically picks losers.

So: **keep equilibrium optional and off by default.** The instinct to use it
"only when needed" is right; the data says "needed" is rarer than the teaching
implies.

## 10. News filter

`src/swatlas/news.py`, sourced from the ForexFactory weekly JSON feed. Blocks
new entries in a configurable window around high-impact releases for the
symbol's currencies (EURUSD → EUR+USD, XAUUSD → USD, since there is no
metal-specific release). Verified firing correctly around US CPI: EURUSD and
XAUUSD blocked −30/+45 min, EURGBP untouched.

**It is deliberately not backtested, and that is defensible.** The feed serves
the current week only, so there is no 2004-2026 calendar to replay. More
importantly a news filter is a **cost control, not an alpha source** — what it
prevents is spread blowout, slippage, and gap-through-stop, and this backtester
assumes a *constant* spread. It is structurally blind to the exact harm the
filter avoids. Backtesting it would show "fewer trades, slightly less edge
captured" and hide the entire benefit. Judge it live on execution quality.

Existing positions are never closed by the filter — their stops already sit on
the broker's server, and pulling them mid-release is usually worse than
sitting still.

## 11. Does any of it transfer to what a $1,000 account can trade?

This is the question that actually decides the live configuration. Gold needs
~$17,000 to size a 0.5% risk trade at the broker minimum, so on a small balance
the only tradeable instruments are FX majors.

Both designs, **frozen on gold and transferred unchanged** (no retuning — that
would just refit), against retail FX costs:

| TF | Symbol | Donchian 20/10 | SMC 3/0.25 | random | verdict |
|---|---|---|---|---|---|
| H4 | EURUSD | −0.128 | −0.049 | −0.021 | both worse than random |
| H4 | GBPUSD | −0.068 | −0.085 | −0.045 | both worse than random |
| H4 | AUDUSD | −0.131 | −0.150 | −0.087 | both worse than random |
| H1 | EURUSD | −0.090 | −0.152 | −0.097 | ~random |
| H1 | GBPUSD | −0.077 | −0.089 | −0.095 | ~random |
| H1 | USDJPY | **−0.001** | −0.027 | −0.103 | clearly beats random, ~break-even |

Expectancy in R. **0 of 7 combinations was profitable for any design.**

The gold trend edge does **not** transfer to FX majors. That is unsurprising in
hindsight: gold is a commodity with genuine trend persistence, while FX majors
are the most heavily arbitraged instruments in existence.

### A cost-model bug found here

USDJPY initially showed −6.17 R per trade with a −96% drawdown, which is not a
result, it is a defect. `Contract.notional()` computed
`volume * contract_size * price`. That is right when the contract is
denominated in the quote currency (EURUSD: 100,000 EUR priced in USD) and wrong
when the base currency *is* the account currency (USDJPY: 1 lot is 100,000 USD,
not 100,000 × 158). Financing was being charged at **158x** the true rate.

Fixed by deriving notional from tick value, which is already in account
currency: `notional = (price / tick_size) * tick_value * volume`. Exact for
majors, crosses and metals alike. Gold and the USD-quoted pairs were unaffected
(the formulas coincide there), so §1-10 above stand unchanged. `selftest.py`
now pins all three cases.

## Conclusions

1. **Do not trade this design with real money.** Out-of-sample it is a coin flip
   after costs, on a sample where gold tripled.
2. **Trend-following on gold does contain a real signal** — it beat random entries
   in-sample and out-of-sample, consistently, across a broad parameter plateau.
   The signal is just smaller than retail transaction costs.
3. **Mean-reversion on gold is actively harmful.** Fading moves lost to random
   entries in every configuration tested.
4. **Intraday gold is off the table** at retail spreads: ~10% of ATR per round
   trip on M15/H1.
5. **The honest benchmark is buy-and-hold**, which returned 14.5%/yr out-of-sample
   with no execution risk, no overnight financing, and no code.

## Where the remaining leverage actually is

Ranked by expected impact, not by how interesting they are to build:

1. **Cheaper execution.** Retail → tight ECN moves expectancy from +0.004 to
   +0.023 R. Nothing else tested moves the number that much.
2. **Trade less, hold longer.** Cost is per round trip; edge accrues per unit of
   trend captured. D1 with long lookbacks had the lowest cost drag (2.6% of ATR).
   The D1 out-of-sample transfer showed +0.157 R on 25 trades — too few to trust,
   but the right direction.
3. **Diversify across instruments.** One symbol at Sharpe 0.2 is noise; ten
   uncorrelated symbols at Sharpe 0.2 is a portfolio. This is how managed-futures
   funds make weak trend signals work, and it is the single most proven route.
4. **Non-price information** — COT positioning, real yields, DXY, ETF flows. Gold
   responds to real rates far more than to its own chart. This leaves technical
   analysis behind entirely, which is the point.

What is *not* worth doing: tuning Donchian parameters further, adding indicator
filters, or trying more oscillators on M15. The variance-ratio and
autocorrelation results say there is very little there to find, and the cost
table says anything found would have to be ~15x larger than typical to matter.
