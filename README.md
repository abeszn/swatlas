# Swatlas — MetaTrader 5 trading bot

A Python trading bot that uses the MetaTrader 5 terminal purely as an execution
venue. Strategy, risk management, logging, and backtesting live in Python.

**Status: demo only.** The bot refuses to start on a real-money account unless you
deliberately change `execution.allow_live_account` in `config.yaml`, and it dry-runs
by default (real signals and sizing, no orders sent).

---

## Setup

### 1. Install the MetaTrader 5 terminal

Not yet installed on this machine. The Python API is a client that talks to a running
terminal — it cannot reach a broker on its own. Download MT5 **from your broker's own
site** (preferred, so the terminal is pre-pointed at their servers) or from
`metatrader5.com`, and install it yourself.

Then, in the terminal:

- **File → Open an Account** → pick your broker → **Open a demo account**. Note the
  login, password, and the exact server name (e.g. `ICMarketsSC-Demo`).
- **Tools → Options → Expert Advisors** → tick *Allow automated trading*.
- Make sure the **Algo Trading** toolbar button is green.
- Add your symbol to **Market Watch**. Broker symbol names vary — `EURUSD`,
  `EURUSD.a`, `EURUSDm` are all real. Use the exact string in `config.yaml`.

### 2. Credentials

```bash
copy .env.example .env
```

Fill in your demo login, password, and server. `.env` is gitignored. If you leave
them blank the bot attaches to whatever account the terminal is already logged into.

### 3. Verify

The venv already exists with everything installed. Check the whole chain end to end:

```bash
.venv\Scripts\python.exe scripts\check_connection.py
```

This prints the account mode (must say `DEMO`), spread, bar history, and the current
signal. Fix anything it complains about before going further.

---

## Running it

Offline logic test — no terminal needed, verifies indicators, sizing, and backtester:

```bash
.venv\Scripts\python.exe scripts\selftest.py
```

Backtest on history pulled from the terminal:

```bash
.venv\Scripts\python.exe scripts\backtest.py --bars 20000
```

Dry run against live prices (logs what it *would* do, sends nothing):

```bash
.venv\Scripts\python.exe scripts\run_bot.py
```

Live on demo — actually places orders:

```bash
.venv\Scripts\python.exe scripts\run_bot.py --execute
```

Ctrl+C stops the loop. **Open positions stay open** and keep their stop loss and take
profit; close them in the terminal if you don't want them running unattended.

Logs go to the console and to `logs/swatlas.log` (14 days retained).

---

## Dashboard

```bash
.venv\Scripts\python.exe scripts\run_dashboard.py
```

Then open **http://127.0.0.1:8760**. Live account state, open positions with P&L,
closed-trade history, the current signal, and a candle chart with the strategy's
moving averages. Auto-refreshes; no build step, no CDN, works offline.

It surfaces the things that silently break a bot: a REAL-account badge, an
Algo-Trading-off warning, a market-closed notice, and a flag when the broker's
reported contract spec disagrees with the verified one.

**Chart controls.** Pick any **symbol** (Market Watch by default — the broker
lists 12,540 instruments, so the full list is behind a toggle), any
**timeframe** (M1 / M5 / M15 / M30 / H1 / H4 / D1), and a **timezone** (UTC,
your local zone, London, New York, Tokyo, Sydney, Dubai). The timezone applies
to the chart's time axis, the last-closed-bar readout and the closed-trades
table. Changing the chart also repoints the Strategy panel, so you can inspect
any instrument without touching `config.yaml` — the header keeps showing which
symbol the *bot* actually trades.

**A note on times.** MT5 reports bar and tick timestamps in the broker's server
time, not UTC — this one runs at UTC+3. `swatlas/timeutil.py` measures that
offset and converts, so everything displayed is true UTC before your chosen
timezone is applied. Without it charts sat 3 hours in the future and any
hour-of-day analysis landed on the wrong session.

**Trade proposals.** The **Scan** button runs the same scanner as
`scripts/scan_market.py` (shared code in `swatlas/scanner.py`, so the CLI and UI
cannot propose different trades) and shows each surviving setup as a card with
the suggested lot size, stop, target and dollar risk — plus the ones it filtered
out and why. **Approve** is two-step, like Close.

`risk %` and `max corr` are adjustable inline. The **send real orders**
checkbox is **off by default**: approving simulates and reports what would have
happened. Tick it to trade for real.

Nothing about the order is taken from the browser. Approving sends only a
symbol and a direction; the lot size, stop and target are **re-derived
server-side** against current price and volatility, and the trade is rejected
with a 409 if the signal has changed since the scan. A stale tab therefore
cannot place an oversized or reversed trade.

**Closing positions.** Each row has a **Close** button and there is a **Close
all**. Both are two-step: the first click turns the button red and says
`Confirm?`, the second executes; it disarms itself after 8 seconds. There is no
browser pop-up on purpose — embedded webviews suppress `confirm()`, returning
`false` instantly without ever showing a dialog, which made the button appear
to do nothing. Results and failures appear as an in-page banner.

**Localhost only, and there is no authentication.** It displays balances and can
close positions — do not expose the port or put it behind a tunnel.

## Placing a one-off test trade

Proves the whole execution path — verified contract maths, ATR sizing, risk
budget, stop/target placement, filling-mode negotiation — without running the bot.

```bash
.venv\Scripts\python.exe scripts\place_test_trade.py --side buy
```

Shows the plan and sends nothing. Add `--confirm` to actually place it,
`--close --confirm` to close what it opened. It refuses to run on a real-money
account regardless of flags.

---

## Strategies

Selected by `strategy.name` in `config.yaml`. All three run through the same
`generate_signals` used by the backtester, so what you test is what trades.

| Name | What it does | What the research found |
|---|---|---|
| `ma_cross` | MA crossover with an ATR buffer | Break-even at best. The default only because it is the simplest full-pipeline exercise. |
| `donchian` | Channel breakout, trailing exit | Strongest family: 85% of the parameter grid profitable in-sample, beat random entries in **and** out of sample. Still break-even after retail costs. |
| `smc` | Structure (BOS/CHoCH), order blocks, fair value gaps, optional equilibrium | Good in-sample (+0.033 R), **inverted out-of-sample** (−0.034 R). Available, not recommended. |

```yaml
strategy:
  name: donchian
  params: {entry: 20, exit_n: 10}
```

```yaml
strategy:
  name: smc
  params: {swing: 3, min_fvg_atr: 0.25, require_equilibrium: false}
```

**Equilibrium is off by default on purpose.** Requiring a discount/premium
retracement made results *worse than random* (−0.146 R): after a break of
structure, a pullback deep enough to reach discount usually means the trend has
already failed, so the filter selects losers. See [research/RESEARCH.md](research/RESEARCH.md).

Verify the SMC primitives — 30 checks including a lookahead-bias causality test:

```bash
.venv\Scripts\python.exe scripts\verify_smc.py
```

## News filter

Blocks new entries around high-impact economic releases for the traded symbol's
currencies, using the ForexFactory weekly calendar feed. Cached to disk; the
feed is rate-limited.

```yaml
news:
  enabled: true
  min_impact: high
  before_minutes: 30
  after_minutes: 45
```

Open positions are never closed by it — their stops are already broker-side.
It is a **cost control, not an alpha source**: it prevents spread blowout and
slippage, which is why it is not backtested (the backtester assumes constant
spread and is blind to that harm).

## Scanning multiple symbols

```bash
.venv\Scripts\python.exe scripts\scan_market.py
```

Scans a basket, reporting for each: market open? → signal? → sizeable within the
risk budget? → clear of news? Add `--execute` to place the survivors, capped by
`--max-trades` and `--risk-budget`.

### Correlation-aware selection

The scanner does **not** treat symbols as independent. Before selecting, it:

1. **Computes effective correlation** — position correlation is price
   correlation times the product of the two directions. Measured on H4 returns,
   USDCAD and AUDUSD are −0.65 correlated, so a USDCAD SHORT beside an AUDUSD
   LONG is **+0.65 as positions**: they win and lose together.
2. **Budgets correlation-adjusted risk** — `sqrt(r' C r)` rather than a naive
   sum. Independent trades add in quadrature (two 0.5% trades = 0.71%, not 1%);
   correlated ones add closer to linearly.
3. **Caps net currency exposure** — three short-JPY legs are one large JPY
   position however different the symbols look.

Open positions are included in all three, so a new trade must diversify the
whole book rather than just the rest of the scan.

```
Already open (2): USDCAD SHORT ($4.83 at risk), AUDUSD LONG ($4.82 at risk)
  rejected NZDUSD LONG: correlated +0.83 with AUDUSD LONG (limit 0.60)
  rejected EURUSD LONG: correlated +0.75 with AUDUSD LONG (limit 0.60)
  ...
Selected 1 trade(s): USDJPY BUY
  naive sum            $19.57 (1.95% of balance)
  correlation-adjusted  $9.66 (0.96% of balance)
  diversification ratio 2.03x
```

Tunable with `--max-corr`, `--max-currency-risk`, `--risk-budget`.

**Why the default is 0.60, not 0.70:** measured FX-major correlations run
0.4–0.85 (EURUSD/GBPUSD 0.83, AUDUSD/NZDUSD 0.85), so a 0.70 limit blocks
almost nothing — the real USDCAD/AUDUSD pair at +0.65 sailed straight through
it. Expect a tight limit to reject most FX candidates: majors genuinely are
mostly one bet on the dollar, and saying so is the point.

## The baseline strategy

Intentionally simple — it exists so the plumbing is proven end to end, not because
it makes money. Per completed bar on `EURUSD M15`:

- **Long** when the 20-period SMA is above the 50-period SMA *and* close is more than
  0.10×ATR above the slow MA.
- **Short** on the mirror condition.
- **Flat** otherwise — but flat means "no fresh edge", not "exit". Open positions are
  left to their stop or target; only an *opposite* signal closes early and reverses.
- Stop at 2×ATR, target at 3×ATR, size such that the stop costs 0.5% of balance.

The ATR buffer exists because a bare crossover whipsaws itself to death when the MAs
sit on top of each other.

### Risk controls

| Control | Where | Default |
|---|---|---|
| Risk per trade | `risk.risk_per_trade_pct` | 0.5% of balance |
| Lot floor | `risk.min_lot` | 0.10 lots |
| Hard lot ceiling | `risk.max_lot` | 0.50 lots |
| Max concurrent positions | `risk.max_open_positions` | 1 |
| Daily realised-loss halt | `risk.max_daily_loss_pct` | 3% |
| Real-account refusal | `execution.allow_live_account` | false |
| Dry run | `execution.dry_run` | true |

Position size is computed from your risk budget, then clamped into
`[min_lot, max_lot]`. With `min_lot` set above zero, a trade whose risk-based
size falls below the floor is sized **up** to `min_lot` — this **deliberately
risks more than `risk_per_trade_pct`**, and the bot logs the real percentage it
risks on each floored trade. Set `min_lot: 0.0` to restore the conservative
behaviour where such trades are skipped instead of over-risked.

---

## Layout

```
config.yaml              every tunable; nothing is hardcoded in strategy code
.env                     credentials (gitignored)
src/swatlas/
  config.py              typed config loading + validation
  indicators.py          SMA, Wilder ATR — pure pandas, no MT5 import
  strategy.py            signal generation; swap this to change the strategy
  risk.py                position sizing, daily loss guard
  mt5_client.py          all terminal I/O, safety checks, order sending
  engine.py              the live loop
  backtest.py            bar-by-bar replay of the same signal code
scripts/
  check_connection.py    pre-flight diagnostics
  selftest.py            offline logic tests (22 checks)
  backtest.py            backtest CLI
  run_bot.py             live/dry-run CLI
```

`strategy.py` is the file to edit. `generate_signals(frame, config)` is a pure
function from an OHLC DataFrame to a `signal` column, and both the backtester and
the live engine call it — so what you test is what trades.

---

## Backtest honesty

The backtester is deliberately pessimistic:

- Entries fill at the **next** bar's open, never the signal bar's close. Filling at
  the signal bar's close is lookahead bias and is the most common reason a backtest
  looks great and live trading doesn't.
- When a bar's range covers both stop and target, it takes the **stop**. Bar data
  can't tell you which came first, so it assumes the bad one.
- Spread is charged on entry and exit.

Still not modelled: slippage, commission, swap/overnight financing, variable spread,
requotes, weekend gaps. A positive backtest is a reason to forward-test on demo for
a few weeks — not a reason to go live.

A note on what you'll see: running this on synthetic random-walk data produces a
loss, which is the correct result. Any strategy applied to a random walk loses
roughly the spread. If a backtest on real data looks dramatically better than one on
random data, be suspicious of overfitting before you're pleased.

---

## Known limits

- The terminal must stay running on this Windows box; if it closes or the PC sleeps,
  the bot stops managing positions (stops and targets still live on the broker's
  server, so open trades remain protected).
- Single symbol, single position. Multi-symbol means one process per symbol with a
  distinct `magic`, or a rework of the engine loop.
- No reconnect-with-backoff. Transient errors are logged and the cycle is skipped;
  a terminal that dies permanently needs a manual restart.
- The daily-loss guard uses UTC midnight, which is not your broker's server midnight.
