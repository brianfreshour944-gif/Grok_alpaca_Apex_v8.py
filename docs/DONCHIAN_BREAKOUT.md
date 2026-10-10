# Donchian-channel breakout — build notes and honest results

Branch: `feat/donchian-breakout` (off `main`). Additive — no change to
`config.py`, `main_bot.py`, or the transformer pipeline.

## What was built

| file | role |
|---|---|
| `donchian_breakout.py` | the strategy: `donchian_channel`, `atr`, `breakout_signal`, `turtle_units`, `backtest_donchian`. Pure numpy/pandas, no config/alpaca import. |
| `donchian_bot.py` | operational shell: daily bars → target book → rebalance plan → paper ledger or (opt-in) live orders via a pluggable broker. |
| `research/fetch_okx_daily.py` | fetch daily OHLCV from OKX (no auth). |
| `research/donchian_backtest.py` | per-symbol + basket backtest, buy&hold baseline, parameter sweep, walk-forward. |
| `research/donchian_improve.py` | enhancement study: base + one lever at a time + combos, OOS fixed-variant walk-forward, risk-normalised comparison. |
| `research/run_donchian_daily.sh` | cron entrypoint (paper only). |
| `tests/test_donchian_breakout.py`, `tests/test_donchian_bot.py` | 34 tests, dependency-isolated. |

## The rules (classic Turtle / Donchian)

- **Entry** — close breaks above the highest HIGH of the prior 20 bars
  (long) or below the lowest LOW (short, off by default).
- **Exit** — close crosses the opposite 10-bar band.
- **Size** — one *unit* is sized so a 1-ATR adverse move costs `risk_pct`
  (1%) of equity: `unit_fraction = risk_pct * price / ATR`.
- **Stop** — protective stop `2 × ATR` behind entry, checked intra-bar.

The band at bar *t* is built from bars `[t-window, t-1]` only, so the close at
*t* is compared against a band that cannot contain itself. Tests pin this
(no-look-ahead is the one thing that must never regress).

## Backtest (real OKX daily bars, 15 symbols, 2020-06 → 2026-10)

Equal-weight basket, entry=20 / exit=10 / ATR=20 / stop=2, **Alpaca taker 25 bps/side**:

| sample | tot% | CAGR | Sharpe | maxDD | trades | win | PF | exposure | B&H |
|---|---|---|---|---|---|---|---|---|---|
| FULL (25 bps) | +84.6 | +10.2 | 0.73 | −22.9% | 594 | 0.32 | 1.87 | 0.39 | +827% |
| FULL (0 bps) | +92.6 | +11.0 | 0.77 | −21.7% | 594 | 0.32 | 1.91 | 0.39 | +827% |
| RECENT 2024+ (25 bps) | +9.1 | +3.2 | 0.38 | −11.7% | 268 | 0.31 | 1.37 | 0.36 | +11.4% |

Reading it honestly:

- **Fees barely matter** (+84.6 → +92.6 with zero fees). The breakout wins on
  gross trend capture, not cost timing. Low turnover (~40 rebalances/symbol over
  6 years) is the reason.
- **The win rate is low (32%) and the profit factor is high (1.9).** This is the
  canonical trend-following shape: many small losers, a few large winners. Do
  not read 32% as "bad" or 1.9 as "great" in isolation.
- **It underperforms buy-and-hold badly on these assets.** The 15-coin
  buy&hold basket is +827% over the same span. A breakout is a *risk-managed*
  way to hold trend, not a way to beat buy&hold on assets that only go up.
- **Recent edge is thin** (+3.2%/yr, Sharpe 0.38) and in line with the rest of
  this programme: crypto edges have compressed since 2024.

## Is the tuned peak real? (sweep + walk-forward)

Sweep of entry × exit (basket total %, 25 bps):

```
   entry\exit    5     10     20     40     55
         10  +51.5    -      -      -      -
         20  +55.8 +84.6    -      -      -
         40  +38.6 +54.3 +94.9    -      -
         55  +32.9 +52.2 +100.3 +88.6   -
         80  +30.5 +49.7  +99.9 +107.3 +103.7
```

A broad plateau at long windows, not a knife-edge. But "pick the best cell"
is exactly the curve-fitting trap, so the walk-forward picks params on each
train fold and reports the *next* fold:

| fold | train-best | train% | test% | test B&H% |
|---|---|---|---|---|
| 1 | (80,40) | +103.1 | −4.2 | −71.4 |
| 2 | (80,10) | −2.4 | +8.0 | +274.7 |
| 3 | (40,20) | +17.3 | +4.1 | +10.6 |
| 4 | (80,10) | +10.0 | −2.2 | −30.6 |

The train-winning params do **not** carry forward: test returns are roughly
flat (−4.2, +8.0, +4.1, −2.2) while buy&hold swings −71% … +275%. The
breakout's real value shows up in that last column's contrast — it sidesteps
the drawdowns — not in a stable alpha. **Do not tune to the sweep peak.**

## Making it earn more: the enhancement study

"Improve for earnings" = lift the P&L. Four standard Donchian/Turtle refinements
were added as opt-in parameters (all OFF at their neutral value, so the base rule
is reproduced exactly — a test pins this):

| lever | what it does |
|---|---|
| `trend_filter=N` | only enter long when close > its N-bar SMA (regime gate) |
| `exit_mode="midpoint"` | exit on the exit band's midpoint instead of its far edge |
| `pyramid_units=K` | add units as the trade runs in favour (Turtle add-units) |
| `trail_atr=X` | ratchet the protective stop behind the run's extreme |

Each was measured on the same 15-symbol OKX panel and, critically, compared
against base on **four out-of-sample test folds** (`research/donchian_improve.py
--walk-forward`). Deltas are base-relative, per fold, in %:

| variant | f1 | f2 | f3 | f4 | mean | folds won | OOS Sharpe |
|---|---|---|---|---|---|---|---|
| trail2 | −4.8 | −15.7 | −12.4 | −3.5 | −9.1 | 0/4 | −1.20 |
| trail3 | −2.5 | −10.8 | −9.9 | −1.7 | −6.2 | 0/4 | −0.85 |
| py2 (pyramid only) | −5.7 | +8.4 | +3.9 | −5.9 | +0.2 | 2/4 | −0.25 |
| midpoint | +6.4 | −6.1 | +2.3 | +3.6 | +1.6 | 3/4 | −0.11 |
| trend100 | +5.0 | +2.3 | +3.3 | +4.1 | +3.7 | 4/4 | −0.07 |
| **trend100+mid+pyr2** | **+7.5** | **+2.1** | **+8.8** | **+4.4** | **+5.7** | **4/4** | **+0.18** |

What the table says, honestly:

- **The trailing stop is a mistake here.** It loses on every fold and destroys
  the Sharpe. It cuts winners short in a market whose edge *is* the fat tail.
  Kept as an option, not enabled.
- **The trend gate is the robust winner.** Beating base on 4/4 folds is the
  strongest signal in this study. It trades less (fewer counter-trend false
  breakouts) without hurting win rate.
- **Pyramiding is a return amplifier, not an edge.** On its own it only wins
  2/4 folds; it mostly scales exposure (drawdown scales too, so ret/DD barely
  moves). It helps only *combined* with the trend gate.
- **`trend100 + midpoint + pyr2` is the one thing that clearly beats base.**
  It is the only candidate with a positive mean OOS Sharpe, wins all four
  folds, and improves both Sharpe (0.73 → 0.79) and return/drawdown
  (3.69 → 5.69) on the full sample. That is what the bot now runs by default.

### Base vs enhanced, full and recent

| config | FULL tot% | Sharpe | maxDD | ret/DD | RECENT tot% | Sharpe | maxDD |
|---|---|---|---|---|---|---|---|
| base (textbook) | +84.6 | 0.73 | −22.9% | 3.69 | +9.1 | 0.38 | −11.7% |
| **enhanced** (trend100+mid+pyr2) | +64.3 | 0.79 | −11.3% | 5.69 | +14.2 | 0.55 | −9.9% |

Note the enhanced config gives up some *absolute* full-sample return but roughly
**halves the drawdown** and improves the recent sample on every measure. That is
the honest trade: better risk-adjusted, not a higher peak. Four folds on one
crypto regime is suggestive, not conclusive — paper-trade it.

## Verdict

A Donchian breakout is a **legitimate, positive-expectancy, regime-robust
trend filter** (profit factor ~1.9 over six years, fee-insensitive), but on
these 15 assets it **underperforms buy-and-hold** and its recent edge is thin
(~+3%/yr). Treat it as a **diversifier / drawdown-control overlay**, not a
profit engine — consistent with every other edge tested in this repo
(`research/*_RESULTS.md`). Paper-trade it; do not deploy on these numbers.

## Running it

```bash
# fetch daily history (no auth)
python research/fetch_okx_daily.py --days 2200

# honest study (base) and the improvement study (enhancements)
python research/donchian_backtest.py --recent 2024-01-01 --sweep --walk-forward
python research/donchian_improve.py --recent 2024-01-01 --walk-forward

# paper ledger (nothing traded); idempotent by date. Defaults to ENHANCED.
python donchian_bot.py --paper --ledger donchian_ledger.csv
python donchian_bot.py --paper --base            # textbook rule instead

# decisions from ALPACA's own daily bars (same keys as every other module) —
# the venue you actually trade on. --source okx (default) keeps research numbers
# reproducible and falls back to the on-disk cache if the network is blocked.
python donchian_bot.py --paper --source alpaca
python donchian_bot.py --paper --source okx --days 400

# live is DOUBLE-GATED: --live alone is refused
python donchian_bot.py --live --i-understand-the-risk
```

Live trading maps OKX symbols to Alpaca pairs (`BTCUSDT → BTC/USD`) and only
orders symbols Alpaca actually lists (`ALPACA_TRADEABLE`); research-only symbols
appear in the ledger but are never ordered. All orders are **market** orders
(entries and exits), matching the backtest's fill assumption. Orders whose
notional falls below Alpaca's crypto minimum (`MIN_ORDER_NOTIONAL`, default $1,
override with `DONCHIAN_MIN_NOTIONAL`) are skipped locally rather than sent to
be rejected. Sizing uses the account's **equity**, not buying power.

Cron (paper only), 01:30 UTC:
```
30 1 * * * cd /path/to/repo && bash research/run_donchian_daily.sh >> donchian_daily.log 2>&1
```

## Frozen rule (do not tune without re-doing the walk-forward)

Band `entry=20, exit=10, ATR=20, risk_pct=1%, stop_atr=2.0`, long-only, fee
25 bps. The **base** rule uses the textbook defaults, chosen *before* looking at
the sweep. The **enhanced** default adds `trend_filter=100, exit_mode="midpoint",
pyramid_units=2` — the only combination that beat base on all four OOS folds, and
therefore the only one promoted. The trailing stop and the SMA-200 gate were
tested and NOT promoted. Any further change is a new hypothesis and needs its
own out-of-sample validation; do not promote a value because it looks good on
the full sample.
