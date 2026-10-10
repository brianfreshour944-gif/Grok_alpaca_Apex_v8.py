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
| `research/run_donchian_daily.sh` | cron entrypoint (paper only). |
| `tests/test_donchian_breakout.py`, `tests/test_donchian_bot.py` | 22 tests, dependency-isolated. |

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

# honest study
python research/donchian_backtest.py --recent 2024-01-01 --sweep --walk-forward

# paper ledger (nothing traded); idempotent by date
python donchian_bot.py --paper --ledger donchian_ledger.csv

# live is DOUBLE-GATED: --live alone is refused
python donchian_bot.py --live --i-understand-the-risk
```

Live trading maps OKX symbols to Alpaca pairs (`BTCUSDT → BTC/USD`) and only
orders symbols Alpaca actually lists (`ALPACA_TRADEABLE`); research-only symbols
appear in the ledger but are never ordered. Market orders are used for exits,
matching the repo's protective-exit convention.

Cron (paper only), 01:30 UTC:
```
30 1 * * * cd /path/to/repo && bash research/run_donchian_daily.sh >> donchian_daily.log 2>&1
```

## Frozen rule (do not tune without re-doing the walk-forward)

`entry=20, exit=10, ATR=20, risk_pct=1%, stop_atr=2.0`, long-only, fee 25 bps.
These are the textbook defaults, chosen *before* looking at the sweep. Any
change is a new hypothesis and needs its own out-of-sample validation.
