# Forward test — operational notes (paper, no trading)

## What runs

`research/run_forward_daily.sh` (cron 01:00 UTC):

1. `fetch_okx.py --days 100` → refresh `okx_cache/perp` (funding + perp candles)
   and `okx_cache/spot` (spot candles) from OKX live.
2. `carry_forward_test.py` → append new days to `forward_ledger.csv`
   (idempotent by date; re-running adds nothing).

Both are paper only. Nothing is traded. The book comes from
`basis_strategy.select_legs`, shared with the backtest, so the ledger cannot
diverge from what the backtest measured.

Cron line:
```
0 1 * * * cd /path/to/repo && bash research/run_forward_daily.sh >> forward_test.log 2>&1
```

## Why OKX, not Binance

The Binance research caches (`perp_cache/`, `spot_cache/`) are static and end
2026-09-30. For **fresh** forward data:

- `api.binance.com` / `fapi.binance.com` → **HTTP 451** (geo-blocked here).
- Binance Vision **daily funding** archives do not exist (404) and the monthly
  ones lag several days → no daily funding.
- Binance Vision daily **klines** do exist (perp + spot) but funding is the
  blocker.
- **OKX** is reachable, carries 8h funding, and has both the perp swap and the
  spot pair → it drives the whole forward test. History reaches back ~100 days.

`okx_cache/` is kept separate so the Binance research caches stay pristine and
reproducible.

## Honest forward result (2026-08-08 → 2026-10-09, k=3, hold=126, ~20 bps all-in)

| metric | value |
|---|---|
| hypothetical net | **+17 bps (+0.17%)** |
| winning days | **55%** |
| funding term | positive |

That is **flat**, and it is what the research predicted: 2026 funding has
compressed to ~0, so the carry harvest has nothing to harvest. The forward test
is doing its job — it is confirming the edge is a **funding-regime** effect, not
a steady premium. **Do not deploy on this.** Let it accrue and watch the funding
term, not the noise.

## Freeze

K=3, 42-day rebalance (`--hold 126`). The original directive's 24h rebalance is
reproducible with `--hold 3`; it measures −86%/yr because it churns the carry
away. Documented in `carry_forward_test.py` and `BASIS_RESULTS.md`.
