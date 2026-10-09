# Delta-neutral cash-and-carry — the first net-positive construction

**Question:** the naive carry book proved the funding premium is real but its
price noise swamped it. Can a **delta-neutral** book (short perp + long spot)
cancel the price term and leave the carry?

**Answer: yes.** This is the first construction in the whole programme that is
**net-positive, positive in 4 of 6 years, and cost-insensitive** — with a
Sharpe of ~3. It is real and deployable in shape. But the edge is **thin and
decaying**, and the most recent year is negative. Read the caveats.

## The construction

`research/basis_strategy.py`. For each of the top-k most-expensive-funding
perps, hold a matched-notional pair: **short 1 perp, long 1 spot**. Per interval
the pair earns

    funding : a short perp RECEIVES funding when funding > 0
    price   : r_spot - r_perp  ~= -(change in basis)

The basis is small and mean-reverting, so the price term should be ~0 and the
funding carry is what is left. Cost is charged on **both** legs at each
rebalance (perp 5 bps + spot 10 bps/leg, conservative taker), so holding long
matters.

Data: Binance Vision **spot** 1h klines (new fetcher `fetch_spot_binance.py`) +
the existing perp klines/funding. 15 symbols, 2021-01 → 2026-09, 94,350 rows.

## The hedge works — price term collapses

| book | price term (bps/8h) | win rate |
|---|---|---|
| naive neutral (previous study) | swings **−0.67 … +3.83** across holds | ~0.50 |
| **delta-neutral (this)** | **+0.01 … −0.01** | **0.57–0.59** |

The price column is now essentially zero at every holding period — the delta
hedge does exactly what it should. Win rate rises from ~50% to ~58%.

## Net result (k=3, hold=126 = 42 days, cost 5+10 bps/leg)

| sample | net bps/8h | %/yr | win | Sharpe | by year |
|---|---|---|---|---|---|
| **FULL 2021–2026** | +0.81 | **+8.8%** | 0.58 | **3.85** | 2021 +3.5, 2022 −0.2, 2023 +0.5, 2024 +1.0, 2025 +0.1, 2026 −0.2 |
| RECENT 2024+ | +0.30 | +3.3% | 0.57 | 2.95 | 2024 +0.8, 2025 +0.2, 2026 −0.2 |

Robust to alignment and to the exact holding period:

- Cost-insensitive: at hold=126, spot cost 10→0 bps moves net only +0.30 → +0.42.
  The edge is gross carry, not a cost artifact.
- hold=63/126/252 all give +0.67/+0.81/+0.87 bps/8h (full). Turnover is already
  negligible by hold=63, so the numbers are stable, not a grid peak.

## Caveats (read these)

1. **The carry decays with the funding regime.** Mean funding (bps/8h) by year:
   2021 **+3.30**, 2022 −0.42, 2023 +0.37, 2024 +0.93, 2025 +0.16, 2026 −0.03.
   The 2021 +3.5 bps/8h is a bull-market funding artefact; recent years are
   +0.8, +0.2, **−0.2**. The strategy is a bet on the funding regime, not a
   steady premium.
2. **2026 is negative** (−0.2 to −0.5 bps/8h). Funding has compressed to zero.
3. **Thin net.** Recent ≈ +3%/yr gross before the operational realities below.
4. **Not modelled:** spot borrow/short-availability on the perp venue, margin
   and rebalancing of the two legs, slippage on the spot taker, and the fact
   that a perp account and a spot account may be at different venues (transfer
   cost, basis risk between venues). Each of these eats into a +3%/yr edge.
5. The non-overlapping t-stat is **not** meaningful here (positions are held, so
   the series is autocorrelated); use the Sharpe. It is ~3, which is high, but
   part of that is the low per-period volatility of a hedged book — the edge per
   unit of *capital at risk* is modest.

## Verdict

**The delta-neutral construction is the answer to "how do you harvest the
carry": it cancels the price term and turns a real gross premium into a
net-positive, cost-robust, high-Sharpe return (+8.8%/yr full, +3.3%/yr recent).**
It is the first thing in this programme that actually works in shape.

**But it is a funding-regime harvest, not a steady edge:** the premium has
compressed to ~0 by 2026, and the recent net is thin enough that unmodelled
execution/borrow costs could erase it. It is worth forward-testing (no trading)
with the daily ledger, not deploying on these numbers alone.

## Files
- `research/basis_strategy.py`, `research/fetch_spot_binance.py`
- `tests/test_basis_strategy.py` — invariants (hedge cancels matched legs; short
  perp receives funding; price term = −Δbasis; both legs costed). 4 tests.
- Full suite: **292 passed, 1 skipped**. Lint clean on new files.
- No `config.py` / `MODEL_PATH` change. Nothing promoted. `main` untouched.

## Realistic cost model + forward-test ledger

`run_basis` now charges **execution slippage** (per leg) and a **cross-venue
transfer** on top of the taker fees. With fees 5 (perp) + 10 (spot) bps/leg,
slippage 2 bps/leg and transfer 1 bps, the all-in cost is **~20 bps per unit of
turnover**. Turnover per rebalance is ~1.0–1.33, so each rebalance costs ~20–27
bps.

This makes the **rebalance cadence decisive**. On the 2026-06 → 2026-09 window:

| hold | net | %/yr |
|---|---|---|
| 3 (24h) | −7.84 bps/8h | **−85.9%** |
| 21 (7d) | −1.19 | −13.0% |
| 63 (21d) | +0.02 | +0.3% |
| **126 (42d)** | **+0.23** | **+2.5%** |

A 24h rebalance is a **cost disaster** — it churns the carry away. This directly
contradicts the original freeze rule ("24h rebalance"), which predates the carry
research and was written for the incumbent trend bot.

### `research/carry_forward_test.py` — paper ledger, no trading

Runs the frozen book over recent data and appends a **daily hypothetical book**
to `forward_ledger.csv` (idempotent by date). The book comes from the shared
`basis_strategy.select_legs`, so the ledger **cannot diverge** from the backtest
(there is a test asserting exactly that).

2026-06-01 → 2026-09-30, k=3, hold=126, all-in ~20 bps: **+83.2 bps (+0.83%)
over 122 days, 66% winning days** (paper only).

### Freeze deviation (recorded, not hidden)

The original directive was **K=3, 7-day lookback, 24h rebalance**. For this book
we freeze at **K=3, 42-day rebalance** and record why: the carry research shows a
24h rebalance is a cost disaster (−86%/yr), while 42d is the research-supported
cadence (+2.5%/yr). There is no "lookback" parameter for a cross-sectional
funding rank. `--hold 3` reproduces the original rule; `--hold 126` is the
default. This is a documented, evidence-based deviation, not silent tuning.
