# Perp-venue rebuild: better inputs, dollar-neutral book — result

**Question:** does the "better inputs on a perp venue" direction produce a
tradeable, statistically robust edge?

**Answer: no — not for a simple factor book.** The rich perp inputs (relative
strength, funding/carry, order-flow) do carry real cross-sectional information,
but it is **regime-decaying and not statistically significant in the recent
era**. A dollar-neutral factor book at realistic perp costs is ≈0 in 2024+.

This is the second independent negative, and it is consistent with the first:
the incumbent's ceiling is not the model, and it is not merely "use a perp
venue" either. The edge that existed in 2021–2023 has been arbitraged down.

## Data (real)

`research/fetch_perp_binance.py` pulls **Binance USDⓈ-M perpetual** history from
Binance Vision (api.binance.com is geo-blocked; Binance.US is spot-only):

- 15 symbols × 1h klines, 2021-01 → 2026-09 = **~50,000 bars each**.
- 8-hourly funding-rate history per symbol.
- Klines carry **taker-buy volume** → order-flow imbalance.

| | |
|---|---|
| symbols | BTC ETH BNB SOL XRP ADA DOGE LTC LINK DOT AVAX BCH TRX ATOM ETC |
| panel | **755,040** symbol-hours |
| span | 2021-01-01 → 2026-09-30 |
| cost model | 5 bps/leg taker (Binance perp VIP0 ~4.5 taker / 2 maker) |

## Book construction

Dollar-neutral, at each hourly rebalance: long top-k by score, short bottom-k,
equal weight per side (gross long = gross short = 1). Forward return over H
hours. Net = Σ wᵢrᵢ − cost·turnover. `research/perp_strategy.py`.

## Factor results — full history, H=8h, k=3, 5 bps

| factor | sign | gross | net | win | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|
| momentum_24_xs | + | +8.9 | +5.1 | .48 | **28.2** | 3.7 | 3.0 | 4.8 | −6.6 | −5.0 |
| momentum_168_xs | + | +9.6 | +8.1 | .50 | **26.1** | 8.0 | 7.8 | 6.8 | −2.0 | −0.1 |
| reversal_flow_24 | − | +5.0 | +4.9 | .49 | **18.7** | 5.6 | −0.7 | 7.4 | −4.6 | 2.6 |
| carry_short_funding | − | +6.2 | +5.1 | .51 | **26.3** | −0.8 | 4.0 | −4.2 | 4.3 | −0.2 |
| carry_short_funding_z | − | +4.8 | +3.3 | .51 | 9.9 | 4.4 | 2.0 | −1.8 | 6.1 | −2.2 |
| vol_defensive | − | −7.2 | −8.5 | .52 | −37.6 | −2.3 | −8.7 | −2.4 | 2.8 | −0.7 |
| **COMBO** (mom168+carry+flow) | + | +8.5 | +6.9 | .49 | **34.7** | 4.3 | 4.3 | −3.3 | −0.8 | 0.4 |

**Every momentum/flow factor is strongly positive in 2021 and decays to ≈0 or
negative by 2025–2026.** This is crowding, not a persistent premium. Carry
(short funding) is the most persistent but small (≈4–5 bps) and noisy.

## The recent regime is what matters — and it is ≈0

Robustness sweep, **2024-01 onward**, net bps/rebalance:

| H (h) | k | combo | mom168 | flow | carry |
|---|---|---|---|---|---|
| 8 | 3 | −1.4 | +1.7 | +1.7 | −0.0 |
| 24 | 3 | −2.2 | +3.4 | +4.7 | −0.1 |
| 72 | 2 | −2.5 | +5.4 | **+23.3** | +5.0 |
| 72 | 3 | +0.4 | +10.1 | **+18.3** | +3.3 |
| 72 | 4 | +3.5 | +5.0 | +4.6 | +4.9 |

One cell looks interesting — **flow reversal at a 72h hold** (+18–23 bps). It
does **not** survive scrutiny:

- **Quarterly swings:** 2024Q1 −90, 2024Q3 +71, 2024Q4 **+322**, 2025Q2 −81,
  2025Q4 **−111**, 2026Q3 −6. All the mean comes from one blow-off quarter.
- **Win rate 47%** (below a coin flip — fat right tail, not consistency).
- **Non-overlapping t-stat +1.06** (n=334). Not significant.
- mom168 t=+0.68; combo t=−0.38.

The permutation null gives p=0.000 for these factors, but that only says *some*
cross-sectional structure exists — it does **not** say the structure is
tradeable at cost in the current era. The t-stat, not the null, is the gate
that matters, and it fails.

## Verdict

**No deployable edge.** A dollar-neutral perp factor book built from exactly the
"better inputs" the earlier research called for lands at ≈0 net in 2024+ with no
statistically significant result. Combined with the spot result (retraining the
11 features → +4 bps gross, dead at 50 bps), both authorised directions are now
tested and both are negative.

What this does **not** say: that no perp strategy works — only that this simple,
transparent factor book does not, at these costs, in this era. It also does not
close the door on *slower* carry harvesting (the funding-carry branch) which
trades infrequently and was the most persistent signal here.

## Files

- `research/fetch_perp_binance.py` — Binance Vision perp klines + funding.
- `research/perp_strategy.py` — panel build, dollar-neutral book, per-factor and
  combined evaluation, recent-regime cut, permutation null.
- `tests/test_perp_strategy.py` — book invariants (dollar-neutrality, long/short
  selection, cost monotonicity). 6 tests pass with `test_oos_eval.py`.
- `perp_cache/` — ~50k bars × 15 symbols + funding (gitignored).
- **No `config.py` / `MODEL_PATH` change. Nothing promoted. `main` untouched.**
