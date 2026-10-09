# Funding-carry harvesting — the most promising, and its honest ceiling

**Question:** the perp factor study found carry (short funding) to be the most
*persistent* signal. Does a low-turnover funding-carry book clear cost in the
recent era?

**Answer: the carry premium itself is real, large and highly significant — but a
naive market-neutral book cannot realise it, because the book's price noise is
an order of magnitude larger than the carry it harvests.**

This is the first genuinely significant *gross* signal in the whole research
programme. It is not, on its own, a deployable strategy.

## The setup

`research/carry_strategy.py`. At each 8h funding stamp, short the top-k
most-expensive-funding perps and long the bottom-k, dollar-neutral. A position
opened at `ts` is credited the **next** funding stamp (`funding_fwd`) — no
look-ahead. Cost is charged on turnover, which falls as the holding period
(`hold`) lengthens.

Mean funding, 2021–2026 (bps/8h): LINK +1.29, LTC +1.20, XRP +1.19, DOGE +1.17,
ADA/ETH +1.05, BTC +0.99 … BNB −0.05. Perps are structurally long-crowded, so
funding is positive on average and the *dispersion* between symbols is the
harvestable signal (~2 bps/8h between extremes).

## Rebalancing every 8h destroys it

At `hold=1`, k=3, recent (2024+): **gross carry +2.2 bps but turnover 1.59 per
interval → ~8 bps cost → net −6.0 bps/8h.** The churn is 3–4× the carry. Holding
longer is the whole game:

| hold (intervals) | 1 | 3 | 9 | 21 | 42 | 90 |
|---|---|---|---|---|---|---|
| net bps/8h (cost 5) | −6.03 | −2.50 | +1.53 | **+4.08** | +1.08 | −0.00 |
| net bps/8h (cost 0) | +1.91 | +1.29 | +2.92 | **+4.69** | +1.40 | +0.15 |

At the optimum the net is **almost cost-insensitive** (5→0 bps moves it only
+0.6) — proof the edge is gross carry, not a cost-timing artifact.

## Why the "+4.08" is not the answer

It is a **grid artifact**, and the diagnostics say so:

| k | hold | carry | price | net | non-ov. t |
|---|---|---|---|---|---|
| 3 | 9 | +1.07 | +1.85 | +1.53 | −1.65 |
| 3 | **21** | +0.87 | +3.83 | **+4.08** | **+0.38** |
| 3 | 30 | +0.79 | −0.67 | −0.34 | −1.05 |
| 4 | 21 | +0.74 | +2.59 | +2.77 | +0.21 |
| 5 | 21 | +0.61 | +2.78 | +2.88 | −0.12 |
| 6 | 21 | +0.54 | +2.83 | +2.91 | −0.84 |
| 7 | 21 | +0.48 | +3.00 | +3.08 | −0.54 |

- The **carry** column is stable and *hugely* significant (t ≈ **+37**) across
  every cell — the funding premium is unambiguously real.
- The **price** column is the problem: it swings −0.67 to +3.83 across adjacent
  holds, has no structure, and makes the net non-significant (t ≈ 0, ranging
  −1.97 to +0.38).
- Widening the book (k=5–7) to dilute idiosyncratic noise does **not** rescue it.

So the +4 bps net at hold=21 is the price-PnL noise landing positive, not a
carry effect. Pick a neighbouring hold and it disappears.

## Verdict

**Funding carry is a real, significant, cost-robust gross premium (~+8–9%/yr on
a neutral book), but it is not harvestable by this simple book** — the neutral
book's price PnL variance dominates the carry at every holding period tested.

To actually harvest it you need the price component to be *neutralised*, not
merely diversified: e.g. a delta-matched construction, a basis/perp-vs-spot
hedge, or trading funding directly against an index. That is a larger build and
a different project. It is also the **only** lead in this whole programme with a
significant gross edge, so it is the one worth pursuing if the goal is real.

## Files
- `research/carry_strategy.py` — panel, no-lookahead funding, hold/cost sweep.
- `tests/test_carry_strategy.py` — invariants (short receives funding; earned
  cash comes from `funding_fwd`; turnover falls with hold; cost monotone). 4 tests.
- Full suite: **288 passed, 1 skipped**. Lint clean.
- No `config.py` / `MODEL_PATH` change. Nothing promoted. `main` untouched.
