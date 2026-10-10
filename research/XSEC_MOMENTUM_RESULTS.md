# Cross-sectional momentum — a genuinely different signal SOURCE

Everything tested before this is a **time-series** rule on one price series
(Donchian breakout, ML direction, funding timing). Those all share the same
failure mode: they make a directional call and inherit the market's beta.

This tests a **different mechanism**: rank the whole universe each day and hold
the *relative* winners against the *relative* losers. The book is
**dollar-neutral**, so market direction cancels and the only thing traded is
**cross-sectional dispersion**. It is a beta-free bet that a breakout cannot
express.

Module: `research/xsec_momentum.py`. Data: the 15-symbol OKX daily panel
(2020-06 → 2026-10). No look-ahead: ranks at bar *t* use closes up to *t*, the
book earns *t+1*; weights drift between rebalances and costs are charged on the
traded notional at every rebalance.

## Headline config (lookback 30d, k=3, hold 7d, 25 bps/side)

| | total | Sharpe | t | maxDD | beta | turnover/day |
|---|---|---|---|---|---|---|
| **net** | **+202.5%** | **+0.97** | **+2.43** | −38.3% | **+0.01** | 0.117 |
| gross | +489.1% | +0.97 | +2.43 | −35.0% | +0.01 | 0.117 |

- **Market-neutral as designed**: beta vs the equal-weight market is **+0.01**.
- **Significant against a real null**: a random-ranking shuffle (200 trials)
  gives mean Sharpe −0.05, sd 0.47; the real +0.97 is **p = 0.010**.

## Per year (net, lookback 30 / hold 7)

| 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| +70.1 | +11.0 | +23.8 | +11.6 | −11.2 | −21.4 |

Positive through 2024, then **negative in 2025–2026** — the same regime decay as
everything else in this repo.

## Robustness (this is the important part)

Sharpe grid, net, k=3 (`--scan`):

| look\hold | 1 | 3 | 7 | 14 | 30 |
|---|---|---|---|---|---|
| 10 | +1.32 | +0.71 | +0.60 | +0.51 | +0.33 |
| 20 | +1.14 | +0.82 | +0.41 | +0.34 | −0.09 |
| 30 | +1.07 | +0.79 | +0.97 | +0.64 | −0.21 |
| 60 | +0.88 | +0.89 | +0.37 | +0.37 | +0.14 |
| 90 | +0.88 | +0.63 | +0.82 | +0.33 | +0.41 |

**92% of the 25 cells have Sharpe ≥ 0** (mean +0.60, median +0.63). Compare the
Donchian result, where neighbouring parameters flipped the sign of the total
return. This is a much broader plateau — the signature of a real effect rather
than a tuned peak.

The `hold=3` row is the most stable region (all lookbacks positive):

| lookback | 10 | 20 | 30 | 60 | 90 |
|---|---|---|---|---|---|
| net total % | −42.9 | +14.5 | +34.2 | +169.4 | +32.9 |
| Sharpe | +0.71 | +0.82 | +0.79 | +0.89 | +0.63 |
| null p | — | 0.025 | 0.025 | 0.015 | — |

Sub-periods (lookback 60 / hold 3): 2020-22 +277% (Sharpe 1.42), 2023-24 +24%
(0.98), 2025-26 −30% (−0.28). The effect was strong while the post-2020 bull
regime ran and has faded with it.

Cross-sectional **reversal** (short the winners) loses badly — Sharpe −0.88 —
confirming the sign is right: it is momentum, not reversal.

## Cost is the whole game (again)

| hold | turnover/day | gross | net | cost drag |
|---|---|---|---|---|
| 1 | 0.354 | +699.7% | +7.2% | 692 pp |
| 3 | 0.192 | +298.5% | +34.2% | 264 pp |
| 7 | 0.117 | +489.1% | +202.5% | 287 pp |

A daily rebalance churns the edge away; the same lesson the funding-carry study
found. Hold long, rebalance rarely.

## Honest verdict

This is the **most promising distinct source found so far**: the only one that
is **market-neutral**, clears a **real null** (p ≈ 0.01–0.025), and sits on a
**broad robust plateau** rather than a tuned peak. That is a meaningfully
different and better evidence profile than the time-series rules.

But it shares the repo-wide decay: the edge is concentrated in the 2020–2024
regime and is **negative in 2025–2026**.

Caveats that must not be waved away:

- **Survivorship**: 15 majors that exist *today*. Assets that died are absent,
  which flatters a long-the-winners factor.
- **Concentration**: k=3 per side; a single blowup in a short can dominate.
- **Unmodelled costs**: short borrow/funding, cross-venue slippage, and the
  actual dispersion of crypto liquidity. Only the 25 bps taker is charged.
- **One venue, one universe.** Not evidence of a tradable live edge.

Do not deploy. Paper-forward-test it, and re-run the null as months accrue.

Run:
```bash
python research/xsec_momentum.py --lookback 30 --k 3 --hold 7
python research/xsec_momentum.py --scan          # robustness grid
python research/xsec_momentum.py --reverse       # shows it is momentum
```
