# Can the cross-sectional signal be traded on ALPACA? (spot crypto only)

**Short answer: the market-neutral book cannot. Alpaca crypto is spot-only.**

Alpaca's own crypto asset object says it directly:

```json
{ "symbol": "BTC/USD", "class": "crypto", "tradable": true,
  "marginable": false, "shortable": false, "easy_to_borrow": false }
```

There is **no short leg, no margin, no borrow** for crypto. Alpaca's short
selling / margin features and its Hard-to-Borrow "locates" are **equities-only**
(they require a $2k equity account and do not apply to crypto). So the
dollar-neutral construction in `XSEC_MOMENTUM_RESULTS.md` is **not placeable**
on the platform you are actually using. This is also why the repo's existing live
Donchian path is long-only — same constraint, already respected.

## What *is* placeable: a long-only momentum tilt

Hold the top-k relative winners equal-weight, cash otherwise. It keeps full
market **beta**, so the honest benchmark is not zero — it is **buy & hold**.
`research/alpaca_xsec.py` runs this on the **Alpaca-tradeable** names only
(`ALPACA_TRADEABLE`: 10 of the 15 panel symbols).

Common-history subset (7 names: ADA, BCH, BTC, DOGE, ETH, LINK, LTC),
2020-06 → 2026-10, 25 bps:

| strategy | total | Sharpe | maxDD | turnover/day |
|---|---|---|---|---|
| **equal-weight buy & hold** | **+1209.0%** | **0.92** | — | — |
| longonly top-3, hold 3 | +860.2% | 0.94 | −87.7% | 0.137 |
| longonly top-3, hold 7 | +623.8% | 0.84 | −88.0% | 0.083 |
| longonly top-3, hold 14 | +1506.8% | 0.93 | −90.3% | 0.054 |
| *(reference, NOT placeable)* neutral, hold 7 | +48.3% | 0.56 | −54.5% | 0.088 |

Sub-periods (longonly hold 7 vs buy & hold):

| period | momentum | Sharpe | B&H |
|---|---|---|---|
| 2020-22 | +332.1% | 1.08 | +417.6% |
| 2023-24 | +258.4% | 1.47 | +308.7% |
| 2025-26 | **−61.3%** | −0.53 | −42.7% |

## Honest read

- **The ranking adds essentially nothing over buy & hold.** The long-only tilt
  has roughly the *same Sharpe* (0.84–0.94 vs 0.92) but a *lower* return in most
  configurations. It just picks concentrated beta, with an 88–90% drawdown.
- **The clean, placeable edge is gone.** The dollar-neutral version — the one
  that actually beat the market — is the one Alpaca cannot express.
- **The tilt still shares the decay**: −61% in 2025-26 vs −43% for buy & hold,
  i.e. the momentum selection made the drawdown *worse*, not better.
- The dollar-neutral reference here (+48%) is much weaker than in the 11/15-name
  study because the tradeable subset is only 7 names — fewer spread
  opportunities. That reinforces the "thin universe" caveat.

## Alpaca crypto constraints this respects

- **Spot only**: market, limit, stop-limit orders; `time_in_force` `gtc` | `ioc`;
  fractional `qty`; min order notional ~$1 (already guarded by `MIN_ORDER_NOTIONAL`).
- **24/7** trading; no PDT / margin rules for crypto (because there is no margin).
- Paper and live are **different domains and different keys** — the live path
  must target the right host (`alpaca_env()` already surfaces which).

## Verdict

On Alpaca, this signal source is **not worth trading**: the market-neutral edge
is unexpressible, and the expressible long-only version is dominated by simply
holding the coins. Recommendation: **do not build this into the live bot.**
If a market-neutral book is genuinely wanted, it needs a venue that allows short
crypto (a perp/CFD venue) — the same conclusion the funding-carry research
reached, where the hedge required a perp leg.

Run:
```bash
python research/alpaca_xsec.py --lookback 30 --k 3 --hold 7
```
