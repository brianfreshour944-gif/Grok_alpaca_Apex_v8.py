# Cross-sectional momentum — composition / survivorship test

The headline result (`XSEC_MOMENTUM_RESULTS.md`) uses names that all still
exist today. That invites two objections: *the edge is one lucky name*, and
*it depends on which names are in the universe*. `research/xsec_composition.py`
attacks both.

Method: candidate symbols that **all share a common history** (so the long/short
legs are always comparable) — 11 of the 15, since BNB/AVAX/DOT/SOL listed later.
Config: lookback 30d, k=3, hold 7d, 25 bps. Full book on 11 names: **+193.9%,
Sharpe +0.86**.

## 1. Who earned it? (per-symbol PnL contribution)

| symbol | contrib | | symbol | contrib |
|---|---|---|---|---|
| DOGEUSDT | **+1.01** | | ETHUSDT | +0.06 |
| BTCUSDT | +0.51 | | ATOMUSDT | −0.02 |
| ADAUSDT | +0.26 | | BCHUSDT | −0.19 |
| ETCUSDT | +0.26 | | LTCUSDT | −0.29 |
| XRPUSDT | +0.26 | | | |
| TRXUSDT | +0.17 | | | |
| LINKUSDT | +0.14 | | | |

**The top name is 32% of total gross contribution** — above the ~25% "healthy"
line. DOGE is load-bearing. That is the concentration risk made concrete.

## 2. Leave-one-out (drop one symbol, re-run)

| dropped | Sharpe | total % |
|---|---|---|
| none | +0.86 | +193.9 |
| ADA | +0.74 | +110.3 |
| BTC | +0.78 | +145.8 |
| TRX | +0.72 | +108.0 |
| **DOGE** | **+0.58** | **+22.7** |
| others | +0.78 … +0.92 | +136 … +250 |

**Dropping DOGE alone cuts the total from +194% to +23%.** Every other single
drop keeps a positive, useful book — so it is *not* one-name-fragile overall,
but it *is* materially DOGE-dependent.

## 3. Symbol holdout (trade a random 11-minus-5 subset)

20 random 5-name holdouts: Sharpe **mean +0.41, sd 0.19, min +0.06, max +0.87,
100% positive**. The edge survives on arbitrary sub-universes — weaker (it
scales with universe size, which is expected: fewer names = fewer spread
opportunities) but never negative.

## 4. Null

Random-ranking shuffle (200 trials) on the 11-name book: mean Sharpe −0.05,
sd 0.40; real +0.86 → **p = 0.010**. Still clears the noise null.

## Honest verdict

The composition test **weakens but does not kill** the momentum result:

- **Survives survivorship-style composition changes** (symbol holdout 100%
  positive; leave-one-out positive except a mild DOGE hit). This is better than
  most "it's just the winners" stories.
- **But it is materially DOGE-dependent** (top-name share 32%; −194%→+23% when
  dropped). A k=3 book on 11-15 names *will* be concentrated; that is the cost
  of the construction, not a bug, but it must be disclosed.
- Universe size matters: the more names, the better the diversification of the
  dispersion bet. 15 is thin.

Caveats unchanged from the parent study: no genuinely dead coins in the panel
(true survivorship is still unmeasured), unmodelled short borrow/slippage, and
the 2025–2026 regime decay.

**Do not deploy.** Paper-forward-test, and if pursued, widen the universe and
cap per-name weight so no single symbol can be 32% of the book.

Run:
```bash
python research/xsec_composition.py --lookback 30 --k 3 --hold 7
```
