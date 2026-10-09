# GrokApex retraining: data, honest scoreboard, and result

**Question:** can retraining the grok model make the bot profitable?

**Answer: no — not on these features at this venue.** This documents the data,
the leakage fixes, the retrain, and why it lands where it does. It is a *negative*
result, and a useful one: it localizes the wall to the **features + venue cost**,
not the model.

## Data (real, not cached leftovers)

`research/fetch_alpaca_bars.py` pulls Alpaca 15m crypto bars -> `bt_cache/*.csv`
(gitignored). Alpaca crypto history begins **2021-01-01**.

| symbol | bars | | symbol | bars |
|---|---|---|---|---|
| BTC | 202,027 | | AVAX | 171,108 |
| ETH | 202,021 | | LINK | 201,745 |
| SOL | 161,667 | | BCH | 195,148 |
| DOGE | 201,822 | | DOT | 110,051 |
| LTC | 201,359 | | **ADA** | **22,786 (listing 2026-02, excluded)** |

9 symbols x 2021-01..2026-10 = **1,646,876 rows** (vs the repo's earlier ~504k).

## Honest scoreboard (`research/oos_eval.py`)

The existing `walk_forward` is not leakage-free: with an expanding split the last
`H` training labels reach into the test block, and the 20-bar feature window
straddles the train/test boundary. `oos_eval.py` runs it **naive** vs
**purged + embargoed** (drop `H` boundary train rows, embargo the first 20 test
rows). Both run in seconds because features are computed once over the full frame.

**The gap between the two columns was ~0.0003 IC.** Leakage was *not* the
illusion — the illusion was the belief that the incumbent's in-sample edge would
survive out of sample.

## Edge vs horizon (the decisive chart)

Purged walk-forward, 9 symbols, gross = top-decile mean forward return:

| horizon | label | AUC | IC | top-hit (base) | GROSS bps | fee needed |
|---|---|---|---|---|---|---|
| 8 | 2h (incumbent) | 0.519 | +0.029 | 0.528 (0.501) | **+4.4** | 4.4 |
| 24 | 6h | 0.513 | +0.020 | 0.523 (0.499) | +11.5 | 11.5 |
| 96 | 24h | 0.508 | +0.014 | 0.517 (0.494) | +27.7 | 27.7 |
| 288 | 72h | 0.496 | −0.003 | 0.493 (0.488) | +113.2 | 113.2 |
| 672 | 7d | 0.492 | −0.017 | 0.489 (0.481) | +209.4 | 209.4 |

Two readings, both fatal to the current shape:

1. **At 2h (what the bot trades) the top decile is worth +4.4 bps gross.** Even a
   *free* venue (Binance perp taker 10 bps, maker ~4) fails. Alpaca's 50 bps is
   ~11x too high.
2. **Gross rises with horizon but IC goes negative and hit-rate drops below
   base.** The 7d "+209 bps" is **beta/trend, not skill** — a long-biased sign
   effect, exactly the circular-shift trap flagged elsewhere in this repo. The
   only horizon with a real (small) IC is ~2-6h, and there the dollars are tiny.

## The retrain itself

Trained the drop-in `StandardScaler + HistGradientBoostingClassifier` on all
1.65M rows (`research/train_model.py`, `--horizon 8 --walk-forward`):

- walk-forward: mean AUC **0.519**, IC **+0.029**, top-decile gross **+4.5 bps**,
  net(−50) **−45.5 bps**.
- artifact `research/apex_gbdt_2021.joblib`; verified to load inside the live
  `SafeMLPredictor` (`sklearn champion`, predicts 0.5028 for the last BTC bar).

**A model trained on 3x more data lands at the same +4 bps.** More data did not
move the edge, which is the whole point: the ceiling is the 11 features, not the
sample.

## Independent OOS splits (reproducing the incumbent verdict)

Purged single split, H=8, cost 50 bps:

| train < | test rows | AUC | IC | top-hit (base) | gross | net |
|---|---|---|---|---|---|---|
| 2025-01-01 | 557,178 | 0.5181 | +0.0258 | 0.526 (0.500) | +2.2 | **−47.8** |
| 2026-01-01 | 242,199 | 0.5199 | +0.0290 | 0.525 (0.497) | +3.2 | **−46.8** |

Same conclusion as the original research (OOS IC ~0.00–0.03, no net edge),
now with 3x the data and leakage removed.

## Why

- **Signal:** 11 microstructure features carry ~+4 bps/trade of real-but-tiny
  OOS edge at 2h. That is the known ceiling for this feature set.
- **Cost:** Alpaca crypto = **50 bps round trip** (25/side, taker), enforced at
  `config.py:110`. Interest (a probability) is not dollars (net edge).
- **Shape:** `MAX_HOLD_HOURS = 4.0` on 15m bars. A 4h hold cannot clear 50 bps
  unless the model is roughly 60% directional — it is ~52%.

## What this does NOT rule out

The features are the constraint, not "ML" broadly. The levers that remain:

1. **Better inputs** — order-flow imbalance, cross-sectional relative strength,
   funding/perp basis, longer-horizon momentum. **All of these require a perp
   venue**; Alpaca spot does not expose them. (The repo's funding-carry work is
   the same conclusion from the other direction.)
2. **Lower cost / longer hold** — a perp venue at 4-5 bps and a multi-day hold.

Neither is a retrain on the current 11 features. That is the finding.

## Files

- `research/fetch_alpaca_bars.py` — pull Alpaca 15m history into `bt_cache/`.
- `research/oos_eval.py` — purged/embargoed walk-forward + date split, naive-vs-
  purged comparison, net-of-cost.
- `research/edge_sweep.py` — edge vs horizon, with breakeven round-trip fee.
- `research/apex_gbdt_2021.joblib` — retrained candidate (drop-in, NOT promoted).
- `bt_cache/*.csv` — 1.65M bars (gitignored).
- **No `config.py` / `MODEL_PATH` change. Nothing promoted. `main` untouched.**
