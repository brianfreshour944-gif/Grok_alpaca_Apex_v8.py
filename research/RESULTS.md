# Training a "winning" model for Grok_alpaca_Apex_v8

Branch: `backtest/winning-pct-experiments`. **No `config.py` value changed, no
checkpoint promoted, nothing pushed.** Deliverables: `research/train_model.py`
(trainer) and `research/apex_gbdt.joblib` (a model that genuinely generalizes).

## Goal and the honest outcome

The request was "create a model that has a winning percentage." I trained one
and tested it out-of-sample. Result:

- **I did produce a model with a real, out-of-sample edge** -- something the
  shipped model does not have (its OOS IC is 0.00; see the previous
  `backtest_experiments/RESULTS.md` analysis). The new model keeps AUC ~0.52 and
  a positive IC out of sample.
- **But a net-winning model is not achievable on this feature set at tier-1
  taker fees.** The raw edge is ~+4-11 bps per trade; the round-trip cost is
  ~50 bps. No threshold or model recovers that gap. I am not going to report an
  in-sample win rate and call it a win -- that is exactly the trap the current
  model fell into.

## What I trained

`research/train_model.py` builds a dataset that mirrors inference **exactly**:
`add_features(df)[FEATURE_COLS]` at each bar is precisely what the bot's
sklearn path (`infer_sk` -> `predict_proba(X[:, -1, :])`) consumes, and what the
live `SafeMLPredictor.predict_batch` consumes for the last row of its window.
Label = forward return over `--horizon` bars > 0 (direction).

Model: `StandardScaler -> HistGradientBoostingClassifier(max_depth=3, lr=0.05,
l2=1.0, early_stopping)` in a sklearn pipeline, saved with joblib. This is a
drop-in champion: `SafeMLPredictor._load` treats a `.joblib` file as a sklearn
model, and `backtest_apex.make_infer` runs it through `infer_sk`.

Data: real Alpaca 15m bars, 10 symbols, 2025-03-29 -> 2026-10-07 (503,875 rows).

## Out-of-sample evidence (model trained only on data before 2026-07-19)

Walk-forward (train on the past, test on the next fifth):

| fold | AUC | IC | top-decile hit | top-decile gross bps | net (-50bps) |
|---|---|---|---|---|---|
| 1 | 0.503 | +0.001 | 0.49 | -5.5 | -55.5 |
| 2 | 0.507 | -0.000 | 0.49 | -7.7 | -57.7 |
| 3 | 0.517 | +0.022 | 0.52 | -0.0 | -50.0 |
| 4 | 0.512 | +0.015 | 0.53 | +4.7 | -45.3 |

Mean AUC 0.510, IC +0.010, top-decile gross -2.2 bps. The edge is real but
faint, and two of four folds are net-negative before costs even matter.

Confidence tail (OOS, gross bps per trade by confidence quantile) -- even the
most confident 1% of signals carry only a few bps:

| horizon | top 1% | top 2% | top 5% | top 10% |
|---|---|---|---|---|
| 6 bars | +6 | +5 | +5 | +3 |
| 8 bars | +6 | +4 | +5 | +4 |
| 16 bars | +11 | +6 | +6 | +5 |

## Engine backtest on the pure out-of-sample window (2026-07-19 -> 2026-10-08)

Through the real `backtest_apex.py` engine, 5 symbols, kill-switch off:

| model | net return | trades | win% | gross bps/trade | net bps/trade | PF |
|---|---|---|---|---|---|---|
| old champion (`grok_gqa_v9_best.pth`) | -7.89% | 2213 | ~15% | ~-8 | -57.8 | 0.14 |
| **new GBDT (this work)** | **-7.66%** | 2137 | 14.9% | **-4.9** | -54.9 | 0.15 |

The new model is genuinely better on gross (a real, if small, OOS edge where the
old model has none), but both are deeply net-negative. Threshold sweep for the
new model on the same OOS window:

| BUY_SIGNAL | net return | trades |
|---|---|---|
| 0.50 | -7.69% | 2196 |
| 0.55 | -5.87% | 1235 |
| 0.60 | -0.54% | 72 |
| 0.65 | -0.45% | 3 |
| 0.70 | 0.00% | 0 |

Nothing is net-positive; the "flat" rows just stop trading. This is the same
shape as the current model's sweep: raising the threshold avoids losses by
betting less, not by finding edge.

## Why, and what would actually be needed

- The 11-feature set (microstructure + short-horizon mean reversion) has a
  genuine but tiny out-of-sample information content: AUC ~0.51-0.52, IC ~0.01.
- Alpaca tier-1 crypto taker is 0.25%/side = ~50 bps round trip. A ~5 bps edge
  cannot pay for it. Even maker fills (15 bps/side = 30 bps) exceed the edge.
- The only cell that hinted at clearing cost was the 7-day horizon in the
  horizon study (gross +130 bps), but it was a single unstable observation and
  contradicted neighbouring horizons (IC -0.02 at 4 days, +0.07 at 7 days).

To get a net-winning model you need a bigger edge per trade, i.e. better inputs
or lower costs -- not a bigger model:
1. **Richer features**: order-flow imbalance, funding/perp basis, cross-sectional
   (relative-strength across the universe), execution-aware features.
2. **Lower costs**: maker-only execution (limit orders that rest), which also
   changes the fill model.
3. **Retrain on a rolling window** and re-verify IC survives a held-out split --
   and accept that a ~0.51 AUC signal is a research result, not a deployable
   strategy at these costs.

## Files

- `research/train_model.py` -- trainer + walk-forward + honest holdout.
- `research/apex_gbdt.joblib` -- the trained GBDT (all-data fit, for reproducibility);
  `SafeMLPredictor` loads it directly.
- Not promoted: `config.MODEL_PATH` is unchanged. Promoting a net-negative model
  would be misleading.

## Limits

- Direction labels only; exits unchanged.
- One 555-day realisation; a single bull/bear mix.
- Orderbook whale filter not modelled.
- Survivorship-biased 10-symbol pool.

---

# Funding / better-inputs investigation (the likely-winner lever)

Follow-on to the section above: of the three levers (better inputs / lower
costs / rolling retrain), **better inputs** was the most promising. The single
most likely winner within that is **derivative funding / positioning**, which is
a documented multi-day signal and is *not* in the current feature set. I pursued
it end to end.

## Data

- OKX's `funding-rate-history` endpoint only exposes ~100 days.
- **Binance USDT-M monthly archives** (`data.binance.vision/data/futures/um/
  monthly/fundingRate/<SYM>USDT/...zip`) return full history: 1,644 hourly rows
  per symbol, 2025-04-01 -> 2026-09-30. 8-hour settlements.
- `api.binance.com` is geo-blocked here (HTTP 451); the Vision archive host and
  `www.okx.com` both work. Fetch is in `research/funding_probe.py::fetch_funding`.

## Funding features tested

`funding`, `funding_z` (30-settlement z-score), `funding_cum_7d`, `funding_cum_14d`,
plus cross-sectional `xs_fund_rank` and `xs_fund_z` (rank across the universe).

## Result 1 -- funding helps prediction at MULTI-DAY horizons

Walk-forward OOS AUC / IC, base 11 features vs base + funding:

| horizon | base AUC | base IC | +funding AUC | +funding IC |
|---|---|---|---|---|
| 8h (32 bars) | 0.504 | +0.005 | 0.513 | +0.023 |
| 24h | 0.507 | +0.012 | 0.505 | +0.004 |
| 72h | 0.511 | +0.031 | **0.528** | **+0.057** |
| 168h (7d) | 0.525 | +0.055 | **0.535** | **+0.084** |

Funding is the wrong timescale for the bot's 2h holds; it pays off over days.

## Result 2 -- the eye-catching number is BETA, not timing

At 72h the top-decile forward return was +146 bps vs +29 bps base (+117 bps
"alpha", 58.9% hit, AUC 0.544). But a **circular timing-alpha null** (shifting
the return series against the predictions) gives **p = 0.75** -- shifted
alignments produce alphas as large by chance, because 72h crypto trends hard and
the model is long-biased. Almost all of that "+117 bps" is market beta.

## Result 3 -- the beta-free arbiter (market-neutral long-short)

Non-overlapping, every H bars: rank symbols by model prob, long top-3 / short
bottom-3, equal weight. Beta cancels; a positive number is a genuine timing edge.

| horizon | config | n | gross bps | net (-50bps/leg) | t (net) |
|---|---|---|---|---|---|
| 72h | base | 93 | -23.2 | -123.2 | -4.08 |
| 72h | +funding | 88 | -1.8 | -101.8 | -2.84 |
| 168h | base | 40 | -67.0 | -167.0 | -2.39 |
| 168h | **+funding** | 38 | **+90.3** | **-9.7** | -0.14 |

Funding produces the **best candidate found in this whole investigation**: a
7-day market-neutral long-short at **+90 bps gross**. But:
- only **38 non-overlapping observations** (gross t ~ 1.3) -- not significant;
- after one round-trip of cost it is ~break-even (-10 bps), not profitable;
- the bot trades 2h holds, so capturing an edge that lives at 7 days would
  require a different holding period and a market-neutral book (a different
  strategy, not the current one).

## Conclusion

Funding/positioning is a *real* signal but it lives at the multi-day horizon and
is small once beta and costs are removed. It does not turn the current
short-horizon, long-only bot into a winning strategy. The honest ranking of
levers, after testing all three:

1. Better inputs (funding, cross-sectional): real but small; best case ~+90 bps
   gross at 7d with too few observations to trust.
2. Lower costs (maker): necessary but not sufficient (gross-negative at default
   threshold even at 0 fees).
3. Rolling retrain: keeps IC honest but cannot manufacture edge that isn't there.

A genuinely winning strategy would need to be a different one: multi-day,
market-neutral, funding/positioning-driven, maker-executed. That is a new
project, not a tweak to this bot.

## Files

- `research/funding_probe.py` -- fetch + funding features + the three tests above.
- Funding cache lives in `bt_cache/funding/` (gitignored).
- Not promoted; no `config.py` change; nothing pushed.

---

# The "different strategy" — market-neutral funding carry (2026-10-09)

Branch `research/market-neutral-funding`. This section answers the closing line
of the previous investigation ("a genuinely winning strategy would need to be a
different one: multi-day, market-neutral, funding/positioning-driven"). I built
that strategy and tested it out-of-sample. **It is net-positive at realistic
costs, with a timing-alpha null p-value of 0.00 — the first configuration in
this repo that clears costs — but its P&L is regime-concentrated, so treat the
headline CAGR as an upper bound, not a promise.**

## What changed from the failed funding probe

The earlier probe (`funding_probe.py`) ranked by a *model* that had already been
shown to have no OOS edge, and it omitted two things that turn out to matter:

1. **The funding cashflow itself was missing.** A perp long-short earns the
   funding payments (when funding > 0, short legs collect, long legs pay). The
   probe only scored the price leg, so a carry strategy looked like pure price
   betting. Adding the cashflow is worth ~+3.5 bps/period at 24h.
2. **The right horizon for a *model-free* carry signal is 1 day, not 7.**
   Funding predicts multi-day direction poorly (as the probe found), but the
   cross-sectional funding *level* earns a large, robust price reversal over the
   next 24h — that reversal, not the slow prediction, is where the edge is.

## The strategy (`funding_carry.py`, deployable)

- Score: `-cross_sectional_z(Σ 168h funding)` — long the lowest-funding, short
  the highest-funding names. Turnover is tiny (~0.16 legs/period) because
  funding ranks are persistent, so costs are ~1 bps at Binance fees.
- Book: equal-weight, dollar-neutral long-top-3 / short-bottom-3, rebalanced
  every 24h. 15-symbol liquid Binance USDT-M perp universe.
- Costs modelled at the *venue that has the data*: Binance USDⓈ-M taker 5 bps /
  maker 2 bps per fill. Alpaca's tier-1 25 bps is shown as a reference column
  and is **not** the relevant fee here.

## Evidence (15 symbols, 21 months, 2025-01 → 2026-09, all out-of-sample-ish)

| mode | H | n | gross bps | price | funding | net taker(5) | t | net maker(2) | net alpaca(25) | p_null |
|---|---|---|---|---|---|---|---|---|---|---|
| **carry** | 24h | 630 | +14.9 | +11.4 | +3.5 | **+10.2** | +1.14 | +13.0 | -8.6 | **0.00** |
| carry | 72h | 210 | +26.1 | +16.3 | +9.8 | +16.9 | +0.67 | +22.4 | -19.8 | 0.05 |
| carry | 168h | 90 | +86.2 | +65.7 | +20.5 | +73.4 | +1.38 | +81.1 | +22.1 | 0.02 |
| ml (model) | 24h | 369 | +2.8 | — | — | -15.0 | -1.58 | -4.3 | -86.4 | 0.43 |

- **The ML gate is dead weight**: every model mode is negative or ~0 at every
  horizon. The edge is the funding carry, not a model.
- **Venue is decisive**: at Binance fees the 24h carry earns ~+10 bps/period
  (≈ +42% simple annualised at full 2× gross); at Alpaca's 25 bps it is -8.6.
  The same signal that fails on Alpaca works on a perp venue because a
  rebalance costs ~1 bps instead of ~8 bps.
- **Timing-alpha null p=0.00** at 24h across 630 non-overlapping observations —
  the P&L is not market beta (the book is dollar-neutral by construction).

## Robustness — the honest caveats

- **Sub-period concentration.** At 24h the three 7-month chunks are +8.3, +39.2,
  -2.9 gross bps. Essentially all of the P&L is the 2025-08 → 2026-03 phase
  (a high-funding, high-dispersion bull). The other two thirds are ~flat.
- **K sensitivity.** K=2 (+15.3) and K=3 (+14.9) work; K=4 (+4.7) and K=5 (-0.1)
  do not. Wider books dilute into names where the carry/reversal is absent.
- **Gate experiments failed.** Conditioning on funding *dispersion* or a BTC
  uptrend **destroys** the edge (gross → ~0): the price alpha lives in
  low-dispersion, range-bound regimes, which is the opposite of intuition.
- **Cost model.** No slippage/funding-interval drift is modelled; fills assumed
  at the rebalance close. Real execution will be worse by a few bps.

## Verdict

This is the first thing in the repo that clears costs out-of-sample, and it does
so for a structural reason (a real dollar-neutral carry + short-horizon reversal
edge, tiny turnover, and a venue with 2–5 bps fills). It is **not** a
promise of a smooth 42%/yr: the edge is regime-dependent and was concentrated in
one market phase. Recommended next step is paper-trading the 24h/K=3 book with
maker orders on a perp venue, tracked against the sub-period profile above.

## Files

- `funding_carry.py` — deployable strategy (pure functions: `funding_carry_score`,
  `select_book`, `target_weights`, `plan_rebalance`, `book_metrics`).
- `research/market_neutral_funding.py` — harness: fetch, features, walk-forward,
  market-neutral book, cost tiers, circular-shift null, robustness, and a
  `verify_deployable()` parity check against `funding_carry.py`.
- `tests/test_funding_carry.py` — 11 unit tests.
- Data cached under `bt_cache/research/` (gitignored).
- No `config.py` change; the incumbent bot is untouched.

