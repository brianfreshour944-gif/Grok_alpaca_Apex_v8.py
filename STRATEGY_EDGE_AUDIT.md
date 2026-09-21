# Strategy & Trading Edge Audit — Grok_alpaca_Apex_v8.py

**Date:** 2026-09-20
**Method:** Actual execution wherever the codebase supports it, not just
reading. This repo is meaningfully different from this project's other two
bots (`Apex_oracle_bot`, `Apex_Committee_Bot`): it ships real analysis
tooling for most of this framework's 20 sections
(`backtest_validation.py`, `walkforward_robustness.py`,
`monte_carlo_failure.py`, `ablation_testing.py`, `signal_calibration.py`,
`regime_analysis.py`, `risk_analysis.py`, `execution_cost_analysis.py`,
`exit_analysis.py`, `complexity_budget.py`, `drift_monitor.py`,
`adaptive_monitor.py`, `stress_test.py`, `stress_test_market_drops.py`,
`portfolio_crash_test.py`). I ran every one of them rather than assuming
what they'd say.

**The one fact that gates everything below:** this session's
`CORRECTNESS_AUDIT.md` (same day, same repo) found and fixed a bug that
made the transformer model return a constant, information-free `0.5`
signal for every symbol, every cycle, from 2026-09-17 until the fix
landed a few hours before this report. **Any trade history that exists
from that window was not driven by real model predictions**, regardless
of what the logs show. This isn't a caveat on the edge analysis — it
*is* the edge analysis's biggest finding, so it's repeated at point of
use throughout.

---

## 1. Strategy & Trading Edge

**Core edge, read from the actual decision code (`main_bot.py:661`,
`config.py`'s `get_regime_params`):** trend-following momentum, gated by
an ML signal. Entries require `trend == "up"` (price above a
volatility-adaptive EMA, `regime.py`) **and** `signal > buy_signal`
(regime-adjusted threshold around the transformer's output probability).
This is **trend persistence**, filtered by a microstructure-feature
model (`feature_engineering.py`'s 11 features: Parkinson/Garman-Klass
vol, Kyle lambda, signed flow, VWAP-z, Amihud illiquidity, autocorrelation,
etc.) — closest category match: **momentum + market microstructure**,
not mean-reversion or breakout.

**Why the edge should exist / persist / decay:** not stated anywhere in
the code, same gap as this project's other two bots. The feature choices
cite real academic literature (Parkinson 1980, Garman-Klass 1980, Kyle
1985, Amihud 2002) for *why those features carry information in
general*, but nothing states why *this specific model*, trading 15-minute
crypto bars, should have retained an edge specifically. `config.py`
itself documents **survivorship bias** in the trading universe (lines
54-59): the 10 `DYNAMIC_UNIVERSE_CANDIDATES` symbols were chosen because
the model was trained on them, not because of any principled inclusion
criterion — an unusually honest, self-flagged admission worth crediting.

**Edge verification — now partially answerable, unlike the other two
bots, once data accumulates:** `ablation_testing.py`,
`signal_calibration.py`, `regime_analysis.py`, `walkforward_robustness.py`,
and `backtest_validation.py` all exist and are wired to genuinely test
exactly what this section asks (per-asset/regime/parameter robustness,
curve-fitting vs. real signal, profit concentration in best trades). **I
ran all of them. Every one returned "No experiences found."** There is
currently zero trade history in `live_experiences.jsonl` in this
environment — meaning the infrastructure to answer this section exists
and is sound, but has never actually been run against real outcomes.

### Required Output
- **Edge confidence: Low.** Not because the feature engineering is naive
  (it isn't) — because nothing has ever measured whether the trained
  model's predictions correlate with real forward returns, and for the
  last 3 days in production it provably didn't (Finding 1/2 of
  `CORRECTNESS_AUDIT.md`).
- **Primary evidence:** theoretical (feature citations) only.
- **Biggest weakness:** the exact tooling needed to produce real evidence
  exists in this repo and has simply never been run against real
  outcomes yet.
- **Recommended test:** let the bot run with the fix in place for at
  least the `min_data_for_adaptation` threshold this repo itself already
  defines (`adaptive_monitor.py`: 50 trades), then run
  `signal_calibration.py --threshold-test` and `ablation_testing.py` for
  real numbers instead of a "no data" placeholder.

---

## 2. Data Integrity & Information Quality

**Point-in-time integrity: verified correct by reading + confirmed by
the passing test suite.** `data_feeds.get_clean_ohlcv_dataframe()`
filters out any bar whose close time is still in the future
(`bar.timestamp + 15min <= now_utc`) before returning — no look-ahead.
`feature_engineering.py` computes every feature from trailing
`.rolling()`/`.shift()` windows only.

**Data leakage: none found in the feature pipeline. One was found and
fixed in the pipeline's *execution*, not its math** — the
`np.where()`-returns-ndarray bug (`CORRECTNESS_AUDIT.md` Finding 1) broke
`add_features()` entirely rather than leaking data; now fixed and
verified with a clean `(N, 11)` output, no NaN/inf.

**Feature quality (which features actually predict returns, which are
redundant):** cannot be answered — `feature_analysis.py` exists
specifically for this (correlation matrix, feature importance, drift by
regime) and returned "No experiences found" when run. Same gate as
Section 1.

### Required Output
- **Data-quality score: Good for pipeline correctness (verified by
  execution), Unknown for feature predictive value (no data yet).**
- **Potential leakage detected:** none in the feature math itself.
- **Most valuable / redundant features:** not determinable yet — rerun
  `feature_analysis.py --importance` once `live_experiences.jsonl` has
  data.

---

## 3. Signal Generation & Decision Quality

**Entry:** `trend == "up" and signal > regime_params["buy_signal"]`
(`main_bot.py:661`), gated further by an orderbook whale filter, an ATR
sentinel veto (`atr_pct > 6.0`), and a "wild regime + low conviction"
veto. **Exit:** purely mechanical (`exit_logic.evaluate_exit`) — trailing
stop, time-decay stop-loss, max-hold, weak-signal exit, checked in that
priority order. Decision is deterministic given the same inputs (no
sampling), model-based for entry, rule-based for exit.

**Does every model contribute measurable value?** Two "models" exist:
the transformer champion (trades) and a GBT shadow challenger (never
trades, logs only — confirmed by reading `shadow_model.py`'s own
docstring and by the fact `main_bot.py` never routes its output into any
decision). The champion's contribution cannot currently be measured
independently of the entry-gate's `trend` filter — that requires
`ablation_testing.py`'s "System − Model" comparison, which needs data
that doesn't exist yet.

**Signal calibration table (predicted probability vs. actual win rate):
cannot be produced.** `signal_calibration.py` computes exactly this
table and the "top 10%/20%/30% of signals" threshold test — I ran it,
it needs `live_experiences.jsonl`. Unlike `Apex_Committee_Bot`, this
isn't a missing-instrumentation problem — `main_bot.py:803` already logs
`signal`, `regime`, `features`, and `feature_importance` on every entry
via `log_entry_experience()`, and `log_exit_outcome()` on every exit.
The instrumentation is real; there's simply no trade history behind it
yet **and none of the history from 2026-09-17 onward is usable even once
it exists**, since it was captured with `features={}` (Finding 2,
`CORRECTNESS_AUDIT.md`) and a signal that was always exactly `0.5`.

---

## 4. Market Regime Detection

`regime.compute_regime_and_trend()` classifies wild/normal/quiet by
ATR% against a fixed 1.5% baseline, verified correct (explicit NaN guard,
fails safe to "normal/neutral" rather than silently miscomparing —
`regime.py:41-43`). `regime_params` correctly loosens/tightens the entry
threshold and profit-target/stop-loss per regime
(`config.get_regime_params`).

**Regime performance matrix: cannot be filled in.** `regime_analysis.py`
exists specifically to build this table (trades/win-rate/avg-PnL/profit-
factor/max-DD per regime) — ran it, "No experiences found." The
classification logic itself is sound (verified by reading); whether the
strategy actually makes money in the regimes it's tuned for is
unmeasured.

---

## 5. Risk Management

**Position sizing:** volatility-adjusted (`regime.calculate_adjusted_risk`
scales down once ATR% exceeds 1.5%, verified: at ATR=8% risk scales to
19% of baseline, at ATR=12% to 12% — confirmed by running `stress_test.py`
directly, not just reading), further scaled by a Half-Kelly multiplier
from the model's own signal probability (`portfolio.calculate_kelly_multiplier`),
capped by `MAX_SINGLE_TRADE_USD` and a 20%-of-equity concentration cap
(`MAX_POSITION_PCT`). **So: no, the bot does not risk the same dollar
amount when volatility doubles — verified by execution, not just by
reading the formula.** This is a materially more sophisticated sizing
model than either of this project's other two bots.

**Portfolio risk — real numbers, from actually running the stress
suite, not from historical data (none exists) but from synthetic
scenarios exercising the bot's real exit logic:**

`portfolio_crash_test.py` (10-position, equal-weight/concentrated
portfolios, correlated crash scenarios) — **4/4 scenarios PASS**:
| Scenario | Max drawdown | Threshold | Result |
|---|---|---|---|
| 10-position correlated progressive crash (hours) | 2.00% | 10.00% | PASS |
| 10-position correlated flash crash (minutes) | 5.00% | — | PASS |
| 20%-concentrated (new cap), flash crash, 2-step fill delay | 5.00% | 6.00% | PASS |
| 10% drop then 24h recovery | 2.00% | 12.00% | PASS |

**Risk of Ruin under stress — the most important number in this
report, with a caveat that matters as much as the number:**
`stress_test.py` calls `risk_analysis.calculate_risk_of_ruin()` with a
**deliberately adversarial 40% win rate** (not a measured one — there is
no measured one yet), 1:1 reward:risk (matching `PROFIT_TARGET_PCT` =
`STOP_LOSS_PCT` = 2%), and a $200 risk-per-trade on $10,000 equity:

```
Win Rate: 40% (stressed)
Risk of Ruin: 0.578
Interpretation: VERY HIGH RISK
```

This is exactly the kind of adversarial stress test the framework asks
for (§5: "Do not assume historical win rate will remain constant") and
deserves to be taken seriously — a 40% win rate at 1:1 payoff is a
guaranteed-losing edge (expected value per trade = 0.4×2% − 0.6×2% =
−0.4%), so a high ruin probability is the *correct* answer for that
input, not a bug.

**But the number itself is likely overstated relative to how this bot
actually trades, and I traced exactly why (read, not run further —
noting this precisely rather than asserting a corrected number I
haven't computed):** `calculate_risk_of_ruin()`'s Monte Carlo simulation
(`risk_analysis.py:394-423`) takes `risk_per_trade` as a **fixed dollar
value computed once** from the *starting* equity and never
recalculates it as the simulated equity changes across the 252-trade
path. The real bot sizes every trade as a percentage of *current*
equity (`calculate_adjusted_risk`, `calculate_kelly_multiplier`) — true
fixed-fractional position sizing, which makes literal bankruptcy
(equity hitting exactly `$0`, this simulation's definition of "ruin")
far harder to reach than a fixed-dollar-bet model, because position size
shrinks as losses accumulate. **The 0.578 figure is a valid worst-case
signal that the strategy has no edge at a 40% win rate — it is not a
faithful simulation of this bot's actual capital-preservation behavior
under that win rate**, and should not be quoted as "58% chance of
blowing up the account" without that distinction.

**Tail risk (flash crash, gap, exchange outage, stale prices, duplicate
orders):** `stress_test.py`'s full battery ran real numbers against a
synthetic $10k/5-position book — flash crash −0.40%/−$39.60, gap-down
−0.80%/−$80, extreme volatility risk reduced 81-87% by the vol-scaling
sizing, exchange outage max loss $10-40, stale prices $5-10 potential
loss. All rated LOW-MEDIUM except **liquidity collapse, rated CRITICAL**
with no quantified loss figure — the tool itself flags this as the one
scenario it can't bound (mitigation text: "use limit orders, reduce size
in low-liquidity conditions" — advisory, not enforced anywhere in the
live code I read in the correctness audit).

---

## 6. Stop-Loss & Exit Logic

Already covered in depth by the correctness audit's reading of
`exit_logic.py` (verified correct: trailing stop scales with ATR%,
clamped between 1-2%; time-decay stop tightens at 1h/2h held; priority
order matches its own docstring exactly). **New here:**
`stress_test_market_drops.py` actually exercises this logic across a
range of ATR levels with a real synthetic price path:

```
ATR=  1.0%: exited at step 14, trailing_stop=1.00%  Time-Decay Stop loss (-1.50% <= -1.50%)
ATR=  2.0%: exited at step 14, trailing_stop=1.00%  Time-Decay Stop loss (-1.50% <= -1.50%)
ATR=  4.0%: exited at step 14, trailing_stop=2.00%  Time-Decay Stop loss (-1.50% <= -1.50%)
ATR=  8.0%: exited at step 14, trailing_stop=2.00%  Time-Decay Stop loss (-1.50% <= -1.50%)
ATR= 12.0%: exited at step 14, trailing_stop=2.00%  Time-Decay Stop loss (-1.50% <= -1.50%)
ATR= 20.0%: exited at step 14, trailing_stop=2.00%  Time-Decay Stop loss (-1.50% <= -1.50%)
```

Every ATR level in this synthetic scenario exits via the time-decay
stop before the trailing stop ever gets a chance to differentiate
(trailing_stop correctly widens from 1.00% to 2.00% as ATR rises from
1% to 4%+, but this particular test's price path triggers the -1.5%
stop-loss first regardless). **Whether the stop is too tight/too wide
against real market noise (MAE/MFE analysis) cannot be answered** —
`exit_analysis.py --stop-test` exists for exactly this and needs real
trade history.

---

## 7. Execution & Trading Costs

`execution_cost_analysis.py` exists (fees, spread, slippage, market
impact, partial-fill rate, a 2×/3×-cost stress test) — ran it, needs
`live_experiences.jsonl`. From the correctness audit's reading of
`orders.py`: fills are polled for up to 60s and real
`filled_avg_price`/slippage-vs-expected-price is logged
(`orders.py:135-144`), so once trades accumulate this tool has real data
to work with — unlike `Apex_Committee_Bot`, where fee/fill tracking had
to be added from scratch. `orders.py`'s own comment (verified against
the installed `alpaca-py==0.33.0` source in the correctness audit) is
correct that no commission field exists on Order objects — fees are
not currently captured at all, only slippage.

---

## 8-10. Backtesting Integrity, Walk-Forward, Monte Carlo

**Infrastructure exists and I ran all three; none can produce output
yet.** `backtest_validation.py` → "Skipped: No experience data found".
`walkforward_robustness.py` → "Error: No experiences found".
`monte_carlo_failure.py` (bootstrapped trade resampling, randomized
slippage/fees/delays — the real answer to "how badly can this fail," not
the fixed-scenario stress tests above) → "Error: No experiences found".
This is the same structural gate as every data-dependent section above:
the tools are real, the trade history isn't yet.

---

## 11. Model & AI Contribution

Two models: the transformer champion (trades) and GBT shadow challenger
(logs only, confirmed never routed into a decision). `ablation_testing.py`
exists for exactly the "System − Model A" comparison this section asks
for — ran it, needs data. **What I can say from the correctness audit
without needing that data:** for the last 3 days, the champion's
contribution to any BUY decision was exactly zero (constant `0.5`
signal) — meaning any trades placed in that window were driven entirely
by the `trend == "up"` EMA filter, not the model. That's not a
statement about the model's long-run value; it's a statement that the
last 3 days of history, whatever it shows, doesn't measure it.

---

## 12. Adaptive / Self-Learning Behavior

This is the one area where this bot goes well beyond the other two in
this project — `adaptive_monitor.py` defines an explicit
Continue → Reduce Risk → Pause → Retrain → Rollback ladder with hard
limits (`ADAPTIVE_LIMITS`: max 5% parameter change/day, minimum 50
trades before adapting, max 3 consecutive adaptations before forcing
human review, 24h cooldown between adaptations) and
`promotion_gate.py` implements an actual champion/challenger promotion
workflow with accuracy/F1 floors and automatic demotion criteria. **I
verified this is real code, not aspirational documentation** — I read
`promotion_gate.py` in full during the correctness audit and it
genuinely backs up the champion file, evaluates both models on held-out
validation data, and only promotes on a measured improvement. This
directly answers several of this section's questions in the
affirmative: yes there's a hard limit on autonomous changes, yes it can
revert (backup + `--demote`), yes retraining is walk-forward validated
before promotion — **but the whole pipeline runs on
`live_experiences.jsonl`/model retraining data that doesn't exist yet
either**, so it's unexercised in practice.

---

## 13. Performance Metrics

Cannot be computed — every metric this section asks for (Sharpe,
Sortino, Calmar, profit factor, expectancy, VaR) depends on realized
trade outcomes that don't exist yet. The computation code for most of
these lives in `risk_analysis.py`/`drift_monitor.py`'s `analyze_*`
functions and is straightforward, standard math (verified by reading,
not flagged as suspect) — it's a data problem, not a code problem.

---

## 14. Benchmark Against Simpler Alternatives

`performance_dashboard.py` exists; not run in this pass (also gated on
real trade/equity history). No benchmark against buy-and-hold or a
single-indicator baseline exists in any script I found — worth adding
once there's data, same recommendation as for the other two bots in
this project.

---

## 15. Operational Reliability

**Important finding: `operational_reliability.py` is not a test.**
Despite its module docstring ("Tests failure scenarios and fail-safe
behaviors") and its `--scenario` CLI flag implying it exercises real
code paths, I read the source: it prints a **hand-written, static
dictionary** (`FAILURE_SCENARIOS`) of claimed detection/response/recovery
behavior for each failure mode. Nothing in it calls into `main_bot.py`,
`orders.py`, or `portfolio.py` to verify the claims are true.

**One claim I checked against the actual code and found false:** the
table says, for "Model Unavailable": *"Response: Use fallback (shadow
GBT), or pause."* But `shadow_model.py`'s own docstring states "This
model NEVER trades," and `main_bot.py` never routes its output into any
trading decision (confirmed in the correctness audit) — there is no
fallback path at all. Worse: `main_bot.py:201` instantiates
`SafeMLPredictor` at **module import time**, with no surrounding
try/except, and `SafeMLPredictor._load()` raises `FileNotFoundError` if
the model file is missing. **The actual behavior of "model unavailable"
is that the entire process fails to start** — not "pause," not
"fallback," a hard crash before the trading loop ever begins. This is
exactly the category-7 risk the correctness audit's process warns about:
a document describing intended behavior was taken as evidence without
checking it against the code, and in this case the intended behavior it
describes doesn't exist.

**One claim I checked and found true:** "Database Failure... Response:
Continue trading (DB is logging only)" — verified in the correctness
audit's read of `database.py`: every function no-ops or catches its own
exceptions when `DATABASE_URL` is unset/unreachable. This one is
accurate.

I did not exhaustively verify the other 10 scenario claims in the table
against the code — flagging that limit explicitly rather than implying
uniform verification.

---

## 16. Monitoring & Alerts

Discord alerting exists for BUY/SELL (`main_bot.py`, fire-and-forget
`asyncio.create_task` with an error callback — a good pattern, verified
in the correctness audit) and for drift (`drift_monitor.py`, whose
alert path was broken — missing `httpx` dependency — and fixed in this
session's other commit). No alerting exists yet for the things Section
16 explicitly asks about beyond P&L/drawdown: "model confidence suddenly
changes," "feature distribution changes" are computed by
`adaptive_monitor.py`'s drift functions but nothing pushes them to
Discord automatically the way `drift_monitor.py` does for its own
metrics — they're print-only CLI reports today.

---

## 17. Human Oversight

Not formally coded as a permissions model (same gap as the other two
bots), but the pieces exist informally and are more mature than
`Apex_Committee_Bot`'s: `promotion_gate.py`'s CLI requires an interactive
`y/N` confirmation before promoting a model unless `--force-promote` is
passed, and before demoting unless scripted. `main_bot.py`'s daily-loss
kill-switch explicitly logs "Bot halted. Manual restart required after
reviewing losses" (verified in the correctness audit's read) and does
not auto-resume. These are real, code-verified human-in-the-loop gates
for the two most consequential actions (model changes, halting after
losses) — worth documenting explicitly as a policy rather than leaving
it implicit in the code, but functionally present.

---

## 18. Complexity Budget

**Ran `complexity_budget.py` — real static output:**
```
Total Components: 24
Total Lines of Code: 940
Avg LOC/Component: 39.2

Category               Count     LOC % of Total
infrastructure             4     400      42.6%
model                      2     280      29.8%
feature                   11     120      12.8%
risk                       4      75       8.0%
exit                       3      65       6.9%

RECOMMENDATIONS: KEEP 0 | EVALUATE 0 | CONSIDER REMOVING 0
```
The KEEP/EVALUATE/REMOVE recommendation engine needs real ablation
results to classify anything — currently empty for the same reason as
every other data-gated section. The raw inventory itself is a genuine,
useful answer to "how much is here": 11 of 24 tracked components are
individual features (12.8% of LOC), 2 are the model itself (29.8% —
the transformer architecture + predictor wrapper), consistent with what
I read directly in `ml_predictor.py`/`feature_engineering.py`.

---

## 19. Kill Criteria

Real, code-verified (correctness audit reading of `main_bot.py`):
`MAX_DRAWDOWN_STOP` (session drawdown) and `DAILY_LOSS_LIMIT` both hard-halt
the process (`break` out of the loop, not just an alert), force-close
all positions first, and require manual restart — no auto-resume path
exists. `adaptive_monitor.py`'s threshold ladder additionally defines
rollback/retrain/pause triggers on drift-score, win-rate, Sharpe, and
feature-drift floors — genuinely matching this section's requested
Pause → Diagnose → Retrain → Re-test → Reauthorize flow in code, not
just in a comment. All of it is currently unexercised for lack of data,
same caveat as Section 12.

---

## 20. The Most Important Questions — Direct Answers

**Edge:** Trend persistence (EMA-based) gated by an 11-feature
microstructure model. No stated reason it should exist or persist
against other participants; the code itself flags survivorship bias in
its own trading universe.

**Evidence:** None yet — not "none exists in principle," but "the tools
to produce it are real and have simply never been run against outcomes
that reflect a working model," since the model has only been functional
for a few hours as of this report.

**Robustness:** Partially demonstrated for the *risk/exit* side by
actual execution today (4/4 crash-stress scenarios pass; vol-scaled
sizing confirmed by running it, not just reading it) — **not**
demonstrated at all for the *signal* side (no calibration, no
walk-forward, no ablation, no Monte Carlo on real outcomes).

**Simplicity:** Unknown — `ablation_testing.py` is the right tool and
is ready; it just hasn't run against real data yet.

**Failure:** Two concrete, code-verified failure modes now exist on
record: (1) the 3-day silent model outage this session found and fixed
in `CORRECTNESS_AUDIT.md`; (2) a newly-found gap in this report — model
file unavailability crashes the whole process at import time despite
`operational_reliability.py` claiming a graceful fallback exists.

**Adaptation:** More genuinely built-out than this project's other two
bots — real promotion-gate and drift-threshold code, not just design —
but entirely unexercised for lack of trade history.

**Complexity:** 24 components, 940 LOC, real inventory produced today;
which ones earn their place is not yet measurable.

**Improvement, single smallest highest-expected-value change:** let the
now-fixed bot accumulate the ~50 trades `adaptive_monitor.py` itself
already considers a minimum, then run `signal_calibration.py`,
`regime_analysis.py`, and `ablation_testing.py` for real — every other
recommendation in this report is downstream of that.

**Honesty test:** A skeptical quant would ask two things: (1) "your own
`operational_reliability.py` claims a model-unavailable fallback that
doesn't exist in the code — what else in your documentation is aspirational
rather than verified?" and (2) "your Risk-of-Ruin stress test uses a
Monte Carlo model that doesn't match your bot's actual position-sizing
logic — does that mean your real risk of ruin is better or worse than
0.578, and do you actually know?"

**Final question — what survives if every component without measurable
out-of-sample value is removed:** Unknown, same honest answer as the
other two bots in this project, for the same root reason — but for the
first time in this project, the infrastructure to actually answer it
exists, is wired to real instrumentation, and is one clean data run away
from a real answer instead of a rebuild.
