# Correctness Audit — Grok_alpaca_Apex_v8.py

**Date:** 2026-09-20
**Method:** Actual execution wherever possible (isolated venv with the exact
pinned `alpaca-py==0.33.0`, real `pandas`/`numpy`/`torch`, the repo's own
test suite, and hand-written reproduction scripts run against the real
`grok_gqa_v9_best.pth` model file) — not just reading. Every finding below
states explicitly whether it was **verified by running code** or is a
**read-only / static-analysis finding**. Static analysis (`pyflakes`) was
run across every top-level `.py` file to make the import/undefined-name
sweep exhaustive rather than sampled.

**Scope note (honesty about depth):** The live trading path — `main_bot.py`,
`orders.py`, `portfolio.py`, `database.py`, `ml_predictor.py`,
`data_feeds.py`, `config.py`, `money.py`, `regime.py`, `exit_logic.py`,
`notifications.py`, `api_utils.py`, `shadow_model.py`,
`experience_capture.py`, `feature_engineering.py`, and the champion/drift
tooling (`promotion_gate.py`, `drift_monitor.py`, `adaptive_monitor.py`) —
got a full read + reproduction pass. The remaining ~16 offline
analysis/diagnostic scripts (`ablation_testing.py`, `backtest_validation.py`,
`complexity_budget.py`, `execution_cost_analysis.py`, `exit_analysis.py`,
`feature_analysis.py`, `monte_carlo_failure.py`, `operational_reliability.py`,
`performance_dashboard.py`, `portfolio_crash_test.py`, `regime_analysis.py`,
`risk_analysis.py`, `signal_calibration.py`, `stress_test.py`,
`stress_test_market_drops.py`, `train_walkforward.py`,
`walkforward_robustness.py`) got the full codebase-wide static sweep
(categories 1 and 2 below are exhaustive across **all** files, verified by
tool, not sampling) but not a full manual read-through — flagging this
per the report's own honesty requirement rather than presenting uniform
depth I didn't actually do.

---

## Finding 1 — `add_features()` crashes unconditionally on every call

- **Severity:** crash (silently swallowed one level up — see Finding 2 for the compounding effect)
- **Verified by:** actually running code, including against the real production model file
- **File:** `feature_engineering.py:239-242`, root cause in `_sanitize()` at `feature_engineering.py:85-90`

```python
trade_size_raw = _sanitize(
    np.where(tc_is_reliable > 0.5, volume / tc_raw, volume),
    fill=0.0,
)
```

`np.where(...)` **always** returns a raw `numpy.ndarray`, never a pandas
Series, regardless of whether its inputs were Series. `_sanitize()` then
calls `.replace([np.inf, -np.inf], fill)` on the result — `.replace()` is a
pandas Series/DataFrame method that doesn't exist on `ndarray`, so this
raises `AttributeError: 'numpy.ndarray' object has no attribute 'replace'`
on **every single call**, for any input, unconditionally. This is not an
edge case — it's a total break of the feature pipeline.

**Reproduced directly:**
```
AttributeError: 'numpy.ndarray' object has no attribute 'replace'
  File "feature_engineering.py", line 239, in add_features
  File "feature_engineering.py", line 88, in _sanitize
    .replace([np.inf, -np.inf], fill)
```
Confirmed with a minimal synthetic OHLCV frame and with the repo's own
`tests/test_feature_engineering.py` (7 of 7 tests in that file fail on
this exact line).

**When introduced:** `git blame` shows this line was added **2026-09-17**,
three days before this audit, in commit `4f154ac2` — ironically titled
"fix: data quality and feature integrity improvements." The repo has a
GitHub Actions workflow (`.github/workflows/tests.yml`) that runs
`pytest tests/ -v` on every push/PR to `main`, which would catch this
(confirmed locally: 11 tests fail). I could not check whether that
workflow actually ran/failed on this push (no `gh`/network access from
this environment) — noting the limitation rather than guessing.

**Minimal fix (proposed, not applied):**
```python
trade_size_raw = _sanitize(
    pd.Series(np.where(tc_is_reliable > 0.5, volume / tc_raw, volume), index=volume.index),
    fill=0.0,
)
```
or simpler, avoid `np.where` entirely:
```python
trade_size_raw = _sanitize(volume.where(tc_is_reliable <= 0.5, volume / tc_raw), fill=0.0)
```

---

## Finding 2 — Consequence of Finding 1: the transformer model has produced zero real predictions since 2026-09-17

- **Severity:** crash / correctness (compounding — this is the actual live-trading impact of Finding 1)
- **Verified by:** actually running `SafeMLPredictor.predict_batch()` against the real `grok_gqa_v9_best.pth` and `feature_scaler.pkl`
- **File:** `ml_predictor.py:269-333` (torch path), `ml_predictor.py:335-358` (sklearn/GBT path)

`predict_batch()`'s torch path wraps its entire body (including the
`add_features()` call at line 276) in one `try/except Exception`. Since
Finding 1 makes `add_features()` raise on every call, this `except` block
(line 331-333) fires every time:
```python
except Exception as e:
    print(f"Batch prediction error: {e}")
    return {symbol: 0.5 for symbol in df_dict.keys()}
```

**Reproduced end-to-end with the real model:**
```
✅ Model weights loaded from grok_gqa_v9_best.pth
✅ Scaler loaded from .\feature_scaler.pkl
Batch prediction error: 'numpy.ndarray' object has no attribute 'replace'
predict_batch result: {'BTC/USD': 0.5}
predictor.last_features (should be non-empty if working): {}
```

**Concrete, traced consequences (all verified by reading the exact
downstream code, using values confirmed by the reproduction above):**

1. **Every symbol gets exactly `0.5` every cycle, forever.** This is not
   "the model is cautious" — it is a hardcoded fallback carrying zero
   information from the model, market data, or anything else.
2. **`predictor.last_features` is always `{}`.** `main_bot.py:812` passes
   `features=predictor.last_features.get(symbol)` into
   `log_entry_experience()` on every BUY — meaning **every entry logged to
   `live_experiences.jsonl` since 2026-09-17 has an empty `features` dict**,
   even though `experience_capture.py`'s own module docstring states its
   entire purpose is "the FEATURE VECTOR the champion model saw at decision
   time." The step-0 retraining data pipeline has been silently capturing
   nothing usable.
3. **Buy-gate arithmetic:** `main_bot.py:661` gates entries on
   `trend == "up" and signal > regime_params["buy_signal"]`. With `signal`
   pinned at `0.5` and `config.py`'s `get_regime_params()` offsets
   (`BUY_SIGNAL=0.51` default; wild=+0.06 → 0.57, normal=+0.00 → 0.51,
   quiet=−0.04 → 0.47), **`0.5` can only ever exceed the threshold in the
   "quiet" regime** (0.5 > 0.47). In "normal" and "wild" regimes the bot
   cannot buy at all right now. In "quiet" regime it can still buy, but
   that decision is driven entirely by `trend` (a separate, real EMA-based
   indicator) and the regime classification — **not by any model
   prediction**, despite every log line since 2026-09-17 reporting an "ML
   Signal" that looks like a real number.
4. **The sklearn/GBT champion path is equally broken** — `_predict_batch_sklearn()`
   (line 335-358) also calls `add_features()` and hits the same crash,
   confirmed by the repo's own `tests/test_ml_predictor.py::test_sklearn_champion_predict`
   failing with `KeyError: 'SOL/USD'` (because `last_features` is never populated).

**Minimal fix:** fix Finding 1's root cause; this finding has no independent
fix. **Recommend, once fixed:** review any trades placed between
2026-09-17 and the fix, since none of them were driven by real model
predictions regardless of what the logs show.

---

## Finding 3 — `backfill_trade_if_missing()` always raises `NameError` when it actually needs to do anything

- **Severity:** crash, silently swallowed
- **Verified by:** actually running code (mocked DB connection, forced the "needs backfill" branch)
- **File:** `database.py:282`

```python
cur.execute("""
    INSERT INTO trades
        (bot_name, exchange, symbol, side, price, quantity,
         value, fee, fill_price, order_id, timestamp)
    VALUES (%s, 'Alpaca', %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, NOW()))
""", (bot_name, symbol, side, fill_price or 0.0, qty, value,
      fee, fill_price, order_id, created))
```

`bot_name` (lowercase) is referenced here, but `backfill_trade_if_missing(order)`
takes no such parameter and no local variable of that name is ever assigned
anywhere in the function. `config.py` exports `BOT_NAME` (uppercase) at
module level, imported at the top of `database.py` — this is a case-typo,
not a genuinely missing import, but the runtime effect is identical: a
`NameError`. Confirmed exhaustively via `pyflakes` (`undefined name
'bot_name'` — the **only** undefined-name hit across the entire codebase)
and by direct execution:

```
ERROR | backfill_trade_if_missing failed for order order-123: name 'bot_name' is not defined
backfill_trade_if_missing returned: False
```

This fires precisely in the scenario the function exists for — the
"needs backfill" branch (`cur.fetchone() is None`, i.e. the trade row is
genuinely missing) — caught by the function's own broad `except Exception
as e: logger.error(...); return False`, so the caller
(`main_bot.py:244`'s startup reconciliation loop) just sees `False` and
silently moves on, believing nothing needed backfilling.

**Impact:** the crash-recovery trade-log reconciliation feature described
in this function's own docstring ("if the process died between
submit_order() success and the record_trade() write... insert one from
the exchange's own order data") **has never worked** — any trade lost to
a crash between order submission and the DB write is permanently
unrecoverable by this mechanism, even though the bot logs no error
indicating that.

**Minimal fix (proposed, not applied):** change `bot_name` → `BOT_NAME` on
line 282 (the constant is already imported at the top of the file, just
unused everywhere else in it — confirmed by `pyflakes`'s
`'config.BOT_NAME' imported but unused` on `database.py:7`, which is
itself indirect confirmation this was meant to be referenced somewhere).

---

## Finding 4 — `get_all_positions()` can silently return `{}` (believing the account is flat) if Alpaca returns a null `market_value`/`current_price` for any single position

- **Severity:** correctness (real per the SDK's own type contract; not confirmed to have occurred in live production data — see below)
- **Verified by:** actually running code, using the exact field types alpaca-py 0.33.0 declares as legal
- **File:** `portfolio.py:66-84`

```python
def get_all_positions() -> dict:
    try:
        positions = trading_client.get_all_positions()
        return {
            p.symbol: {
                "qty":           float(p.qty),
                "avg_entry":     float(p.avg_entry_price),
                "market_value":  float(p.market_value),
                "current_price": float(p.current_price),
            }
            for p in positions
        }
    except Exception as e:
        logger.error(f"get_all_positions failed: {e}")
        return {}
```

Checked against the installed `alpaca-py==0.33.0` source
(`alpaca/trading/models.py:102-151`): `Position.market_value` and
`Position.current_price` are both typed `Optional[str] = None` — the SDK
itself documents these as sometimes absent. `float(None)` raises
`TypeError`. Because the whole fetch is one dict comprehension inside one
try/except, **a null field on any single held position makes the entire
function return `{}`** — not just omit that one position, but report zero
held positions, for every symbol, for that call.

**Reproduced directly:**
```python
fake_pos.market_value = None   # legal per SDK type hint
# ...
result = portfolio.get_all_positions()
# ERROR | get_all_positions failed: float() argument must be a string or a real number, not 'NoneType'
# RESULT: {}
```

**Traced downstream impact in `main_bot.py`:** `current_positions = {}`
would make `open_count=0`, `total_value=0`, `held_symbols=set()` — the
entry logic would think there's full headroom to buy more (risking
over-exposure beyond intended limits), and the exit logic's
`has_position` check (`main_bot.py:563-568`) would be `False` for every
actually-held symbol, **skipping stop-loss/trailing-stop/max-hold
monitoring entirely for that cycle** on positions that are, in reality,
still open and unprotected.

**Honesty note:** I have not confirmed this null condition actually
occurs on live Alpaca crypto positions (unlike equities, crypto trades
24/7, so the plausible-for-equities "no data outside market hours" cause
doesn't obviously apply) — this is a reproduced structural risk given
the SDK's own type contract, not a confirmed production incident. Worth
a defensive fix regardless, since the failure mode (silently believing
you're flat while ~~one~~ or more positions go unmonitored) is severe if
it ever does occur.

**Minimal fix (proposed, not applied):** build the dict per-position with
its own try/except (skip and log just that one position on failure)
instead of one comprehension for all positions, and fall back to `0.0`
for `market_value`/`current_price` rather than raising:
```python
result = {}
for p in positions:
    try:
        result[p.symbol] = {
            "qty": float(p.qty),
            "avg_entry": float(p.avg_entry_price),
            "market_value": float(p.market_value) if p.market_value is not None else 0.0,
            "current_price": float(p.current_price) if p.current_price is not None else 0.0,
        }
    except (TypeError, ValueError) as e:
        logger.error(f"get_all_positions: skipping {p.symbol}, bad data: {e}")
return result
```

---

## Finding 5 — Shadow-inference block raises `UnboundLocalError` on every cycle whenever a GBT challenger exists (compounding with Finding 2)

- **Severity:** silent-failure
- **Verified by:** actually running the exact variable-scoping pattern used in the real code
- **File:** `main_bot.py:545-561`

```python
try:
    _shadow = get_shadow_gbt()
    if _shadow.available():
        _feat_row = predictor.last_features.get(symbol)
        if _feat_row:
            _gbt_prob = _shadow.predict_row(_feat_row)
        log_shadow_prediction(
            symbol,
            gbt_prob=_gbt_prob,          # <- referenced unconditionally
            ...
        )
except Exception as sh_err:
    logger.debug(f"Shadow inference skipped for {symbol}: {sh_err}")
```

`_gbt_prob` is only assigned inside `if _feat_row:`. When `_feat_row` is
falsy (`predictor.last_features.get(symbol)` returns `None` whenever that
symbol's feature window was too short, or — per Finding 2 — **always**,
since `add_features()` currently never lets `last_features` get
populated at all), `_gbt_prob` is referenced before assignment.

**Reproduced directly:**
```
UnboundLocalError("cannot access local variable '_gbt_prob' where it is not associated with a value")
```

Caught by `except Exception as sh_err: logger.debug(...)` — at `DEBUG`
level, invisible under the default `LOG_LEVEL=INFO` (`config.py:18`).
Given Finding 2 means `last_features` is currently *always* empty, this
fires on **every symbol, every cycle**, completely silently, whenever a
`gbt_challenger.joblib` file exists on disk (`_shadow.available()` only
checks `os.path.exists`).

**Impact:** purely diagnostic — `shadow_model.py`'s own docstring says
"This model NEVER trades," so this doesn't affect live trading decisions,
only the step-3/step-4 champion-vs-challenger bake-off's data collection.

**Minimal fix (proposed, not applied):**
```python
_gbt_prob = None
if _feat_row:
    _gbt_prob = _shadow.predict_row(_feat_row)
log_shadow_prediction(symbol, gbt_prob=_gbt_prob, ...)
```

---

## Finding 6 — `drift_monitor.py`'s Discord alert always fails: `httpx` is used but never installed

- **Severity:** silent-failure
- **Verified by:** actually running code
- **File:** `drift_monitor.py:272-285`; missing from `requirements.txt`/`requirements-dev.txt`

```python
import httpx
async with httpx.AsyncClient() as client:
    await client.post(DISCORD_WEBHOOK_URL, ...)
```

`httpx` does not appear anywhere in `requirements.txt` or
`requirements-dev.txt`, and is not pulled in transitively by any pinned
dependency (confirmed: installing the repo's exact pinned requirement set
into a clean venv does not provide `httpx`). Every other Discord-alert
path in this codebase (`notifications.py`) correctly uses `aiohttp`,
which *is* declared.

**Reproduced directly:**
```
WARNING | Failed to send drift alert: No module named 'httpx'
send_drift_alert returned: False
```

**Impact:** `drift_monitor.py` is a standalone diagnostic script (not
imported by `main_bot.py`), so this doesn't affect live trading — but its
one alerting mechanism, the entire point of proactively notifying about
model drift, silently no-ops in any environment built from this repo's
own declared dependencies.

**Minimal fix (proposed, not applied):** either add `httpx` to
`requirements.txt`, or (simpler, avoids a second HTTP client dependency)
replace the `httpx` call with the already-available
`notifications.send_discord_alert()`.

---

## Finding 7 — `drift_monitor.py`: computed `title` variable is discarded; the Discord embed title never reflects severity

- **Severity:** correctness (cosmetic — the description text is still correct)
- **Verified by:** reading code (confirmed via `pyflakes`'s "assigned but never used", not run against a real Discord webhook since Finding 6 blocks that path entirely anyway)
- **File:** `drift_monitor.py:258, 278`

```python
title = "Model Drift Detected" if metrics.get("is_drifting") else "Model Performance Warning"
color = 0xFF0000 if metrics.get("is_drifting") else 0xFFAA00
...
"embeds": [{
    "title": f"Model Drift Detected",   # <- hardcoded, ignores `title`
    "description": description,
    "color": color,                      # <- this one correctly varies
```

`color` correctly reflects severity; `title` was clearly meant to as well
(same conditional, right above) but the embed hardcodes the drift-case
string regardless. A "Performance Warning" (softer signal) would be
mislabeled "Model Drift Detected" (harder signal) in the alert title. Low
impact since Finding 6 means this alert never actually sends today, but
worth fixing alongside it.

**Minimal fix (proposed, not applied):** use the `title` variable:
`"title": title,`.

---

## Category-by-category summary (per the requested process)

**1. Async/await mismatches — checked exhaustively across every file with `await`, confirmed CLEAN.**
Every `await` target in `main_bot.py`, `orders.py`, `portfolio.py`,
`data_feeds.py`, `api_utils.py`, `drift_monitor.py` was traced to its
definition and confirmed genuinely `async def` (cross-referenced via
`grep` across the whole repo, listed and checked one by one — see audit
transcript). Every `async def` in the codebase was confirmed to be called
with `await`, `asyncio.gather`, `asyncio.create_task`, or `asyncio.run`
somewhere — no orphaned coroutines. This is a real, verified negative
result, not an assumption.

**2. Import/name checks — exhaustive via `pyflakes` across all 34 top-level files.**
Exactly **one** undefined-name bug exists codebase-wide: Finding 3
(`database.py:282`). Dozens of unused-import findings exist (mostly
`sys`, `pathlib.Path`, and specific `config.*` constants unused in the
16 offline analysis scripts, apparently copy-pasted boilerplate) — these
are style-only, not correctness bugs, and are not itemized individually
here to keep the report focused on things that actually break at
runtime.

**3. Shared/global mutable state — reviewed, no violations found.**
`TradingBotState` (`main_bot.py`) and `PortfolioState` (`portfolio.py`)
correctly use dicts keyed by symbol for everything that varies per-entity
(`cooldown_until`, `entry_time`, `latest_signals`, `highest_prices`,
`sell_retry_cooldown`, `pending_exit_until`). The genuinely global
singletons (`circuit_breaker`, `latency_tracker`, `predictor`,
`_singleton` in `shadow_model.py`) are correctly scoped as
whole-exchange/whole-model concepts, not per-symbol state incorrectly
collapsed to one — verified by reading their actual usage, not assumed
from naming.

**4. Reproduction — see Findings 1, 2, 3, 4, 5, 6 above; each was actually executed, not just reasoned about.**

**5. Boundary/short-data cases — checked in `data_feeds.py` and `regime.py`, both correct.**
`get_clean_ohlcv_dataframe()` re-checks `len(df) < SEQUENCE_LEN` *after*
the `close > 0` filter (line 144), not just before it (line 122) — the
exact ordering bug class the process asked me to look for is absent
here. `regime.py`'s `compute_regime_and_trend()` explicitly guards
`pd.isna(atr_pct)` and fails safe to `"normal"/"neutral"` rather than
letting a NaN silently pass every comparison as `False` — also correct,
and the code's own comment explains exactly why (matches actual
behavior, verified).

**6. Duplicate/conflicting files — checked, none found.** No file with a
numeric/date suffix pattern exists anywhere in the repo. `Dockerfile`
`CMD ["python", "main_bot.py"]` matches the file's own header comment
("Entry point") — consistent, no ambiguity.

**7. Comments vs. actual code — spot-checked throughout; two mismatches found (Findings 3, 7), both above.
Two explicit SDK-related comments were checked against the real installed
package and found accurate**: `orders.py:119-121` and
`database.py:270-271` both claim "alpaca-py 0.33.0 Order model has no
'commission' field" — confirmed true by reading
`alpaca/trading/models.py`'s actual `Order` class (no such field exists).

**8. Report format — followed above per finding.**

**9. SDK/API attribute verification — done against the actually-installed
`alpaca-py==0.33.0` (matching the repo's own pin, not whatever happened
to be globally installed, which was `0.44.0` — an 11-minor-version gap
that would have made this check meaningless against the wrong version).**
- `Order.filled_avg_price`, `Order.filled_qty`: confirmed to exist (`Optional[Union[str, float]]`) — `orders.py`'s usage is safe.
- `Order.commission`: confirmed **not to exist** — the code's own defensive comments are accurate, and `getattr(order, "commission", None)` correctly never raises.
- `Position.market_value`, `Position.current_price`: confirmed to exist but as `Optional[str] = None` — see Finding 4.
- `TradeAccount.equity`, `.buying_power`, `.cash`: also `Optional[str] = None`. `portfolio.get_buying_power()` handles this safely (function-level try/except, fails to `0.0`, a safe direction). `main_bot.py:324`'s `equity = float(account.equity)` is only caught by the outer per-cycle catch-all (`logger.exception` + 30s sleep + retry) rather than a narrower guard — lower severity than Finding 4 because the blast radius is "skip this whole cycle and retry" rather than "silently believe the account is flat," but the same root cause (trusting a documented-Optional SDK field) applies. Not written up as a separate numbered finding to avoid padding the report with a repeat of the same underlying lesson.
