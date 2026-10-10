# Switching the bot from Grok Apex to the Donchian breakout

**What changed:** the container entrypoint the platform runs on deploy now points
at the **Donchian breakout bot**, not the Grok Apex CNN/GBDT bot. Default is the
**paper ledger** — nothing is traded.

**What did NOT change:** no code was deleted. The entire Grok Apex stack
(`main_bot.py`, `ml_predictor.py`, the `grok_gqa_v9_best.pth` model, the backtest
and research scripts) is still in the repo, just no longer the container command.

## Where the change lives

| file | before | after |
|---|---|---|
| `Dockerfile` | `CMD ["python", "main_bot.py"]` | `CMD ["bash", "donchian_paper_daily.sh"]` |
| `donchian_paper_daily.sh` | — | new: fetch bars → paper ledger → forward report |
| `donchian_live_daily.sh` | — | new: live entrypoint, double-gated |
| `Dockerfile.live` | — | new: live image variant (real orders) |
| `.env.example` | Apex header | Donchian vars added; Apex vars retained |
| `.github/workflows/tests.yml` | `pytest tests/` | runs the Donchian/xsec tests (keyless) |

`donchian_bot.py` itself is unchanged — it is already double-gated
(`--live` alone is refused) and never reads a key directly; live mode reuses
`config.py`'s `trading_client`, i.e. the **same Alpaca keys and `APCA_API_PAPER`
environment** as every other module.

## Three ways to run it

**1. Paper ledger (default, nothing traded, no keys needed).** This is what the
image runs:
```bash
docker run --rm <image>
# == bash donchian_paper_daily.sh
```

**2. Alpaca PAPER account (places simulated orders through the broker API).**
Uses your existing Alpaca paper keys:
```bash
docker run --rm --env-file .env -e APCA_API_PAPER=true donchian-live
# or, directly:
python donchian_bot.py --source alpaca --live --i-understand-the-risk
```

**3. LIVE money.** Real orders, deliberately awkward. Both switches required,
and `APCA_API_PAPER=false`:
```bash
docker build -f Dockerfile.live -t donchian-live .
docker run --rm --env-file .env \
  -e APCA_API_PAPER=false \
  -e DONCHIAN_I_UNDERSTAND_THE_RISK=true \
  donchian-live
```
The bot prints `LIVE ... alpaca_env=live` in its banner, so the environment in
effect is always visible. `APCA_API_PAPER` still defaults to `true`.

## Before you deploy — the honest caveats

This switch does not make the bot profitable; it makes it *the thing that runs*.
From `docs/DONCHIAN_BREAKOUT.md` and the forward ledger:

- The breakout **underperforms buy & hold** on these assets and its edge has been
  **thin/negative since 2024**.
- The 400-day forward paper read was **−5.9%** (vs −44% for holding, in a down
  tape) — drawdown control, not alpha.
- Live crypto on Alpaca is **spot-only** (no shorting), so the book is long-only
  by construction.

Paper-trade first. Do not put real money behind it on these numbers.

## Rollback (how to go back to Grok Apex)

One line — restore the original container command:
```bash
git revert <this-commit>          # if it was committed
# or, manually:
sed -i 's|CMD \["bash", "donchian_paper_daily.sh"\]|CMD ["python", "main_bot.py"]|' Dockerfile
```
Then rebuild/redeploy. Nothing else needs undoing: the Apex code, model, and
config were never removed, and the env vars it reads are all still in
`.env.example`.

## Scheduling (optional)

The paper entrypoint is meant to run daily. Crontab (01:30 UTC):
```
30 1 * * * cd /app && bash donchian_paper_daily.sh >> donchian_daily.log 2>&1
```
