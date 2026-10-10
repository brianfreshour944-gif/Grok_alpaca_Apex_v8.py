#!/usr/bin/env bash
# Daily Donchian-breakout paper run (PAPER ONLY — nothing is traded).
#
# Refresh daily OKX candles, then append one hypothetical book row to the
# ledger (idempotent by date, so a re-run adds nothing).
#
# Crontab (01:30 UTC):
#   30 1 * * * cd /path/to/repo && bash research/run_donchian_daily.sh >> donchian_daily.log 2>&1
set -euo pipefail

cd "$(dirname "$0")/.."

python research/fetch_okx_daily.py --days 30
python donchian_bot.py --paper --ledger donchian_ledger.csv
# mark the ledger to market: forward performance vs equal-weight buy & hold
python research/donchian_forward.py --ledger donchian_ledger.csv --cache okx_daily
