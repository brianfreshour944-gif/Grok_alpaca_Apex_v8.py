#!/usr/bin/env bash
# Daily Donchian-breakout paper run (PAPER ONLY — nothing is traded).
#
# Decisions come from Alpaca's daily bars (this is an Alpaca deployment; OKX's
# public API is geo-blocked here). The bars are saved so the forward report marks
# the ledger on the same prices. Without keys it degrades to the OKX cache.
#
# Crontab (01:30 UTC):
#   30 1 * * * cd /path/to/repo && bash research/run_donchian_daily.sh >> donchian_daily.log 2>&1
set -euo pipefail

cd "$(dirname "$0")/.."

LEDGER="${DONCHIAN_LEDGER:-donchian_ledger.csv}"
CACHE="${DONCHIAN_CACHE:-alpaca_daily}"

python donchian_bot.py --paper --source alpaca --save-cache "$CACHE" --ledger "$LEDGER"
# mark the ledger to market: forward performance vs equal-weight buy & hold
python research/donchian_forward.py --ledger "$LEDGER" --cache "$CACHE" || true
