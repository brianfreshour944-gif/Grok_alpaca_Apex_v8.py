#!/usr/bin/env bash
# Daily forward-test accrual for the delta-neutral carry book. Paper only.
# Refreshes the OKX cache (funding + perp/spot candles), then appends the new
# days to forward_ledger.csv. Idempotent: safe to run more than once a day.
#
# Cron (01:00 UTC daily):
#   0 1 * * * cd /path/to/repo && bash research/run_forward_daily.sh >> forward_test.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."

DAYS=100                                   # OKX history window (~100 days)
START="$(date -u -d "${DAYS} days ago" +%F)"

python research/fetch_okx.py --days "${DAYS}"
python research/carry_forward_test.py \
  --perp-cache okx_cache/perp --spot-cache okx_cache/spot \
  --start "${START}" --ledger forward_ledger.csv
