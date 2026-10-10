#!/usr/bin/env bash
# Container entrypoint: Donchian PAPER run. Nothing is traded.
#
# Fetches daily bars, appends one hypothetical book row to the ledger
# (idempotent by date), and marks it to market vs equal-weight buy & hold.
# No keys required. This is the safe default the image runs.
set -euo pipefail

cd "$(dirname "$0")"
LEDGER="${DONCHIAN_LEDGER:-donchian_ledger.csv}"

echo "Donchian PAPER run (no orders) at $(date -u +%FT%TZ)"

# Refresh daily candles; fall back to the on-disk cache if the network is out.
python research/fetch_okx_daily.py --days 2200 || echo "warn: OKX fetch failed, using cache"

python donchian_bot.py --paper --ledger "$LEDGER"

# Forward performance vs simply holding the same names.
python research/donchian_forward.py --ledger "$LEDGER" --cache okx_daily || true
