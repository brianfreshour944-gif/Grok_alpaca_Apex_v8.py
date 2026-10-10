#!/usr/bin/env bash
# Container entrypoint: Donchian PAPER run. Nothing is traded.
#
# Decisions come from ALPACA's own daily bars: this is an Alpaca deployment, and
# OKX's public API is geo-blocked (HTTP 403) from US hosts, so `--source okx`
# would silently serve a frozen cache. The fetched bars are saved to a close
# cache so the forward report marks the ledger on the SAME prices. If Alpaca keys
# are missing the bot degrades to the on-disk OKX cache (trimmed to
# Alpaca-tradeable symbols) and prints which source it actually used.
set -euo pipefail

cd "$(dirname "$0")"
LEDGER="${DONCHIAN_LEDGER:-donchian_ledger.csv}"
CACHE="${DONCHIAN_CACHE:-alpaca_daily}"

echo "Donchian PAPER run (no orders) at $(date -u +%FT%TZ)"

python donchian_bot.py --paper --source alpaca --save-cache "$CACHE" --ledger "$LEDGER"

# Forward performance vs simply holding the same names, on the same bars.
python research/donchian_forward.py --ledger "$LEDGER" --cache "$CACHE" || true
