#!/usr/bin/env bash
# Container entrypoint: Donchian PAPER run. Nothing is traded.
#
# Decisions come from ALPACA's own daily bars (this is an Alpaca deployment; OKX's
# public API is geo-blocked here). The bars are saved so the forward report marks
# the ledger on the SAME prices. If Alpaca is unreachable the bot degrades to the
# OKX cache and prints which source it actually used.
#
# This replaces a long-running service (the old Apex loop), so it loops with an
# idle sleep instead of exiting — otherwise the platform restarts it in a tight
# loop. Set DONCHIAN_ONESHOT=1 to run once and exit (cron/one-shot use).
set -uo pipefail

cd "$(dirname "$0")"
LEDGER="${DONCHIAN_LEDGER:-donchian_ledger.csv}"
CACHE="${DONCHIAN_CACHE:-alpaca_daily}"
IDLE="${DONCHIAN_IDLE_SLEEP:-3600}"

while true; do
  echo "Donchian PAPER run (no orders) at $(date -u +%FT%TZ)"

  # Never let a venue outage kill the loop (that crash-loops the platform).
  python donchian_bot.py --paper --source alpaca --save-cache "$CACHE" --ledger "$LEDGER" \
    || echo "cycle failed (continuing)"

  # Forward performance vs simply holding the same names, on the same bars.
  if [ -f "$LEDGER" ]; then
    python research/donchian_forward.py --ledger "$LEDGER" --cache "$CACHE" || true
  fi

  if [ "${DONCHIAN_ONESHOT:-0}" = "1" ] || [ "$IDLE" -le 0 ]; then
    break
  fi
  echo "idle ${IDLE}s before the next run (DONCHIAN_ONESHOT=1 exits instead)"
  sleep "$IDLE"
done
