#!/usr/bin/env bash
# Container entrypoint: Donchian LIVE run — REAL ORDERS.
#
# Trading real money requires BOTH explicit switches below. Do not set
# DONCHIAN_I_UNDERSTAND_THE_RISK=true unless you have read docs/DONCHIAN_BREAKOUT.md
# and accepted that the backtest underperforms buy & hold and has decayed since
# 2024. The bot itself additionally refuses --live without the risk flag.
#
# To run on Alpaca's PAPER account instead (still simulated, but places orders
# through the broker API rather than the local ledger), set APCA_API_PAPER=true
# and pass DONCHIAN_LIVE_ARGS="--live --i-understand-the-risk" is NOT needed --
# use the paper entrypoint (donchian_paper_daily.sh) for the ledger instead.
set -euo pipefail

cd "$(dirname "$0")"

if [ "${DONCHIAN_I_UNDERSTAND_THE_RISK:-}" != "true" ]; then
  echo "refused: set DONCHIAN_I_UNDERSTAND_THE_RISK=true to trade live." >&2
  exit 2
fi

echo "Donchian LIVE run (REAL ORDERS) at $(date -u +%FT%TZ)"

# Decisions from the venue you actually trade on.
python donchian_bot.py --source alpaca --live --i-understand-the-risk \
  --ledger "${DONCHIAN_LEDGER:-donchian_ledger.csv}"
