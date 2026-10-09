"""Fetch historical Alpaca 15m crypto bars into bt_cache/ for retraining.

Pulls the config tradeable universe over a long window via the data REST API
(raw HTTP so pagination is explicit), one CSV per symbol named
<SYM>_15m_<start>_<end>.csv -> what research/train_model.load_bars expects.

Usage:
  APCA_API_KEY_ID=... APCA_API_SECRET_KEY=... \
  python research/fetch_alpaca_bars.py --start 2021-06-01 --end 2026-10-09
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "LTC/USD",
           "AVAX/USD", "LINK/USD", "ADA/USD", "BCH/USD", "DOT/USD"]
BASE = "https://data.alpaca.markets/v1beta3/crypto/us/bars"


def _headers() -> dict:
    kid = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
    sec = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    if not kid or not sec:
        sys.exit("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY")
    return {"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec,
            "User-Agent": "apex-retrain/1.0"}


def fetch_symbol(sym: str, start: str, end: str, hdrs: dict) -> list[dict]:
    rows: list[dict] = []
    token = None
    while True:
        q = {"symbols": sym, "timeframe": "15Min", "start": start, "end": end,
             "limit": "10000"}
        if token:
            q["page_token"] = token
        url = BASE + "?" + urllib.parse.urlencode(q)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=hdrs), timeout=60) as r:
                    payload = json.loads(r.read())
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(1.5 * (attempt + 1))
        rows.extend(payload.get("bars", {}).get(sym, []))
        token = payload.get("next_page_token")
        if not token:
            break
        if len(rows) % 100000 < 10000:
            print(f"    {sym}: {len(rows):,} bars ...", flush=True)
        time.sleep(0.25)                               # be gentle on rate limits
    return rows


def write_csv(sym: str, rows: list[dict], start: str, end: str) -> Path:
    out = Path("bt_cache") / f"{sym.replace('/', '_')}_15m_{start}_{end}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("timestamp,open,high,low,close,volume,vwap,trade_count\n")
        for b in rows:
            f.write(f"{b['t']},{b['o']},{b['h']},{b['l']},{b['c']},"
                    f"{b['v']},{b.get('vw', b['c'])},{b.get('n', 1)}\n")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-06-01")
    ap.add_argument("--end", default="2026-10-09")
    ap.add_argument("--symbols", default=None)
    a = ap.parse_args(argv)
    syms = [s.strip() for s in a.symbols.split(",")] if a.symbols else SYMBOLS
    hdrs = _headers()
    for sym in syms:
        t0 = time.time()
        rows = fetch_symbol(sym, a.start, a.end, hdrs)
        if not rows:
            print(f"  {sym}: NO DATA", flush=True)
            continue
        p = write_csv(sym, rows, a.start, a.end)
        print(f"  {sym}: {len(rows):,} bars -> {p} ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
