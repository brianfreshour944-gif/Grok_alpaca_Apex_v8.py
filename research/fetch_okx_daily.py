"""Fetch DAILY candles from OKX (live) for the Donchian breakout research.

OKX is reachable here without auth and carries long daily history, so it is the
simplest reproducible source for a daily-bar breakout study. Writes one CSV per
symbol to `<cache>/<SYM>_1D.csv` with a header row:

    open_time,open,high,low,close,volume

Usage:
  python research/fetch_okx_daily.py --days 2000
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

UNIVERSE = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
            "DOGEUSDT", "LTCUSDT", "LINKUSDT", "DOTUSDT", "AVAXUSDT", "BCHUSDT",
            "TRXUSDT", "ATOMUSDT", "ETCUSDT"]

OKX = "https://www.okx.com/api/v5"
UA = {"User-Agent": "Mozilla/5.0 (apex-donchian-research)"}
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume"]


def _get(url: str, tries: int = 4) -> dict:
    last = None
    for _ in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                d = json.loads(r.read())
            if d.get("code") == "0":
                return d
            last = d.get("msg", d.get("code"))
        except (urllib.error.URLError, OSError, ValueError) as e:
            last = str(e)
        time.sleep(1.0)
    return {"code": "err", "msg": last, "data": []}


def fetch_daily(sym: str, dest: Path, days: int, bar: str = "1D") -> int:
    inst = f"{sym.removesuffix('USDT')}-USDT"
    out = dest / f"{sym}_1D.csv"
    cutoff = int(time.time() * 1000) - days * 86400 * 1000
    url = f"{OKX}/market/history-candles?instId={inst}&bar={bar}&limit=100"
    rows: dict[int, list] = {}
    after = None
    for _ in range(60):
        d = _get(url + (f"&after={after}" if after else ""))
        data = d.get("data", [])
        if not data:
            break
        for c in data:
            rows[int(c[0])] = c
        after = data[-1][0]
        if int(after) < cutoff:
            break
    if not rows:
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(KLINE_COLS)
        for t in sorted(rows):
            c = rows[t]
            w.writerow([c[0], c[1], c[2], c[3], c[4], c[5]])
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=2000)
    ap.add_argument("--cache", default="okx_daily")
    ap.add_argument("--symbols", default=None)
    a = ap.parse_args(argv)
    syms = [s.strip() for s in a.symbols.split(",")] if a.symbols else UNIVERSE
    dest = Path(a.cache)
    print(f"OKX daily: {len(syms)} symbols, last {a.days} days -> {dest}/")
    ok = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(fetch_daily, s, dest, a.days): s for s in syms}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                n = fut.result()
                ok += n > 0
                print(f"  {s}: {n} rows" + ("  EMPTY" if n == 0 else ""), flush=True)
            except Exception as e:                     # noqa: BLE001
                print(f"  {s} FAILED: {type(e).__name__} {e}", flush=True)
    print(f"\nDONE: {ok}/{len(syms)} symbols -> {dest}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
