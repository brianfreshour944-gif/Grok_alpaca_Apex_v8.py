"""Fetch fresh perp/spot candles + funding from OKX (live) for forward testing.

Binance Vision archives only publish funding MONTHLY and lag several days, and
api.binance.com is geo-blocked here. OKX is reachable, carries 8h funding, and
has both the perp swap and the spot pair -- so it can drive a daily forward test
end to end. History reaches back ~100 days, which is plenty for forward accrual.

Writes a self-contained cache (Binance research caches stay untouched):
  <perp>/<SYM>_1h.csv          header + OKX perp candles (ms open_time)
  <perp>/funding/<SYM>.csv     ts,funding_rate (ms)
  <spot>/<SYM>_1h.csv          header + OKX spot candles (ms open_time)

Usage:
  python research/fetch_okx.py --days 60
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

try:
    from fetch_perp_binance import UNIVERSE
except ImportError:
    from research.fetch_perp_binance import UNIVERSE

OKX = "https://www.okx.com/api/v5"
UA = {"User-Agent": "Mozilla/5.0 (apex-carry-forward)"}
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


def _insts(sym: str) -> tuple[str, str]:
    base = sym.removesuffix("USDT")
    return f"{base}-USDT-SWAP", f"{base}-USDT"


def fetch_funding(sym: str, perp: Path, days: int) -> int:
    out = perp / "funding" / f"{sym}.csv"
    inst, _ = _insts(sym)
    cutoff = int(time.time() * 1000) - days * 86400 * 1000
    url = f"{OKX}/public/funding-rate-history?instId={inst}&limit=100"
    rows: list[tuple[int, str]] = []
    after = None
    for _ in range(12):
        d = _get(url + (f"&after={after}" if after else ""))
        data = d.get("data", [])
        if not data:
            break
        for it in data:
            rows.append((int(it["fundingTime"]), it["realizedRate"]))
        after = data[-1]["fundingTime"]
        if int(after) < cutoff:
            break
    if not rows:
        return 0
    rows = sorted({t: r for t, r in rows}.items())
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "funding_rate"])
        w.writerows(rows)
    return len(rows)


def fetch_candles(sym: str, spot: bool, dest_dir: Path, days: int) -> int:
    inst = _insts(sym)[1 if spot else 0]
    out = dest_dir / f"{sym}_1h.csv"
    cutoff = int(time.time() * 1000) - days * 86400 * 1000
    url = f"{OKX}/market/history-candles?instId={inst}&bar=1H&limit=100"
    rows: dict[int, list] = {}
    after = None
    for _ in range(40):
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
    dest_dir.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(KLINE_COLS)
        for t in sorted(rows):
            c = rows[t]
            w.writerow([c[0], c[1], c[2], c[3], c[4], c[5]])   # ts,o,h,l,c,vol
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--perp", default="okx_cache/perp")
    ap.add_argument("--spot", default="okx_cache/spot")
    ap.add_argument("--symbols", default=None)
    a = ap.parse_args(argv)
    syms = [s.strip() for s in a.symbols.split(",")] if a.symbols else UNIVERSE
    perp, spot = Path(a.perp), Path(a.spot)
    print(f"OKX: {len(syms)} symbols, last {a.days} days -> {perp}/ and {spot}/")
    jobs = [(fetch_funding, (s, perp, a.days), f"funding {s}") for s in syms]
    jobs += [(fetch_candles, (s, False, perp, a.days), f"perp {s}") for s in syms]
    jobs += [(fetch_candles, (s, True, spot, a.days), f"spot {s}") for s in syms]
    ok = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(fn, *args): name for fn, args, name in jobs}
        for fut in as_completed(futs):
            try:
                n = fut.result()
                ok += n > 0
                if n == 0:
                    print(f"  {futs[fut]}: EMPTY", flush=True)
            except Exception as e:                     # noqa: BLE001
                print(f"  {futs[fut]} FAILED: {type(e).__name__} {e}", flush=True)
    print(f"\nDONE: {ok}/{len(jobs)} series fetched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
