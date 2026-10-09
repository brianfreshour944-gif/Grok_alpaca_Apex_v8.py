"""Fetch Binance USDⓈ-M perpetual history (klines + funding) from Binance Vision.

Binance Vision monthly archives are the only long, complete source reachable
here (api.binance.com is geo-blocked; api.binance.us is spot-only). Perp klines
carry taker-buy volume (order flow); fundingRate is the 8-hourly carry.

Output (gitignored):
  perp_cache/<SYM>_1h.csv          open_time,o,h,l,c,volume,quote_volume,
                                   count,taker_buy_volume,taker_buy_quote_volume
  perp_cache/funding/<SYM>.csv     ts,funding_rate

Usage:
  python research/fetch_perp_binance.py --start 2021-01 --end 2026-10
"""
from __future__ import annotations

import argparse
import csv
import io
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

V = "https://data.binance.vision/data/futures/um"
UNIVERSE = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
            "DOGEUSDT", "LTCUSDT", "LINKUSDT", "DOTUSDT", "AVAXUSDT", "BCHUSDT",
            "TRXUSDT", "ATOMUSDT", "ETCUSDT"]
UA = {"User-Agent": "Mozilla/5.0 (apex-perp-research)"}


def _months(start: str, end: str) -> list[str]:
    ys, ms = map(int, start.split("-"))
    ye, me = map(int, end.split("-"))
    out = []
    y, m = ys, ms
    while (y, m) <= (ye, me):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def _get_zip_csv(url: str) -> list[list[str]] | None:
    """Download a Binance Vision zip and return its single CSV as rows. 404 -> None."""
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                blob = r.read()
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                name = z.namelist()[0]
                text = z.read(name).decode("utf-8", "replace")
            return [row for row in csv.reader(io.StringIO(text)) if row]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == 3:
                return None
        except (urllib.error.URLError, OSError, zipfile.BadZipFile, ValueError):
            if attempt == 3:
                return None
    return None


def _download(url: str, dest: Path) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"
    blob = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                blob = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return "404"
            if attempt == 3:
                return "err"
        except (urllib.error.URLError, OSError):
            if attempt == 3:
                return "err"
    if blob is None:
        return "err"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(blob)
    tmp.rename(dest)
    return "ok"


def fetch_klines(sym: str, months: list[str], cache: Path) -> int:
    """Concatenate monthly 1h kline files into one CSV per symbol."""
    out = cache / f"{sym}_1h.csv"
    if out.exists():
        return 0
    rows: list[list[str]] = []
    header: list[str] | None = None
    for mo in months:
        url = f"{V}/monthly/klines/{sym}/1h/{sym}-1h-{mo}.zip"
        csv_rows = _get_zip_csv(url)
        if not csv_rows:
            continue
        if csv_rows[0] and not csv_rows[0][0].lstrip("-").isdigit():
            header = csv_rows[0]
            csv_rows = csv_rows[1:]
        rows.extend(csv_rows)
    if not rows:
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(header[:11])
        for r in rows:
            w.writerow(r[:11])
    return len(rows)


def fetch_funding(sym: str, months: list[str], cache: Path) -> int:
    out = cache / "funding" / f"{sym}.csv"
    if out.exists():
        return 0
    ts, rate = [], []
    for mo in months:
        url = f"{V}/monthly/fundingRate/{sym}/{sym}-fundingRate-{mo}.zip"
        csv_rows = _get_zip_csv(url)
        if not csv_rows:
            continue
        head = [c.lower() for c in csv_rows[0]]
        # schema A: calc_time,funding_interval_hours,last_funding_rate
        # schema B: fundingTime,fundingRate
        if "calc_time" in head:
            ti, ri = head.index("calc_time"), head.index("last_funding_rate")
            body = csv_rows[1:]
        else:
            ti, ri = 0, 1
            body = csv_rows
        for r in body:
            if len(r) > max(ti, ri):
                ts.append(r[ti]); rate.append(r[ri])
    if not ts:
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "funding_rate"])
        for t, rt in zip(ts, rate):
            w.writerow([t, rt])
    return len(ts)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01")
    ap.add_argument("--end", default="2026-10")
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--cache", default="perp_cache")
    a = ap.parse_args(argv)
    syms = [s.strip() for s in a.symbols.split(",")] if a.symbols else UNIVERSE
    months = _months(a.start, a.end)
    cache = Path(a.cache)
    print(f"{len(syms)} symbols x {len(months)} months ({a.start}..{a.end})")
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_klines, s, months, cache): ("k", s) for s in syms}
        futs.update({ex.submit(fetch_funding, s, months, cache): ("f", s) for s in syms})
        for fut in as_completed(futs):
            kind, sym = futs[fut]
            try:
                n = fut.result()
                if kind == "k":
                    print(f"  klines {sym}: {n:,} rows", flush=True)
            except Exception as e:                     # noqa: BLE001
                print(f"  {kind} {sym} FAILED: {type(e).__name__} {e}", flush=True)
    ks = sorted(cache.glob("*_1h.csv"))
    fs = sorted((cache / "funding").glob("*.csv")) if (cache / "funding").exists() else []
    print(f"\nDONE: {len(ks)} kline files, {len(fs)} funding files in {cache}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
