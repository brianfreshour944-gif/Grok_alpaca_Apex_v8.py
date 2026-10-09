"""Fetch Binance SPOT 1h klines (the hedge leg of a cash-and-carry trade).

Same Binance Vision source as the perp fetcher, but the spot path and note that
spot kline open_time is in MICROSECONDS (futures use milliseconds).

Output: spot_cache/<SYM>_1h.csv  (open_time,o,h,l,c,volume,...)

Usage: python research/fetch_spot_binance.py --start 2021-01 --end 2026-10
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fetch_perp_binance import UNIVERSE, _get_zip_csv, _months

S = "https://data.binance.vision/data/spot"


def fetch_spot(sym: str, months: list[str], cache: Path) -> int:
    out = cache / f"{sym}_1h.csv"
    if out.exists():
        return 0
    rows: list[list[str]] = []
    header: list[str] | None = None
    for mo in months:
        csv_rows = _get_zip_csv(f"{S}/monthly/klines/{sym}/1h/{sym}-1h-{mo}.zip")
        if not csv_rows:
            continue
        if csv_rows[0] and not csv_rows[0][0].lstrip("-").isdigit():
            header, csv_rows = csv_rows[0], csv_rows[1:]
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01")
    ap.add_argument("--end", default="2026-10")
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--cache", default="spot_cache")
    a = ap.parse_args(argv)
    syms = [s.strip() for s in a.symbols.split(",")] if a.symbols else UNIVERSE
    months = _months(a.start, a.end)
    cache = Path(a.cache)
    print(f"{len(syms)} symbols x {len(months)} months -> {cache}/")
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_spot, s, months, cache): s for s in syms}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                n = fut.result()
                print(f"  spot {sym}: {n:,} rows", flush=True)
            except Exception as e:                     # noqa: BLE001
                print(f"  spot {sym} FAILED: {type(e).__name__} {e}", flush=True)
    print(f"\nDONE: {len(list(cache.glob('*_1h.csv')))} spot files in {cache}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
