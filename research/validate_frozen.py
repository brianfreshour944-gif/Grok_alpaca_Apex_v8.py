"""Frozen-rule validation: backward (2022-2024) and forward (unseen) tests.

The rule is FROZEN (see funding_carry.py): K=3, 168h funding lookback, 24h
rebalance, 15-symbol universe. Nothing here tunes it. This module only *tests*
it on data the tuning never used:

  * BACKWARD: 2022-01 .. 2024-12. The strategy was selected on 2025-01..2026-09,
    so all of 2022-2024 is genuinely pre-tuning. If the edge is real it should
    show up here too; if it only exists in 2025-26 it is a regime, not an edge.
  * FORWARD: the days after the tuning window (2026-10-01 onward). Binance
    Vision monthly archives stop at 2026-09 and daily archives at 2026-10-08,
    so only a handful of genuinely-unseen days exist right now; the daily ledger
    (`--ledger`) is designed to keep appending as new days become available.

Funding for the forward tail is not in the Vision archives (no daily funding
files), so it is pulled live from OKX's public funding-rate-history endpoint
(Binance's own REST API is geo-blocked here). OKX and Binance both settle 8h
funding, so the carry ranking is comparable.

Run:
  python research/validate_frozen.py --backward
  python research/validate_frozen.py --forward
  python research/validate_frozen.py --forward --ledger bt_cache/research/forward_ledger.csv
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import funding_carry as fc

CACHE = os.path.join(ROOT, "bt_cache", "research", "validate")
UA = {"User-Agent": "Mozilla/5.0 (research)"}
KL = ["open_time", "open", "high", "low", "close", "volume", "close_time",
      "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]

# OKX swap instrument ids for the frozen universe.
OKX_ID = {
    "BTC": "BTC-USD-SWAP", "ETH": "ETH-USD-SWAP", "SOL": "SOL-USD-SWAP",
    "BNB": "BNB-USD-SWAP", "XRP": "XRP-USD-SWAP", "ADA": "ADA-USD-SWAP",
    "DOGE": "DOGE-USD-SWAP", "AVAX": "AVAX-USD-SWAP", "LINK": "LINK-USD-SWAP",
    "DOT": "DOT-USD-SWAP", "LTC": "LTC-USD-SWAP", "BCH": "BCH-USD-SWAP",
    "TRX": "TRX-USD-SWAP", "ATOM": "ATOM-USD-SWAP", "UNI": "UNI-USD-SWAP",
}


# ── fetch ────────────────────────────────────────────────────────────────────
def _get(url, timeout=40):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=timeout).read()


def _read_zip_csv(raw: bytes | None):
    if raw is None:
        return None
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
        d = pd.read_csv(zf.open(zf.namelist()[0]), header=None)
        if len(d) and isinstance(d.iloc[0, 0], str):
            d = d.iloc[1:].reset_index(drop=True)
        return d
    except (zipfile.BadZipFile, OSError, ValueError, IndexError):
        return None


def _dt(col) -> pd.DatetimeIndex:
    v = pd.to_numeric(col)
    unit = "us" if v.dropna().abs().median() > 1e14 else "ms"
    return pd.to_datetime(v.astype("int64"), unit=unit, utc=True).dt.tz_localize(None)


def _month_sig(months: list[str]) -> str:
    return f"{months[0]}_{months[-1]}_{len(months)}"


def fetch_symbol_monthly(sym: str, months: list[str]) -> tuple[pd.Series, pd.Series]:
    """Perp close (hourly) + funding rate (8h) for one symbol, months list.

    Cache is keyed by the requested month range so a later, wider request does
    not silently reuse a narrower cached file.
    """
    sig = _month_sig(months)
    cpath = os.path.join(CACHE, f"close_{sym}_{sig}.csv")
    fpath = os.path.join(CACHE, f"funding_{sym}_{sig}.csv")
    closes, funds = [], []
    if os.path.exists(cpath) and os.path.exists(fpath):
        C = pd.read_csv(cpath, index_col=0, parse_dates=True).iloc[:, 0]
        F = pd.read_csv(fpath, index_col=0, parse_dates=True).iloc[:, 0]
        return C, F
    os.makedirs(CACHE, exist_ok=True)
    for mon in months:
        k = _read_zip_csv(_try(_klines_url(sym, mon)))
        if k is not None and len(k):
            k = k.iloc[:, :len(KL)]
            k.columns = KL
            ts = _dt(k["open_time"])
            closes.append(pd.Series(k["close"].astype(float).to_numpy(), index=ts))
        f = _read_zip_csv(_try(_funding_url(sym, mon)))
        if f is not None and len(f):
            f.columns = [str(c) for c in f.columns][:3]
            fts = _dt(f.iloc[:, 0])
            funds.append(pd.Series(f.iloc[:, 2].astype(float).to_numpy(), index=fts))
    C = pd.concat(closes).sort_index() if closes else pd.Series(dtype=float)
    F = pd.concat(funds).sort_index() if funds else pd.Series(dtype=float)
    C = C[~C.index.duplicated(keep="last")]
    F = F[~F.index.duplicated(keep="last")]
    C.to_csv(cpath)
    F.to_csv(fpath)
    return C, F


def _try(url):
    try:
        return _get(url)
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _klines_url(sym, mon):
    return (f"https://data.binance.vision/data/futures/um/monthly/klines/"
            f"{sym}USDT/1h/{sym}USDT-1h-{mon}.zip")


def _funding_url(sym, mon):
    return (f"https://data.binance.vision/data/futures/um/monthly/fundingRate/"
            f"{sym}USDT/{sym}USDT-fundingRate-{mon}.zip")


def load_panel(symbols, months, workers=16) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = {}
    with ThreadPoolExecutor(workers) as ex:
        futs = {s: ex.submit(fetch_symbol_monthly, s, months) for s in symbols}
        for s, fut in futs.items():
            out[s] = fut.result()
    close = pd.DataFrame({s: out[s][0] for s in symbols}).sort_index()
    fund = pd.DataFrame({s: out[s][1] for s in symbols}).sort_index()
    fund = fund.reindex(close.index, method="ffill")
    return close, fund


# ── frozen strategy ──────────────────────────────────────────────────────────
def frozen_book(close: pd.DataFrame, fund: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Run the FROZEN rule on a close/funding panel; per-period P&L rows."""
    import research.market_neutral_funding as M
    if start is not None:
        close = close.loc[start:]
        fund = fund.loc[start:]
    if end is not None:
        close = close.loc[:end]
        fund = fund.loc[:end]
    score = fc.funding_carry_score(fund, lookback=fc.LOOKBACK_HOURS)
    fwd = M.forward_returns(close, fc.HOLD_HOURS)
    return M.long_short_book(score, fwd, fc.HOLD_HOURS, fc.TOP_K, funding=fund)


def _daily_klines_url(sym, day):
    return (f"https://data.binance.vision/data/futures/um/daily/klines/"
            f"{sym}USDT/1h/{sym}USDT-1h-{day}.zip")


def fetch_daily_close(sym: str, days: list[str]) -> pd.Series:
    """Daily perp klines for days not yet in a monthly archive (forward tail)."""
    sig = f"d_{days[0]}_{days[-1]}_{len(days)}"
    cpath = os.path.join(CACHE, f"close_{sym}_{sig}.csv")
    if os.path.exists(cpath):
        return pd.read_csv(cpath, index_col=0, parse_dates=True).iloc[:, 0]
    os.makedirs(CACHE, exist_ok=True)
    parts = []
    for day in days:
        k = _read_zip_csv(_try(_daily_klines_url(sym, day)))
        if k is not None and len(k):
            k.columns = KL
            ts = _dt(k["open_time"])
            parts.append(pd.Series(k["close"].astype(float).to_numpy(), index=ts))
    C = pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)
    C = C[~C.index.duplicated(keep="last")]
    C.name = sym
    C.to_csv(cpath)
    return C


def fetch_okx_funding(sym: str, days: list[str]) -> pd.Series:
    """Live OKX funding history (Vision has no daily funding archives; Binance
    REST is geo-blocked here). OKX and Binance both settle 8h funding."""
    inst = OKX_ID.get(sym)
    if inst is None:
        return pd.Series(dtype=float)
    sig = f"okx_{days[0]}_{days[-1]}"
    cpath = os.path.join(CACHE, f"funding_{sym}_{sig}.csv")
    if os.path.exists(cpath):
        return pd.read_csv(cpath, index_col=0, parse_dates=True).iloc[:, 0]
    rows, after = [], None
    for _ in range(20):
        url = (f"https://www.okx.com/api/v5/public/funding-rate-history?"
               f"instId={inst}&limit=100")
        if after:
            url += f"&after={after}"
        try:
            d = json.loads(_get(url))
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
            break
        data = d.get("data", [])
        if not data:
            break
        rows.extend(data)
        after = data[-1]["fundingTime"]
    if not rows:
        return pd.Series(dtype=float)
    f = pd.DataFrame(rows)
    ts = pd.to_datetime(f["fundingTime"].astype("int64"), unit="ms", utc=True).dt.tz_localize(None)
    s = pd.Series(f["fundingRate"].astype(float).to_numpy(), index=ts).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s.name = sym
    os.makedirs(CACHE, exist_ok=True)
    s.to_csv(cpath)
    return s


def forward_periods(monthly_months: list[str], symbols) -> list[str]:
    """Days after the last complete monthly archive that have a Vision daily file."""
    last = monthly_months[-1]
    y, m = int(last[:4]), int(last[5:7])
    start = pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthBegin(1)
    days = pd.date_range(start, start + pd.Timedelta(days=45), freq="D")
    out = []
    probe = symbols[0]
    for d in days:
        if _try(_daily_klines_url(probe, d.strftime("%Y-%m-%d"))) is not None:
            out.append(d.strftime("%Y-%m-%d"))
    return out


def load_forward(symbols, months, workers=16):
    """Backward monthly panel + forward daily tail (klines + OKX funding).

    The daily pieces are assembled as one frame and concat'd ONCE: concatenating
    per-symbol in a loop stack-duplicates the shared timestamp index instead of
    aligning symbols. OKX funding is clipped to strictly after the monthly window
    so it cannot overlap (and corrupt) the Binance monthly funding series.
    """
    close, fund = load_panel(symbols, months, workers=workers)
    monthly_end = close.index.max()
    days = forward_periods(months, symbols)
    if not days:
        return close, fund, []
    with ThreadPoolExecutor(workers) as ex:
        dc = {s: ex.submit(fetch_daily_close, s, days) for s in symbols}
        df = {s: ex.submit(fetch_okx_funding, s, days) for s in symbols}
        daily_c = {s: c for s in symbols if len(c := dc[s].result())}
        daily_f = {}
        for s in symbols:
            f = df[s].result()
            if len(f):
                f = f[f.index > monthly_end]          # forward tail only
                if len(f):
                    daily_f[s] = f
    if daily_c:
        close = pd.concat([close, pd.DataFrame(daily_c).sort_index()]).sort_index()
    if daily_f:
        fund = pd.concat([fund, pd.DataFrame(daily_f).sort_index()]).sort_index()
    close = close[~close.index.duplicated(keep="last")]
    fund = fund[~fund.index.duplicated(keep="last")]
    # hourly view for the frozen score
    fund = fund.reindex(close.index, method="ffill")
    return close, fund, days


def daily_ledger(close: pd.DataFrame, fund: pd.DataFrame, days: list[str]) -> pd.DataFrame:
    """One row per rebalance day for the ledger, restricted to `days` onward.

    The book is scored on the FULL panel (so the 168h funding lookback is intact
    for the first unseen day), then the ledger keeps only rebalances on/after
    `days[0]`.
    """
    if not days:
        return pd.DataFrame()
    start = pd.Timestamp(days[0])
    book = frozen_book(close, fund)
    if book.empty:
        return book
    book = book[book.index >= start]
    if book.empty:
        return book
    legs = (book["n_long"] + book["n_short"]).to_numpy(float)
    fee = fc.TAKER_BPS
    return pd.DataFrame({
        "date": book.index,
        "price_bps": book["price"] * 1e4,
        "funding_bps": book["funding_cash"] * 1e4,
        "gross_bps": book["gross"] * 1e4,
        "turnover": book["turnover"],
        "net_bps_taker": (book["gross"] - book["turnover"] * legs * fee / 1e4) * 1e4,
        "n_long": book["n_long"], "n_short": book["n_short"],
    })


def report(name: str, book: pd.DataFrame, by_year=True, by_quarter=False):
    import research.market_neutral_funding as M
    cov = M.Cov()
    s = M.summarize_periods(book, cov, fc.HOLD_HOURS)
    per_year = 8760 / fc.HOLD_HOURS
    print(f"\n  === {name} ===")
    print(f"  n={s['n']}  hist {book.index.min()} -> {book.index.max()}")
    print(f"  gross {s['gross_bps']:+.1f} bps/period | net taker(5) {s['net_bps_taker']:+.1f} "
          f"(t={s['t_taker']:+.2f}) | net maker(2) {s['net_bps_maker']:+.1f} | net alpaca(25) {s['net_bps_tier1']:+.1f}")
    print(f"  price {s.get('price_bps', float('nan')):+.1f} | funding {s.get('funding_bps', float('nan')):+.1f} "
          f"| turnover {s['turnover']:.2f}")
    print(f"  simple annualised (taker): {((s['net_bps_taker'])/1e4)*per_year*100:+.1f}%  "
          f"(maker): {((s['net_bps_maker'])/1e4)*per_year*100:+.1f}%")
    if by_year:
        print(f"  {'year':>6} {'n':>4} {'gross':>8} {'netT':>8} {'tT':>6} {'netM':>8}")
        for y, g in book.groupby(book.index.year):
            sy = M.summarize_periods(g, cov, fc.HOLD_HOURS)
            print(f"  {y:>6} {sy['n']:>4} {sy['gross_bps']:>+8.1f} {sy['net_bps_taker']:>+8.1f} "
                  f"{sy['t_taker']:>+6.2f} {sy['net_bps_maker']:>+8.1f}")
    if by_quarter:
        print(f"  {'quarter':>10} {'n':>4} {'gross':>8} {'netT':>8}")
        for q, g in book.groupby(book.index.to_period("Q")):
            sq = M.summarize_periods(g, cov, fc.HOLD_HOURS)
            print(f"  {q!s:>10} {sq['n']:>4} {sq['gross_bps']:>+8.1f} {sq['net_bps_taker']:>+8.1f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backward", action="store_true", help="2022-2024")
    ap.add_argument("--forward", action="store_true", help="2025-2026 + unseen tail")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--quarter", action="store_true")
    ap.add_argument("--ledger", default=None, help="write per-day forward ledger CSV")
    a = ap.parse_args(argv)

    if a.backward:
        months = [f"{y}-{m:02d}" for y in (2022, 2023, 2024) for m in range(1, 13)]
        print(f"Backward test: {len(fc.DEFAULT_UNIVERSE)} symbols, 2022-01..2024-12 "
              f"(pre-tuning window)")
        close, fund = load_panel(fc.DEFAULT_UNIVERSE, months)
        print(f"  panel {close.shape[1]} syms x {close.shape[0]:,} h "
              f"({close.index.min()} -> {close.index.max()})")
        report("BACKWARD 2022-2024 (never tuned on)", frozen_book(close, fund),
               by_year=True, by_quarter=a.quarter)
        full = [f"{y}-{m:02d}" for y in (2022, 2023, 2024, 2025, 2026) for m in range(1, 13)]
        full = [m for m in full if m <= "2026-09"]
        closef, fundf = load_panel(fc.DEFAULT_UNIVERSE, full)
        report("FULL 2022-01..2026-09 (incl. the 2025-26 tuning window)",
               frozen_book(closef, fundf), by_year=True)

    if a.forward:
        months = [f"{y}-{m:02d}" for y in (2025, 2026) for m in range(1, 13)]
        months = [m for m in months if m <= "2026-09"]
        print(f"Forward test: {len(fc.DEFAULT_UNIVERSE)} symbols, 2025-01..2026-09 "
              f"(tuning) + unseen tail")
        close, fund, days = load_forward(fc.DEFAULT_UNIVERSE, months)
        print(f"  panel {close.shape[1]} syms x {close.shape[0]:,} h "
              f"({close.index.min()} -> {close.index.max()})")
        if days:
            print(f"  unseen daily tail: {days[0]} .. {days[-1]}  ({len(days)} days)")
        else:
            print("  unseen daily tail: NONE available yet (monthly archive already current)")
        # full forward window incl. tuning, for context
        report("FORWARD 2025-01..end (tuning + unseen)", frozen_book(close, fund),
               by_year=True, by_quarter=True)
        if days:
            led = daily_ledger(close, fund, days)
            if not led.empty:
                print("\n  === UNSEEN DAILY LEDGER (frozen rule, taker 5bps) ===")
                print(led.to_string(index=False))
                tot = led["net_bps_taker"].sum()
                print(f"  {len(led)} unseen rebalances | cumulative net {tot:+.1f} bps "
                      f"| mean {led['net_bps_taker'].mean():+.1f} bps/period")
                if a.ledger:
                    os.makedirs(os.path.dirname(a.ledger), exist_ok=True)
                    led.to_csv(a.ledger, index=False)
                    print(f"  ledger written -> {a.ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
