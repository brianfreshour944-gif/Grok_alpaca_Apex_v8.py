"""Mark the Donchian paper ledger to market — turn intentions into a record.

`donchian_bot.py --paper --ledger X` appends the *target* book each day but does
not track whether that book made money. This reads the ledger, holds each day's
book until the next row, charges turnover costs, and reports forward
performance against equal-weight buy & hold of the same names.

This is the honest forward test for the ONE strategy Alpaca can actually run
(long-only spot). It is price-data agnostic: point --cache at whatever bars the
cron fetched (OKX cache here; Alpaca daily bars where the network allows).

Ledger row format (from donchian_bot.append_ledger):
  date,equity,mode,book,gross_weight,n_legs
  book = "BTCUSDT:1:+0.050|ETHUSDT:-1:-0.030|..."  (symbol:signal:weight)

Usage:
  python research/donchian_forward.py --ledger donchian_ledger.csv --cache okx_daily
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.xsec_momentum import load_close


def _normalize_dates(close: pd.DataFrame) -> pd.DataFrame:
    """Floor bar timestamps to a calendar date so they line up with ledger dates.

    Venue bars are stamped intraday (OKX dailies are 16:00 UTC); the ledger is
    date-only. Without this the mark compares mismatched days and can run
    backwards."""
    out = close.copy()
    out.index = out.index.normalize()
    return out[~out.index.duplicated(keep="last")].sort_index()


def parse_book(field: str) -> dict[str, float]:
    """'BTCUSDT:1:+0.05|ETHUSDT:0:0.000' -> {sym: weight} (zero weights kept)."""
    out: dict[str, float] = {}
    if not field:
        return out
    for part in field.split("|"):
        bits = part.split(":")
        if len(bits) == 3:
            try:
                out[bits[0]] = float(bits[2])
            except ValueError:
                continue
    return out


def read_ledger(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open() as f:
        return list(csv.DictReader(f))


def mark_ledger(rows: list[dict], close: pd.DataFrame,
                fee_bps: float = 25.0) -> dict:
    """Mark each ledger book forward to the next ledger date.

    Returns the book's forward equity curve, daily returns, and the same series
    for an equal-weight buy & hold of every symbol that appears in the ledger."""
    if len(rows) < 2:
        return {"n_days": 0}
    dates = pd.to_datetime([r["date"] for r in rows], utc=True).normalize()
    books = [parse_book(r.get("book", "")) for r in rows]
    syms = sorted({s for b in books for s in b})
    px = _normalize_dates(close).reindex(columns=syms)

    eq = [1.0]
    daily = []
    bench = [1.0]
    for i in range(len(rows) - 1):
        d0, d1 = dates[i], dates[i + 1]
        w = pd.Series(books[i]).reindex(syms).fillna(0.0)
        p0 = px.loc[d0] if d0 in px.index else px.iloc[(px.index < d0).sum() - 1]
        p1 = px.loc[d1] if d1 in px.index else px.iloc[(px.index < d1).sum() - 1]
        ret = (p1 / p0 - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        port = float((w * ret).sum())
        # drifted weights at d1, for a like-for-like turnover measure
        w_drift = (w * (1.0 + ret)) / (1.0 + port) if (1.0 + port) != 0 else w
        w_next = pd.Series(books[i + 1]).reindex(syms).fillna(0.0)
        turnover = float((w_next - w_drift).abs().sum())
        cost = turnover * fee_bps / 1e4
        eq.append(eq[-1] * (1.0 + port - cost))
        daily.append(port - cost)
        # benchmark: equal-weight all ledger symbols, held the whole time
        bench.append(bench[-1] * (1.0 + float(ret.mean())))

    d = np.array(daily)
    eq = np.array(eq)
    bench = np.array(bench)
    return {
        "n_days": len(rows) - 1,
        "start": str(dates[0].date()),
        "end": str(dates[-1].date()),
        "equity": eq,
        "daily": d,
        "bench_equity": bench,
        "book_total_return": float(eq[-1] - 1.0),
        "bench_total_return": float(bench[-1] - 1.0),
        "book_sharpe": _sharpe(d),
        "book_maxdd": _maxdd(eq),
        "bench_sharpe": _sharpe(np.diff(bench) / bench[:-1]),
        "bench_maxdd": _maxdd(bench),
        "avg_daily_return": float(d.mean()) if d.size else 0.0,
        "win_days": float((d > 0).mean()) if d.size else 0.0,
    }


def _sharpe(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(365))


def _maxdd(eq: np.ndarray) -> float:
    peak = np.maximum.accumulate(eq)
    return float(np.min(eq / peak - 1.0))


def backfill_ledger(ledger: str, cache: str, days: int, source: str = "okx",
                    strategy: dict | None = None, mode: str = "enhanced",
                    step: int = 1) -> int:
    """Rebuild a paper ledger the way the cron would have accrued it: walk the
    last `days` calendar days, and each day from `step`-many days of history
    compute the target book via donchian_bot and append one row.

    This exists so the forward test is reproducible in the sandbox (the real
    cron simply appends today's row each day). No orders, no network beyond the
    cache."""
    import donchian_bot as bot

    close = load_close(cache)
    if close.empty:
        print(f"no price data in {cache}/")
        return 1
    # symbols need high/low/close; the cache stores close, so approximate the
    # daily range from close (the live path uses real OHLC from the venue).
    all_syms = list(close.columns)
    end = close.index.max()
    start = end - pd.Timedelta(days=days)
    dates = close.index[(close.index >= start) & (close.index <= end)][::step]
    n = 0
    for d in dates:
        hist = close.loc[:d]
        if len(hist) < bot.db.DEFAULT_ENTRY + 2:
            continue
        data = {}
        for s in all_syms:
            c = hist[s].dropna()
            if len(c) < bot.db.DEFAULT_ENTRY + 2:
                continue
            data[s] = pd.DataFrame({"high": c, "low": c, "close": c})
        if not data:
            continue
        targets = bot.target_book(data, 10_000.0, strategy=strategy)
        if bot.append_ledger(ledger, str(d.date()), 10_000.0, targets, mode=mode):
            n += 1
    print(f"backfilled {n} ledger rows -> {ledger}")
    return 0


def forward_report(ledger: str, cache: str, fee_bps: float) -> int:
    rows = read_ledger(ledger)
    if not rows:
        print(f"no ledger rows in {ledger} — run donchian_bot.py --paper first")
        return 1
    close = load_close(cache)
    if close.empty:
        print(f"no price data in {cache}/ — cannot mark the ledger")
        return 1
    m = mark_ledger(rows, close, fee_bps)
    if m["n_days"] == 0:
        print(f"ledger has {len(rows)} row(s); need >=2 dates to mark forward")
        return 0
    print(f"FORWARD TEST  {ledger}  {m['start']} -> {m['end']}  "
          f"({m['n_days']} days, fee {fee_bps}bps)")
    print(f"  {'':<16}{'total%':>9} {'sharpe':>7} {'maxDD%':>8}")
    print(f"  {'donchian book':<16}{m['book_total_return']*100:>+9.1f} "
          f"{m['book_sharpe']:>7.2f} {m['book_maxdd']*100:>+8.1f}")
    print(f"  {'buy & hold':<16}{m['bench_total_return']*100:>+9.1f} "
          f"{m['bench_sharpe']:>7.2f} {m['bench_maxdd']*100:>+8.1f}")
    print(f"  win days {m['win_days']*100:.0f}%  "
          f"mean daily {m['avg_daily_return']*100:+.3f}%")
    if m["n_days"] < 20:
        print("  (too few days to conclude anything yet — keep the cron running)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="donchian_ledger.csv")
    ap.add_argument("--cache", default="okx_daily")
    ap.add_argument("--fee-bps", type=float, default=25.0)
    ap.add_argument("--backfill-days", type=int, default=0,
                    help="rebuild a ledger over the last N days before reporting")
    ap.add_argument("--base", action="store_true", help="use the textbook rule")
    a = ap.parse_args(argv)
    if a.backfill_days > 0:
        import donchian_bot as bot
        strat = None if a.base else bot.ENHANCED
        mode = "base" if a.base else "enhanced"
        backfill_ledger(a.ledger, a.cache, a.backfill_days, strategy=strat, mode=mode)
    return forward_report(a.ledger, a.cache, a.fee_bps)


if __name__ == "__main__":
    raise SystemExit(main())
