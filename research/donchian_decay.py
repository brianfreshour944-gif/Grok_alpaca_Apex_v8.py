"""Do any levers rescue the Donchian edge after it decays (2025-2026)?

The per-year read shows the strategy worked through 2024 and went flat in
2025-2026. This tests the obvious fixes against that specific problem:
  * shorter trend filter (does a faster regime gate adapt?),
  * a MARKET-regime gate (only take breakouts while the whole basket is above
    its own N-day average — trade the whole tape, not just your own symbol),
and reports FULL, per-year, and leave-one-fold-out (OOS) deltas vs base.

Honest framing: the market gate uses the equal-weight basket's own SMA, known
at each close, so there is no look-ahead; but choosing a lever by looking at
these numbers WOULD be in-sample selection. Treat the OOS column as the verdict.

Usage:
  python research/donchian_decay.py --recent 2024-01-01
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.donchian_backtest import load_daily, run_symbols


def market_regime(data: dict[str, pd.DataFrame], window: int) -> dict:
    """Per-symbol bool entry gate: the equal-weight basket close is above its
    own `window`-day SMA. Aligned to each symbol's own index (missing -> False)."""
    closes = pd.DataFrame({s: d["close"] for s, d in data.items()})
    mkt = closes.mean(axis=1)
    above = mkt > mkt.rolling(window).mean()
    return {s: above.reindex(d.index).fillna(False).to_numpy() for s, d in data.items()}


# Each variant is (label, kwargs, regime_window_or_None). The market gate is the
# only one that needs the extra regime array.
LEVERS = [
    ("base",                 {},                                   None),
    ("trend50",              {"trend_filter": 50},                 None),
    ("trend100",             {"trend_filter": 100},                None),
    ("trend200",             {"trend_filter": 200},                None),
    ("mktgate100",           {"trend_filter": 100},                100),
    ("mktgate200",           {"trend_filter": 100},                200),
    ("mktgate200+mid",       {"trend_filter": 100, "exit_mode": "midpoint"}, 200),
]


def metrics(data, levers, fee_bps, regime_window):
    reg = market_regime(data, regime_window) if regime_window else None
    return run_symbols(data, regime=reg, fee_bps=fee_bps, **levers)["__basket__"]


def by_year(data, fee_bps):
    print(f"\nPER-YEAR basket total % (taker {fee_bps}bps)")
    print("  {:<16} {}".format("lever", " ".join(f"{y:>7d}" for y in range(2021, 2027))))
    for label, kw, rw in LEVERS:
        row = []
        for y in range(2021, 2027):
            lo, hi = pd.Timestamp(f"{y}-01-01", tz="UTC"), pd.Timestamp(f"{y+1}-01-01", tz="UTC")
            sub = {s: d[(d.index >= lo) & (d.index < hi)] for s, d in data.items()}
            sub = {s: d for s, d in sub.items() if len(d) > 60}
            if not sub:
                row.append("      -")
                continue
            row.append(f"{metrics(sub, kw, fee_bps, rw)['total_return']*100:>+7.1f}")
        print("  {:<16} {}".format(label, " ".join(row)))


def oos_folds(data, fee_bps):
    """Leave-one-fold-out: report each lever on folds it did NOT pick. No lever
    selects params here, so the deltas vs base are genuinely out of sample."""
    t0 = min(d.index.min() for d in data.values())
    t1 = max(d.index.max() for d in data.values())
    idx = pd.date_range(t0, t1, periods=6)
    print(f"\nOOS per fold (taker {fee_bps}bps): delta% vs base, mean & win count")
    print("  {:<16}  F1..F4 (delta)            mean-dv  win/4")
    for label, kw, rw in LEVERS:
        if label == "base":
            continue
        dv = []
        for i in range(4):
            te = {s: d[(d.index >= idx[i + 1]) & (d.index < idx[i + 2])]
                  for s, d in data.items()}
            te = {s: d for s, d in te.items() if len(d) > 60}
            if not te:
                continue
            e = metrics(te, kw, fee_bps, rw)["total_return"]
            b = metrics(te, {}, fee_bps, None)["total_return"]
            dv.append((e - b) * 100)
        wins = sum(x > 0 for x in dv)
        print("  {:<16}  {}  {:>+7.1f}  {:>4}/{}".format(
            label, " ".join(f"{x:>+6.1f}" for x in dv), np.mean(dv), wins, len(dv)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="okx_daily")
    ap.add_argument("--fee-bps", type=float, default=25.0)
    ap.add_argument("--recent", default=None)
    a = ap.parse_args(argv)
    data = load_daily(a.cache)
    if not data:
        print(f"no data in {a.cache}/ — run research/fetch_okx_daily.py first")
        return 1
    t0 = min(d.index.min() for d in data.values())
    t1 = max(d.index.max() for d in data.values())
    print(f"loaded {len(data)} symbols, {t0.date()} -> {t1.date()}")

    print(f"\nFULL basket (taker {a.fee_bps}bps)")
    print("  {:<16} {:>8} {:>7} {:>7} {:>6}".format("lever", "tot%", "cagr%", "maxDD%", "sharpe"))
    for label, kw, rw in LEVERS:
        m = metrics(data, kw, a.fee_bps, rw)
        print("  {:<16} {:>+8.1f} {:>+7.1f} {:>+7.1f} {:>6.2f}".format(
            label, m["total_return"] * 100, m["cagr"] * 100,
            m["max_drawdown"] * 100, m["sharpe"]))

    if a.recent:
        sub = {s: d[d.index >= pd.Timestamp(a.recent, tz="UTC")] for s, d in data.items()}
        sub = {s: d for s, d in sub.items() if len(d) > 200}
        print(f"\nRECENT >= {a.recent} basket (taker {a.fee_bps}bps)")
        print("  {:<16} {:>8} {:>7} {:>7} {:>6}".format("lever", "tot%", "cagr%", "maxDD%", "sharpe"))
        for label, kw, rw in LEVERS:
            m = metrics(sub, kw, a.fee_bps, rw)
            print("  {:<16} {:>+8.1f} {:>+7.1f} {:>+7.1f} {:>6.2f}".format(
                label, m["total_return"] * 100, m["cagr"] * 100,
                m["max_drawdown"] * 100, m["sharpe"]))

    by_year(data, a.fee_bps)
    oos_folds(data, a.fee_bps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
