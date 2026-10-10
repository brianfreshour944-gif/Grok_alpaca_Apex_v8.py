"""Measure Donchian enhancements on real data — keep only what survives OOS.

Runs a set of named variants (base + one enhancement at a time + combos) on the
same OKX daily panel, then a walk-forward that compares the BASE rule against
the best-enhanced rule on folds that were never used to choose it. The point is
not to find a big in-sample number; it is to see whether any enhancement earns
its keep out of sample.

Usage:
  python research/donchian_improve.py --recent 2024-01-01
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.donchian_backtest import load_daily, run_symbols

# name -> backtest kwargs (base uses the frozen defaults; each variant flips ONE
# lever so its effect is isolated).
VARIANTS = {
    "base": {},
    "trend50": {"trend_filter": 50},
    "trend100": {"trend_filter": 100},
    "trend200": {"trend_filter": 200},
    "midpoint": {"exit_mode": "midpoint"},
    "trail3": {"trail_atr": 3.0},
    "trail2": {"trail_atr": 2.0},
    "pyr2": {"pyramid_units": 2},
    "pyr4": {"pyramid_units": 4},
    "trend100+mid": {"trend_filter": 100, "exit_mode": "midpoint"},
    "trend100+trail3": {"trend_filter": 100, "trail_atr": 3.0},
    "trend100+pyr2": {"trend_filter": 100, "pyramid_units": 2},
    "mid+pyr2": {"exit_mode": "midpoint", "pyramid_units": 2},
    "trend100+mid+pyr2": {"trend_filter": 100, "exit_mode": "midpoint",
                          "pyramid_units": 2},
    "all": {"trend_filter": 100, "exit_mode": "midpoint", "pyramid_units": 2,
            "trail_atr": 3.0},
}


def evaluate(data, fee_bps, **kw):
    per = run_symbols(data, fee_bps=fee_bps, **kw)
    return per["__basket__"]


def table(label, data, fee_bps):
    print(f"\n{label}  (basket, taker {fee_bps}bps)")
    print("  {:<16} {:>8} {:>7} {:>7} {:>6} {:>5} {:>6}".format(
        "variant", "tot%", "cagr%", "sharpe", "maxDD", "win", "PF"))
    rows = {}
    for name, kw in VARIANTS.items():
        m = evaluate(data, fee_bps, **kw)
        rows[name] = m
        print("  {:<16} {:>+8.1f} {:>+7.1f} {:>7.2f} {:>+6.1f} {:>5.2f} {:>6.2f}".format(
            name, m["total_return"] * 100, m["cagr"] * 100, m["sharpe"],
            m["max_drawdown"] * 100, m["win_rate"], m["profit_factor"]))
    return rows


def folds(data, t0, t1):
    idx = pd.date_range(t0, t1, periods=6)
    for i in range(4):
        hi = idx[i + 1]
        te = {s: d[(d.index >= hi) & (d.index < idx[i + 2])] for s, d in data.items()}
        te = {s: d for s, d in te.items() if len(d) > 60}
        if te:
            yield i + 1, te


def walk_forward_fixed(data, fee_bps, t0, t1):
    """Compare each FIXED enhancement against base on every test fold. This is
    the honest read: no per-fold selection, so the deltas are genuinely OOS."""
    fs = list(folds(data, t0, t1))
    print(f"\nWALK-FORWARD: fixed variants vs base, per test fold, taker {fee_bps}bps")
    print("  {:<16} {}   mean-dv  win/4  meanSharpe".format(
        "variant", " ".join(f"f{i+1:>6}" for i, _ in fs)))
    for name, kw in VARIANTS.items():
        if name == "base":
            continue
        sharpe, dv = [], []
        for _, d in fs:
            e = evaluate(d, fee_bps, **kw)
            b = evaluate(d, fee_bps)
            dv.append((e["total_return"] - b["total_return"]) * 100)
            sharpe.append(e["sharpe"])
        wins = sum(x > 0 for x in dv)
        print("  {:<16} {}   {:>+7.1f}  {:>4}/4   {:>6.2f}".format(
            name, " ".join(f"{x:>+6.1f}" for x in dv), np.mean(dv), wins, np.mean(sharpe)))


def risk_normalized(data, fee_bps):
    """Pyramiding adds exposure, so raw return overstates it. Compare return per
    unit of drawdown (a crude Calmar) and Sharpe, full + recent."""
    print(f"\nRISK-NORMALISED (taker {fee_bps}bps): higher is better on every column")
    print("  {:<10} {:>22} {:>22}".format("variant", "FULL ret/DD, Sharpe", "RECENT ret/DD, Sharpe"))
    for name, kw in VARIANTS.items():
        line = []
        for sub in (data,):
            m = evaluate(sub, fee_bps, **kw)
            line.append(f"{m['total_return']/abs(m['max_drawdown']):>6.2f}, {m['sharpe']:>6.2f}")
        print(f"  {name:<10} {line[0]:>22}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="okx_daily")
    ap.add_argument("--fee-bps", type=float, default=25.0)
    ap.add_argument("--recent", default=None)
    ap.add_argument("--walk-forward", action="store_true")
    a = ap.parse_args(argv)
    data = load_daily(a.cache)
    if not data:
        print(f"no data in {a.cache}/ — run research/fetch_okx_daily.py first")
        return 1
    t0 = min(d.index.min() for d in data.values())
    t1 = max(d.index.max() for d in data.values())
    print(f"loaded {len(data)} symbols, {t0.date()} -> {t1.date()}")
    table("FULL", data, a.fee_bps)
    if a.recent:
        sub = {s: d[d.index >= pd.Timestamp(a.recent, tz="UTC")] for s, d in data.items()}
        sub = {s: d for s, d in sub.items() if len(d) > 200}
        table(f"RECENT >= {a.recent}", sub, a.fee_bps)
    # zero-fee gross to separate signal from cost
    table("FULL, zero fees (gross)", data, 0.0)
    if a.walk_forward:
        walk_forward_fixed(data, a.fee_bps, t0, t1)
    risk_normalized(data, a.fee_bps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
