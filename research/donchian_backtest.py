"""Backtest the Donchian breakout on real daily crypto bars (OKX).

Honest study, not a sales pitch:
  * full-period metrics per symbol + an equal-weight basket,
  * buy & hold for the same symbols as the baseline the breakout must beat,
  * a parameter sweep ONLY to show how fragile a tuned peak is,
  * a walk-forward (pick params on train, report the next fold) to test whether
    the sweep generalises or is curve-fitting.

Fees default to Alpaca's crypto taker (25 bps/side) because that is the venue
the incumbent bot trades; a zero-fee run shows the gross signal.

Data: `python research/fetch_okx_daily.py` -> okx_daily/<SYM>_1D.csv

Usage:
  python research/donchian_backtest.py --sweep --walk-forward
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import donchian_breakout as db


def load_daily(cache: str) -> dict[str, pd.DataFrame]:
    out = {}
    for fp in sorted(glob.glob(os.path.join(cache, "*_1D.csv"))):
        sym = os.path.basename(fp).replace("_1D.csv", "")
        df = pd.read_csv(fp)
        ts = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
        df = df.assign(ts=ts).set_index("ts")[["high", "low", "close"]].astype(float)
        df = df[~df.index.duplicated()].sort_index()
        if len(df) > 200:
            out[sym] = df
    return out


def run_symbols(data: dict[str, pd.DataFrame], **kw) -> dict:
    """Run the backtest per symbol; return per-symbol summaries + a basket."""
    per, curves, trades = {}, {}, {}
    for sym, df in data.items():
        r = db.backtest_donchian(df, **kw)
        per[sym] = {k: r[k] for k in
                    ("total_return", "cagr", "sharpe", "max_drawdown",
                     "n_trades", "win_rate", "profit_factor", "exposure",
                     "buy_hold_return", "final_equity")}
        curves[sym] = r["equity"] / r["initial_equity"]
        trades[sym] = r["trades"]
    # equal-weight basket of the per-symbol normalised equity curves
    if curves:
        norm = pd.DataFrame(curves).sort_index().ffill()
        basket = norm.mean(axis=1)
        all_trades = pd.concat([t for t in trades.values() if not t.empty], ignore_index=True) \
            if any(not t.empty for t in trades.values()) else pd.DataFrame(columns=["pnl"])
        gw = float(all_trades[all_trades["pnl"] > 0]["pnl"].sum()) if not all_trades.empty else 0.0
        gl = float(-all_trades[all_trades["pnl"] <= 0]["pnl"].sum()) if not all_trades.empty else 0.0
        per["__basket__"] = {
            "total_return": float(basket.iloc[-1] - 1.0),
            "cagr": float(basket.iloc[-1] ** (365 / max(len(basket), 1)) - 1.0),
            "sharpe": db._sharpe(basket.pct_change().fillna(0).to_numpy(), 365),
            "max_drawdown": db._max_drawdown(basket.to_numpy()),
            "n_trades": len(all_trades),
            "win_rate": float((all_trades["pnl"] > 0).mean()) if not all_trades.empty else 0.0,
            "profit_factor": float(gw / gl) if gl > 0 else (float("inf") if gw > 0 else 0.0),
            "exposure": float(np.mean([v["exposure"] for k, v in per.items() if k != "__basket__"])),
            "buy_hold_return": float(np.mean([v["buy_hold_return"] for k, v in per.items() if k != "__basket__"])),
            "final_equity": float(basket.iloc[-1]),
        }
        per["__basket__"]["curve"] = basket
    return per


def _fmt_row(name, m):
    return (f"  {name:<10} {m['total_return']*100:>+8.1f} {m['cagr']*100:>+7.1f} "
            f"{m['sharpe']:>6.2f} {m['max_drawdown']*100:>+7.1f} {m['n_trades']:>6d} "
            f"{m['win_rate']:>5.2f} {m['profit_factor']:>6.2f} {m['exposure']:>5.2f} "
            f"{m['buy_hold_return']*100:>+8.1f}")


HEADER = ("  {:<10} {:>8} {:>7} {:>6} {:>7} {:>6} {:>5} {:>6} {:>5} {:>8}"
          .format("symbol", "tot%", "cagr%", "sharpe", "maxDD%", "n", "win", "PF", "expo", "B&H%"))


def report(label, data, **kw):
    print(f"\n{label}  (entry={kw.get('entry')} exit={kw.get('exit')} "
          f"atr={kw.get('atr_window')} stop={kw.get('stop_atr')} "
          f"fee={kw.get('fee_bps')}bps short={kw.get('allow_short')})")
    print(HEADER)
    per = run_symbols(data, **kw)
    for sym in sorted(k for k in per if k != "__basket__"):
        print(_fmt_row(sym, per[sym]))
    print(_fmt_row("BASKET", per["__basket__"]))
    return per


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="okx_daily")
    ap.add_argument("--fee-bps", type=float, default=25.0)
    ap.add_argument("--short", action="store_true", help="allow short breakouts")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--walk-forward", action="store_true")
    ap.add_argument("--recent", default=None)
    a = ap.parse_args(argv)

    data = load_daily(a.cache)
    if not data:
        print(f"no data in {a.cache}/ — run research/fetch_okx_daily.py first")
        return 1
    t0 = min(d.index.min() for d in data.values())
    t1 = max(d.index.max() for d in data.values())
    print(f"loaded {len(data)} symbols, {t0.date()} -> {t1.date()}")
    common = {"entry": db.DEFAULT_ENTRY, "exit": db.DEFAULT_EXIT,
              "atr_window": db.DEFAULT_ATR, "stop_atr": db.DEFAULT_STOP_ATR,
              "allow_short": a.short}

    report("FULL, taker fees", data, fee_bps=a.fee_bps, **common)
    report("FULL, zero fees (gross signal)", data, fee_bps=0.0, **common)
    if a.recent:
        sub = {s: d[d.index >= pd.Timestamp(a.recent, tz="UTC")] for s, d in data.items()}
        sub = {s: d for s, d in sub.items() if len(d) > 200}
        report(f"RECENT >= {a.recent}", sub, fee_bps=a.fee_bps, **common)

    if a.sweep:
        print(f"\nSWEEP (basket total %, taker {a.fee_bps}bps) — entry x exit:")
        print("   entry\\exit " + " ".join(f"{x:>7d}" for x in (5, 10, 20, 40, 55)))
        for entry in (10, 20, 40, 55, 80):
            row = []
            for exit in (5, 10, 20, 40, 55):
                if exit >= entry:
                    row.append("      -")
                    continue
                per = run_symbols(data, entry=entry, exit=exit, fee_bps=a.fee_bps,
                                  atr_window=db.DEFAULT_ATR, stop_atr=db.DEFAULT_STOP_ATR,
                                  allow_short=a.short)
                row.append(f"{per['__basket__']['total_return']*100:>+7.1f}")
            print(f"   {entry:>9d} " + " ".join(row))

    if a.walk_forward:
        print(f"\nWALK-FORWARD (pick entry/exit on train, report next fold), "
              f"basket, taker {a.fee_bps}bps")
        grid = [(e, x) for e in (20, 40, 55, 80) for x in (10, 20, 40) if x < e]
        idx = pd.date_range(t0, t1, periods=6)
        print("   fold      train-best(e,x)   train%    test%   test B&H%")
        for i in range(4):
            lo, hi = idx[i], idx[i + 1]
            tr = {s: d[(d.index >= lo) & (d.index < hi)] for s, d in data.items()}
            te = {s: d[(d.index >= hi) & (d.index < idx[i + 2])] for s, d in data.items()}
            tr = {s: d for s, d in tr.items() if len(d) > 120}
            te = {s: d for s, d in te.items() if len(d) > 60}
            if not tr or not te:
                continue
            best, best_ret = None, -9e9
            for (e, x) in grid:
                r = run_symbols(tr, entry=e, exit=x, fee_bps=a.fee_bps,
                                allow_short=a.short)["__basket__"]["total_return"]
                if r > best_ret:
                    best, best_ret = (e, x), r
            e, x = best
            test = run_symbols(te, entry=e, exit=x, fee_bps=a.fee_bps,
                               allow_short=a.short)["__basket__"]
            print(f"   {i+1:<4} {best!s:>16}  {best_ret*100:>+8.1f}  "
                  f"{test['total_return']*100:>+8.1f}  {test['buy_hold_return']*100:>+8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
