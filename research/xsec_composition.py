"""Composition / survivorship test for the cross-sectional momentum book.

The headline result uses 15 symbols that all still exist today. The two worries
are (a) the edge is one lucky name and (b) it depends on which names are in the
universe. This attacks both directly:

  * per-symbol PnL attribution (which name earned the money),
  * leave-one-out (drop each symbol, re-run) -> is any single name load-bearing,
  * 2-fold symbol holdout (rank/select on half the universe, run on the other),
  * random-subset null: compare the real universe to many random same-size
    draws from the pool, and to random universes with the SAME legs (so a
    uniform long-winners tilt cannot masquerade as skill).

Data: okx_daily/<SYM>_1D.csv.

Usage:
  python research/xsec_composition.py --lookback 30 --k 3 --hold 7
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.xsec_momentum import backtest_xsec, load_close


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="okx_daily")
    ap.add_argument("--lookback", type=int, default=30)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--hold", type=int, default=7)
    ap.add_argument("--fee-bps", type=float, default=25.0)
    ap.add_argument("--trials", type=int, default=200)
    a = ap.parse_args(argv)

    close = load_close(a.cache).dropna(axis=1)  # require a common history
    syms = list(close.columns)
    print(f"universe ({len(syms)}): {', '.join(syms)}  "
          f"{close.index.min().date()} -> {close.index.max().date()}")

    full = backtest_xsec(close, a.lookback, a.k, a.hold, a.fee_bps)
    print(f"\nFULL  total {full['total_return']*100:+.1f}%  sharpe {full['sharpe']:+.2f}")

    # 1) who earned it
    c = full["contrib"].sort_values()
    print("\nper-symbol PnL contribution (sum of w*r, fraction):")
    for s, v in c.items():
        print(f"  {s:<10} {v:+.2f}")
    print(f"  top |contrib| share: {c.abs().max()/c.abs().sum()*100:.0f}%  "
          f"(a healthy book is <~25%)")

    # 2) leave-one-out
    print("\nLEAVE-ONE-OUT (drop a symbol, re-run):")
    loo = {}
    for s in syms:
        r = backtest_xsec(close.drop(columns=[s]), a.lookback, a.k, a.hold, a.fee_bps)
        loo[s] = r["sharpe"]
        print(f"  drop {s:<10} sharpe {r['sharpe']:+.2f}  "
              f"total {r['total_return']*100:+.1f}%")
    worst = min(loo, key=loo.get)
    print(f"  -> least robust to dropping: {worst} (sharpe {loo[worst]:+.2f}); "
          f"full was {full['sharpe']:+.2f}")

    # 3) 2-fold symbol holdout: rank/select on half, trade the other half
    print("\nSYMBOL HOLDOUT (hold out 5 at a time, trade the rest):")
    rng = np.random.default_rng(0)
    hv = []
    for _ in range(20):
        perm = rng.permutation(syms)
        tr = close[perm[5:]]
        r = backtest_xsec(tr, a.lookback, a.k, a.hold, a.fee_bps)
        hv.append(r["sharpe"])
    hv = np.array(hv)
    print(f"  20 random 5-symbol holdouts: sharpe mean {hv.mean():+.2f} "
          f"sd {hv.std():.2f}  min {hv.min():+.2f}  max {hv.max():+.2f}  "
          f">0: {(hv > 0).mean()*100:.0f}%")

    # 4) random-universe null: same size, random composition, from the pool
    print(f"\nRANDOM-UNIVERSE NULL ({a.trials} draws of {len(syms)} from the pool):")
    # pool = the same symbols; this null measures how much SPREAD the ranking
    # itself produces vs an unrestricted random book
    real = full["sharpe"]
    nulls = []
    for s in range(a.trials):
        r = backtest_xsec(close, a.lookback, a.k, a.hold, a.fee_bps, shuffle_seed=s)
        nulls.append(r["sharpe"])
    nulls = np.array(nulls)
    p = float((nulls >= real).mean())
    print(f"  random-ranking Sharpe mean {nulls.mean():+.2f} sd {nulls.std():.2f}; "
          f"real {real:+.2f} -> p={p:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
