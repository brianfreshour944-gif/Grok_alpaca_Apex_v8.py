"""Funding-carry backtest on Binance USDⓈ-M perpetuals.

Carry is different from the price factors: the funding cash flow is *known* at
trade time, so this is an execution/cost question, not a forecasting one. A
market-neutral book that shorts the most-expensive-funding perps and longs the
cheapest collects funding while price risk cancels.

Perp position of weight w (long > 0) pays -w*funding each interval, so shorts
receive positive funding. Returns are reported as:
  gross_carry : funding-only PnL (the harvest)
  price       : price PnL of the same book (should be ~0 if truly neutral)
  net         : gross_carry + price - cost*turnover

Usage:
  python research/carry_strategy.py --k 3 --cost-bps 5
  python research/carry_strategy.py --k 3 --cost-bps 2 --recent 2024-01-01
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

try:                                    # script-style run: research/ on sys.path
    from perp_strategy import load_funding, load_klines
except ImportError:                     # package-style import (pytest)
    from research.perp_strategy import load_funding, load_klines


def build_panel(cache: str, rebal_hours: int = 8) -> pd.DataFrame:
    """Panel at rebalance timestamps: funding rate + forward price return."""
    kl, fd = load_klines(cache), load_funding(cache)
    rows = []
    for sym, df in kl.items():
        c = df["close"]
        r = c.pct_change(rebal_hours).shift(-rebal_hours)  # fwd return over interval
        f = fd.get(sym)
        if f is None:
            continue
        ff = f.reindex(df.index, method="ffill")
        g = pd.DataFrame({"sym": sym, "close": c, "fwd": r, "funding": ff})
        # funding actually earned by a position opened at this bar: the NEXT
        # stamp (a position set at ts is credited funding at ts+interval).
        g["funding_fwd"] = ff.shift(-rebal_hours // 8 if rebal_hours >= 8 else -1)
        rows.append(g)
    P = pd.concat(rows).reset_index().rename(columns={"open_time": "ts"})
    # keep only rows on rebalance boundaries
    P = P[P["ts"].dt.hour % rebal_hours == 0]
    P = P.dropna(subset=["fwd", "funding", "funding_fwd"]).sort_values(["ts", "sym"]).reset_index(drop=True)
    g = P.groupby("ts")["funding"]
    P["funding_xs"] = (P["funding"] - g.transform("mean")) / g.transform("std")
    return P


def _group_index(P: pd.DataFrame):
    ts = P["ts"].to_numpy()
    change = np.flatnonzero(ts[1:] != ts[:-1]) + 1
    bounds = np.concatenate([[0], change, [len(ts)]])
    sym2id = {s: i for i, s in enumerate(sorted(P["sym"].unique()))}
    ids = P["sym"].map(sym2id).to_numpy()
    return bounds, ids, len(sym2id)


def run_carry(P: pd.DataFrame, k: int, cost_bps: float, sign: int = -1,
              hold: int = 1) -> pd.DataFrame:
    """Short top-k funding, long bottom-k (sign=-1). Rebalance every `hold`
    intervals; accrue funding each interval in between.

    Rows: carry/price/net by ts. Cost is charged only when the book changes.
    """
    cost = cost_bps / 1e4
    bounds, ids, nsym = _group_index(P)
    fund = P["funding_fwd"].to_numpy(float)   # funding actually earned going forward
    fwd = P["fwd"].to_numpy(float)
    fxs = sign * P["funding_xs"].to_numpy(float)
    ts_arr = P["ts"].to_numpy()
    prev = np.zeros(nsym)
    idx, carry, price, net, turn_v = [], [], [], [], []
    step, started = 0, False
    for a, b in itertools.pairwise(bounds):
        if b - a < 2 * k + 1:
            continue
        rebal = (not started) or (step % hold == 0)
        if rebal:
            idg, fg = ids[a:b], fxs[a:b]
            order = np.argsort(-fg)             # most extreme score first
            w = np.zeros(nsym)
            w[idg[order[:k]]] += 1.0 / k        # short the most expensive funding
            w[idg[order[-k:]]] -= 1.0 / k       # long the cheapest
            turn = np.abs(w - prev).sum()
        else:
            w, turn = prev, 0.0
        f = np.zeros(nsym)
        f[ids[a:b]] = fund[a:b]
        r = np.zeros(nsym)
        r[ids[a:b]] = fwd[a:b]
        idx.append(ts_arr[a])
        carry.append(-(w @ f) * 1e4)            # long pays f, short receives f
        price.append((w @ r) * 1e4)
        net.append((-(w @ f) + (w @ r) - cost * turn) * 1e4)
        turn_v.append(turn)
        prev = w
        step += 1
        started = True
    return pd.DataFrame({"carry": carry, "price": price, "net": net, "turnover": turn_v},
                        index=pd.DatetimeIndex(idx))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="perp_cache")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--hold", type=int, default=1, help="rebalance every N intervals")
    ap.add_argument("--recent", default=None)
    ap.add_argument("--sweep", action="store_true", help="sweep hold x cost")
    a = ap.parse_args(argv)
    P = build_panel(a.cache)
    print(f"carry panel: {len(P):,} rows, {P['sym'].nunique()} syms, "
          f"{P['ts'].min()} -> {P['ts'].max()}\n")
    mf = P.groupby("sym")["funding"].mean().sort_values(ascending=False)
    print("mean funding by symbol (bps/8h, annualised):")
    print("   " + "  ".join(f"{s}:{v*1e4:+.2f}" for s, v in mf.items()))

    def show(label, sub):
        if sub is None or sub.empty:
            return
        print(f"\n{label}")
        print(f"   {'k':>2} {'hold':>4} {'cost':>5} {'carry':>6} {'price':>6} "
              f"{'turn':>5} {'net':>7} {'%/yr':>7} {'win':>5}  by_year")
        for k in (2, 3, 4):
            for hold in (1, 3, 9, 21):
                res = run_carry(sub, k, a.cost_bps, hold=hold)
                ann = res["net"].mean() * 3 * 365 / 1e4 * 100
                yr = res["net"].groupby(res.index.year).mean().round(1).to_dict()
                print(f"   {k:>2} {hold:>4} {a.cost_bps:>5.0f} "
                      f"{res['carry'].mean():>+6.2f} {res['price'].mean():>+6.2f} "
                      f"{res['turnover'].mean()/hold:>5.2f} {res['net'].mean():>+7.2f} "
                      f"{ann:>+7.1f} {(res['net']>0).mean():>5.2f}  {yr}")

    show("FULL 2021-2026", P)
    if a.recent:
        show(f"RECENT >= {a.recent}", P[P["ts"] >= pd.Timestamp(a.recent, tz="UTC")])

    if a.sweep:
        Pr = P[P["ts"] >= pd.Timestamp(a.recent or "2024-01-01", tz="UTC")]
        print("\nCOST x HOLD sweep, k=3, RECENT (net bps/8h):")
        print(f"   {'hold\\cost':>9} " + " ".join(f"{c:>7.0f}" for c in (5, 3, 2, 1, 0)))
        for hold in (1, 3, 9, 21, 42, 90):
            row = []
            for c in (5, 3, 2, 1, 0):
                row.append(run_carry(Pr, 3, c, hold=hold)["net"].mean())
            print(f"   {hold:>9} " + " ".join(f"{v:>+7.2f}" for v in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
