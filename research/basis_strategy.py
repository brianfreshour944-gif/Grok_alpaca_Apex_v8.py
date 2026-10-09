"""Delta-neutral cash-and-carry: short perp + long spot.

This is the construction that actually neutralises the price term the previous
carry book could not. For each selected symbol we hold a matched-notional pair
(short 1 perp, long 1 spot). Per interval the pair earns:

    funding   : a short perp receives funding when funding > 0
    price     : r_spot - r_perp  ~= -(change in basis)

The basis is small and mean-reverting, so the price term should be near zero,
leaving the funding carry -- if the hedge is tight and costs are low enough.

Cost is charged on BOTH legs at each rebalance, so the holding period matters a
lot (spot taker is expensive).

Usage:
  python research/basis_strategy.py --sweep --recent 2024-01-01
"""
from __future__ import annotations

import argparse
import glob
import itertools
import os

import numpy as np
import pandas as pd

try:
    from perp_strategy import load_funding, load_klines
except ImportError:
    from research.perp_strategy import load_funding, load_klines

KL_COLS = ["open_time", "open", "high", "low", "close", "volume"]


def _load_spot(cache: str) -> dict[str, pd.Series]:
    out = {}
    for fp in sorted(glob.glob(os.path.join(cache, "*_1h.csv"))):
        sym = os.path.basename(fp).replace("_1h.csv", "")
        df = pd.read_csv(fp).iloc[:, :6]
        df.columns = KL_COLS
        raw = df["open_time"].astype("int64").to_numpy()
        # Binance spot switched ms -> us mid-history; the concatenated file has
        # BOTH, so normalise per row (us ~1.6e15, ms ~1.6e12; split at 1e14).
        ms = np.where(raw > 10**14, raw // 1000, raw)
        df["open_time"] = pd.to_datetime(ms, unit="ms", utc=True)
        out[sym] = pd.to_numeric(df.set_index("open_time")["close"], errors="coerce").sort_index()
    return out


def build_panel(perp_cache: str, spot_cache: str, horizon: int = 8) -> pd.DataFrame:
    kl, fd, sp = load_klines(perp_cache), load_funding(perp_cache), _load_spot(spot_cache)
    rows = []
    for sym, df in kl.items():
        if sym not in sp:
            continue
        p, s = df["close"], sp[sym]
        idx = p.index.intersection(s.index)
        if len(idx) < 400:
            continue
        p, s = p.loc[idx], s.loc[idx]
        f = fd.get(sym)
        ff = f.reindex(idx, method="ffill") if f is not None else pd.Series(np.nan, index=idx)
        g = pd.DataFrame({
            "sym": sym, "basis": p / s - 1.0,
            "funding": ff,
            # the stamp actually settled on the position opened at this bar is
            # the NEXT one (funding accrues every 8h), i.e. horizon hours later.
            "funding_fwd": ff.shift(-horizon),
            "fwd_perp": p.pct_change(horizon).shift(-horizon),
            "fwd_spot": s.pct_change(horizon).shift(-horizon),
        })
        rows.append(g)
    P = pd.concat(rows).reset_index().rename(columns={"open_time": "ts"})
    P = P[P["ts"].dt.hour % horizon == 0]
    P = P.dropna(subset=["fwd_perp", "fwd_spot", "funding_fwd"]).sort_values(["ts", "sym"])
    P = P.reset_index(drop=True)
    g = P.groupby("ts")["funding"]
    P["funding_xs"] = (P["funding"] - g.transform("mean")) / g.transform("std")
    return P


def _group_index(P):
    ts = P["ts"].to_numpy()
    change = np.flatnonzero(ts[1:] != ts[:-1]) + 1
    bounds = np.concatenate([[0], change, [len(ts)]])
    sym2id = {s: i for i, s in enumerate(sorted(P["sym"].unique()))}
    return bounds, P["sym"].map(sym2id).to_numpy(), sym2id


def select_legs(P: pd.DataFrame, k: int):
    """Yield (ts, short_syms) for the carry book: the k perps with the most
    expensive funding. Each is SHORTED and hedged with a matching long spot;
    there is no long-bottom-k leg (the book is delta-neutral per pair, not
    long/short in the perp). Single source of truth for backtest and ledger."""
    bounds, ids, sym2id = _group_index(P)
    id2sym = {i: s for s, i in sym2id.items()}
    fxs = P["funding_xs"].to_numpy(float)
    ts_arr = P["ts"].to_numpy()
    for a, b in itertools.pairwise(bounds):
        if b - a < 2 * k + 1:
            continue
        order = np.argsort(-fxs[a:b])               # most-expensive funding first
        loc = ids[a:b]
        yield ts_arr[a], [id2sym[loc[i]] for i in order[:k]]


def run_basis(P: pd.DataFrame, k: int, hold: int = 1,
              perp_cost: float = 5.0, spot_cost: float = 10.0,
              slippage_bps: float = 0.0, transfer_bps: float = 0.0) -> pd.DataFrame:
    """Short top-k funding perps, each hedged long spot. Cost on both legs.

    Beyond the taker fees (perp_cost, spot_cost), `slippage_bps` adds per-leg
    execution slippage and `transfer_bps` adds a cross-venue collateral move;
    both are charged on turnover like the fees.
    """
    bounds, ids, sym2id = _group_index(P)
    nsym = len(sym2id)
    fund = P["funding_fwd"].to_numpy(float)
    fp = P["fwd_perp"].to_numpy(float)
    fs = P["fwd_spot"].to_numpy(float)
    ts_arr = P["ts"].to_numpy()
    cost = (perp_cost + spot_cost + 2 * slippage_bps + transfer_bps) / 1e4
    prev = np.zeros(nsym)
    idx, funding_r, price_r, net, turn_v = [], [], [], [], []
    step, started = 0, False
    legs = select_legs(P, k)
    for a, b in itertools.pairwise(bounds):
        if b - a < 2 * k + 1:
            continue
        ts, short_s = next(legs)
        rebal = (not started) or (step % hold == 0)
        if rebal:
            w = np.zeros(nsym)
            for sy in short_s:                      # short the expensive funding
                w[sym2id[sy]] += 1.0 / k
            turn = np.abs(w - prev).sum()
        else:
            w, turn = prev, 0.0
        f = np.zeros(nsym)
        f[ids[a:b]] = fund[a:b]
        p = np.zeros(nsym)
        p[ids[a:b]] = fp[a:b]
        s = np.zeros(nsym)
        s[ids[a:b]] = fs[a:b]
        funding_r.append((w @ f) * 1e4)             # short perp receives funding
        price_r.append((w @ (s - p)) * 1e4)         # long spot minus short perp
        net.append(((w @ f) + (w @ (s - p)) - cost * turn) * 1e4)
        turn_v.append(turn)
        idx.append(ts_arr[a])
        prev = w
        step += 1
        started = True
    return pd.DataFrame({"funding": funding_r, "price": price_r, "net": net,
                         "turnover": turn_v}, index=pd.DatetimeIndex(idx))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--perp-cache", default="perp_cache")
    ap.add_argument("--spot-cache", default="spot_cache")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--hold", type=int, default=21)
    ap.add_argument("--perp-cost", type=float, default=5.0)
    ap.add_argument("--spot-cost", type=float, default=10.0)
    ap.add_argument("--recent", default=None)
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args(argv)
    P = build_panel(a.perp_cache, a.spot_cache)
    print(f"basis panel: {len(P):,} rows, {P['sym'].nunique()} syms, "
          f"{P['ts'].min()} -> {P['ts'].max()}")
    print("mean basis by sym (bps): " + "  ".join(
        f"{s}:{v*1e4:+.1f}" for s, v in P.groupby('sym')['basis'].mean().items()))

    def show(label, sub):
        if sub is None or sub.empty:
            return
        print(f"\n{label}  (perp {a.perp_cost} + spot {a.spot_cost} bps/leg)")
        print(f"   {'k':>2} {'hold':>4} {'fund':>7} {'price':>7} {'net':>7} "
              f"{'%/yr':>7} {'win':>5}  by_year")
        for k in (3, 4):
            for hold in (21, 63, 126, 252):
                r = run_basis(sub, k, hold, a.perp_cost, a.spot_cost)
                ann = r["net"].mean() * 3 * 365 / 1e4 * 100
                yr = r["net"].groupby(r.index.year).mean().round(1).to_dict()
                print(f"   {k:>2} {hold:>4} {r['funding'].mean():>+7.2f} "
                      f"{r['price'].mean():>+7.2f} {r['net'].mean():>+7.2f} "
                      f"{ann:>+7.1f} {(r['net']>0).mean():>5.2f}  {yr}")

    show("FULL 2021-2026", P)
    if a.recent:
        show(f"RECENT >= {a.recent}", P[P["ts"] >= pd.Timestamp(a.recent, tz="UTC")])

    if a.sweep:
        Pr = P[P["ts"] >= pd.Timestamp(a.recent or "2024-01-01", tz="UTC")]
        print("\nSPOT-COST x HOLD sweep, k=3, hold long, RECENT (net bps/8h):")
        print(f"   {'hold\\spot':>9} " + " ".join(f"{c:>7.0f}" for c in (10, 5, 2, 1, 0)))
        for hold in (21, 63, 126, 252):
            row = [run_basis(Pr, 3, hold, 5.0, c)["net"].mean() for c in (10, 5, 2, 1, 0)]
            print(f"   {hold:>9} " + " ".join(f"{v:>+7.2f}" for v in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
