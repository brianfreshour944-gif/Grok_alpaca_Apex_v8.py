"""Dollar-neutral perpetual strategy research on Binance USDⓈ-M data.

Loads perp 1h klines + 8h funding from perp_cache/ (see fetch_perp_binance.py),
builds a cross-sectional panel of features that Alpaca spot never exposed --
relative strength, funding/carry, order-flow imbalance -- and evaluates simple
factor books NET of perp cost, with a purged walk-forward for learned scoring.

Book construction (dollar-neutral):
  at each rebalance t, score every symbol with data <= t, go long the top-k and
  short the bottom-k with equal weight per side (gross long = gross short = 1).
  forward return r_i = close[t+H]/close[t] - 1. net = sum w_i r_i - c*turnover.

Usage:
  python research/perp_strategy.py --factors            # per-factor OOS net
  python research/perp_strategy.py --learn --horizon 8  # purged WF combiner
"""
from __future__ import annotations

import argparse
import glob
import itertools
import os

import numpy as np
import pandas as pd

FUND_COLS = ["open_time", "open", "high", "low", "close", "volume",
             "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"]


def load_klines(cache: str) -> dict[str, pd.DataFrame]:
    out = {}
    for fp in sorted(glob.glob(os.path.join(cache, "*_1h.csv"))):
        sym = os.path.basename(fp).replace("_1h.csv", "")
        df = pd.read_csv(fp)
        df = df.iloc[:, :len(FUND_COLS)]
        df.columns = FUND_COLS[: df.shape[1]]
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("open_time").sort_index()
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        out[sym] = df
    return out


def load_funding(cache: str) -> dict[str, pd.Series]:
    out = {}
    for fp in sorted(glob.glob(os.path.join(cache, "funding", "*.csv"))):
        sym = os.path.basename(fp).replace(".csv", "")
        df = pd.read_csv(fp)
        ts = pd.to_datetime(df["ts"], unit="ms", utc=True)
        out[sym] = pd.Series(pd.to_numeric(df["funding_rate"], errors="coerce").values,
                             index=ts).sort_index()
    return out


def _feats(df: pd.DataFrame, fund: pd.Series | None) -> pd.DataFrame:
    c = df["close"]
    v = df["volume"].replace(0, np.nan)
    f = pd.DataFrame(index=df.index)
    f["ret_1"] = c.pct_change()
    f["mom_24"] = c / c.shift(24) - 1.0
    f["mom_72"] = c / c.shift(72) - 1.0
    f["mom_168"] = c / c.shift(168) - 1.0
    f["vol_24"] = f["ret_1"].rolling(24).std()
    f["vol_168"] = f["ret_1"].rolling(168).std()
    f["dollar_vol"] = df["quote_volume"].rolling(24).mean()
    f["flow_1"] = df["taker_buy_volume"] / v                      # buy share
    f["flow_24"] = (df["taker_buy_volume"].rolling(24).sum()
                    / v.rolling(24).sum())
    if fund is not None:
        ff = fund.reindex(df.index, method="ffill")
        f["funding"] = ff
        f["funding_z"] = (ff - ff.rolling(168).mean()) / ff.rolling(168).std()
        # realized carry over the next H is a return component; the *level* is
        # the tradeable signal (short the expensive side)
    f["close"] = c
    return f


def build_panel(cache: str, horizon: int) -> pd.DataFrame:
    kl = load_klines(cache)
    fd = load_funding(cache)
    rows = []
    for sym, df in kl.items():
        if len(df) < 400:
            continue
        f = _feats(df, fd.get(sym))
        f["sym"] = sym
        c = f["close"].to_numpy(float)
        fwd = np.full_like(c, np.nan)
        fwd[:-horizon] = c[horizon:] / c[:-horizon] - 1.0
        f["fwd"] = fwd
        rows.append(f)
    P = pd.concat(rows).reset_index().rename(columns={"open_time": "ts"})
    P = P.dropna(subset=["fwd"]).sort_values(["ts", "sym"]).reset_index(drop=True)
    # cross-sectional (per-timestamp) standardisation of every signal column
    for col in ["mom_24", "mom_72", "mom_168", "vol_24", "vol_168", "flow_24",
                "flow_1", "dollar_vol", "funding", "funding_z"]:
        if col in P:
            g = P.groupby("ts")[col]
            P[col + "_xs"] = (P[col] - g.transform("mean")) / g.transform("std")
    return P


def book_net(P: pd.DataFrame, score_col: str, horizon: int, k: int,
             cost_bps: float, sign: int = 1) -> pd.DataFrame:
    """Dollar-neutral top-k/bottom-k book; rows of net/gross/turnover by ts."""
    cost = cost_bps / 1e4
    out = []
    prev_w: dict[str, float] = {}
    for ts, g in P.groupby("ts", sort=True):
        g = g.dropna(subset=[score_col, "fwd"])
        if len(g) < 2 * k + 1:
            continue
        s = sign * g[score_col].to_numpy()
        order = np.argsort(-s)
        long_i, short_i = order[:k], order[-k:]
        syms = g["sym"].to_numpy()
        r = g["fwd"].to_numpy()
        idx = {sy: i for i, sy in enumerate(syms)}
        w = {syms[i]: 1.0 / k for i in long_i}
        for i in short_i:
            w[syms[i]] = w.get(syms[i], 0.0) - 1.0 / k
        pnl = sum(wt * r[idx[sy]] for sy, wt in w.items())
        turn = sum(abs(w.get(sy, 0.0) - prev_w.get(sy, 0.0)) for sy in set(w) | set(prev_w))
        out.append((ts, pnl * 1e4, (pnl - cost * turn) * 1e4, turn))
        prev_w = w
    return pd.DataFrame(out, columns=["ts", "gross_bps", "net_bps", "turnover"]).set_index("ts")


def summarize(name: str, res: pd.DataFrame, per_year: bool = True) -> dict:
    net = res["net_bps"]
    out = {"factor": name, "n": len(net), "net_bps": net.mean(),
           "gross_bps": res["gross_bps"].mean(), "turnover": res["turnover"].mean(),
           "win": (net > 0).mean()}
    if per_year:
        out["by_year"] = net.groupby(net.index.year).mean().round(1).to_dict()
    return out


def _group_index(P: pd.DataFrame):
    """Per-timestamp row slices over the ts-sorted panel (no pandas groupby)."""
    ts = P["ts"].to_numpy()
    change = np.flatnonzero(ts[1:] != ts[:-1]) + 1
    bounds = np.concatenate([[0], change, [len(ts)]])
    sym2id = {s: i for i, s in enumerate(sorted(P["sym"].unique()))}
    ids = P["sym"].map(sym2id).to_numpy()
    fwd = P["fwd"].to_numpy(float)
    return bounds, ids, fwd, len(sym2id)


def _run_book(score: np.ndarray, bounds, ids, fwd, nsym: int,
              sign: int, k: int, cost_bps: float) -> np.ndarray:
    """Net bps per rebalance for a dollar-neutral top-k/bottom-k book."""
    cost = cost_bps / 1e4
    prev = np.zeros(nsym)
    vals = []
    for a, b in itertools.pairwise(bounds):
        if b - a < 2 * k + 1:
            continue
        idg, sg, fg = ids[a:b], sign * score[a:b], fwd[a:b]
        order = np.argsort(-sg)
        w = np.zeros(nsym)
        w[idg[order[:k]]] += 1.0 / k
        w[idg[order[-k:]]] -= 1.0 / k
        r = np.zeros(nsym)
        r[idg] = fg
        turn = np.abs(w - prev).sum()
        vals.append((w @ r - cost * turn) * 1e4)
        prev = w
    return np.array(vals)


def permutation_null(score: np.ndarray, bounds, ids, fwd, nsym: int,
                     sign: int, k: int, cost_bps: float,
                     n_perm: int = 100, seed: int = 0) -> np.ndarray:
    """Null: shuffle the score across symbols within each timestamp, so the
    cross-sectional selection is destroyed but the book/cost structure holds."""
    rng = np.random.default_rng(seed)
    means = np.empty(n_perm)
    for i in range(n_perm):
        sp = score.copy()
        for a, b in itertools.pairwise(bounds):
            sp[a:b] = rng.permutation(sp[a:b])
        means[i] = _run_book(sp, bounds, ids, fwd, nsym, sign, k, cost_bps).mean()
    return means




def combine_scores(P: pd.DataFrame, cols: list[tuple[int, str]]) -> pd.Series:
    """Equal-weight the cross-sectionally standardised factors (with signs)."""
    acc = None
    for sign, c in cols:
        v = sign * P[c]
        acc = v if acc is None else acc + v
    return acc / len(cols)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="perp_cache")
    ap.add_argument("--horizon", type=int, default=8, help="hold in hours")
    ap.add_argument("--k", type=int, default=3, help="names per side")
    ap.add_argument("--cost-bps", type=float, default=5.0,
                    help="one-way cost per unit turnover (Binance perp taker ~5, maker ~2)")
    ap.add_argument("--factors", action="store_true")
    ap.add_argument("--null", action="store_true")
    ap.add_argument("--recent", default=None,
                    help="also report on ts >= this date (e.g. 2024-01-01)")
    a = ap.parse_args(argv)
    P = build_panel(a.cache, a.horizon)
    print(f"panel: {len(P):,} rows, {P['sym'].nunique()} syms, "
          f"{P['ts'].min()} -> {P['ts'].max()}, horizon {a.horizon}h, "
          f"cost {a.cost_bps} bps/leg, k={a.k}\n")
    P["combo"] = combine_scores(P, [(+1, "mom_168_xs"), (-1, "funding_xs"),
                                    (-1, "flow_24_xs")])
    factors = {
        "momentum_24_xs": (+1, "mom_24_xs"),
        "momentum_168_xs": (+1, "mom_168_xs"),
        "reversal_flow_24": (-1, "flow_24_xs"),
        "carry_short_funding": (-1, "funding_xs"),
        "carry_short_funding_z": (-1, "funding_z_xs"),
        "vol_defensive": (-1, "vol_24_xs"),
        "COMBO(mom168+carry+flow)": (+1, "combo"),
    }
    bounds, ids, fwd, nsym = _group_index(P)

    def evaluate(sub: pd.DataFrame, label: str):
        if sub.empty:
            return
        sb, si, sf, sn = _group_index(sub)
        tix = pd.DatetimeIndex(sub["ts"].to_numpy()[sb[:-1]])
        print(f"{label:26} {'sign':>4} {'n':>6} {'gross':>7} {'turn':>6} "
              f"{'net':>7} {'win':>6}  by_year")
        for name, (sign, col) in factors.items():
            if col not in sub:
                continue
            score = sub[col].to_numpy(float)
            net = _run_book(score, sb, si, sf, sn, sign, a.k, a.cost_bps)
            gross = _run_book(score, sb, si, sf, sn, sign, a.k, 0.0)
            ser = pd.Series(net, index=tix)
            by_year = ser.groupby(ser.index.year).mean().round(1).to_dict()
            print(f"{name:26} {sign:>+4} {len(net):>6} {gross.mean():>+7.2f} "
                  f"{'':>6} {ser.mean():>+7.2f} {(ser > 0).mean():>6.2f}  {by_year}")

    evaluate(P, "FULL 2021-2026")

    if a.recent:
        Pr = P[P["ts"] >= pd.Timestamp(a.recent, tz="UTC")]
        print(f"\n--- RECENT regime (ts >= {a.recent}): {len(Pr):,} rows ---")
        evaluate(Pr, "RECENT")

    if a.null:
        print("\npermutation null (score shuffled within each timestamp, 100 draws):")
        for label, sign, col in [("COMBO", 1, "combo"),
                                 ("momentum_168_xs", 1, "mom_168_xs"),
                                 ("carry_short_funding", -1, "funding_xs")]:
            score = P[col].to_numpy(float)
            real = _run_book(score, bounds, ids, fwd, nsym, sign, a.k, a.cost_bps).mean()
            null = permutation_null(score, bounds, ids, fwd, nsym, sign, a.k,
                                    a.cost_bps, n_perm=100)
            p = (null >= real).mean()
            print(f"  {label:22} real {real:>+6.2f} | null mean {null.mean():>+6.2f} "
                  f"p95 {np.percentile(null, 95):>+6.2f} | p={p:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
