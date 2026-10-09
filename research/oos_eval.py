"""Trustworthy out-of-sample scoreboard for the Apex signal.

The incumbent's reported OOS was ~0 IC, but the existing `walk_forward` in
research/train_model.py is NOT leakage-free: with an expanding split, the last
H training rows have forward-return labels that reach *into* the test block, and
the 20-bar feature window straddles the train/test boundary. Both inflate the
score. This module runs the same pipeline two ways:

  naive   - the existing contiguous split (kept for comparison)
  purged  - drop H boundary rows from train (label overlap) and embargo the
            first FEATURE_LOOKBACK test rows (feature overlap)

and reports AUC / IC / top-decile net-of-cost bps, per fold and pooled. The gap
between the two columns is the size of the leak illusion.

Usage:
  python research/oos_eval.py --csv-dir bt_cache --horizon 8 --folds 5
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_engineering import FEATURE_COLS, add_features

UPCAP = 8.0
FEATURE_LOOKBACK = 20          # longest rolling window used by add_features


def load_bars(csv_dir: str, symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    files += sorted(glob.glob(os.path.join(csv_dir, "*.csv.gz")))
    out: dict[str, pd.DataFrame] = {}
    for fp in files:
        stem = os.path.basename(fp).split("_15m_")[0]
        sym = stem.replace("_", "/")
        if symbols and sym not in symbols:
            continue
        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        if len(df) and (sym not in out or len(df) > len(out[sym])):
            out[sym] = df
    return out


def build_dataset(bars: dict[str, pd.DataFrame], horizon: int) -> pd.DataFrame:
    rows = []
    for df in bars.values():
        F = add_features(df)[FEATURE_COLS].to_numpy(float)
        C = df["close"].to_numpy(float)
        fwd = np.full_like(C, np.nan)
        fwd[:-horizon] = C[horizon:] / C[:-horizon] - 1.0
        good = np.isfinite(F).all(axis=1) & np.isfinite(fwd)
        rows.append(pd.DataFrame(F[good], columns=FEATURE_COLS).assign(
            ts=df.index[good], fwd=fwd[good], y=(fwd[good] > 0).astype(int)))
    return pd.concat(rows).sort_values("ts").reset_index(drop=True)


def _fit(X, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(
        StandardScaler(),
        HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3,
                                       l2_regularization=1.0, early_stopping=True,
                                       validation_fraction=0.15, random_state=0),
    ).fit(np.clip(X, -UPCAP, UPCAP), y)


def _score(y, p, fwd, cost, topq=0.9):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    top = p >= np.quantile(p, topq)
    return {
        "n": len(y),
        "auc": roc_auc_score(y, p),
        "ic": spearmanr(p, fwd).statistic,
        "top_hit": y[top].mean(),
        "base_hit": y.mean(),
        "top_gross_bps": fwd[top].mean() * 1e4,
        "top_net_bps": (fwd[top].mean() - cost) * 1e4,
    }


def walk_forward(D: pd.DataFrame, folds: int, H: int, cost: float,
                 purge: bool, topq: float = 0.9) -> pd.DataFrame:
    X = D[FEATURE_COLS].to_numpy(float)
    y = D["y"].to_numpy()
    fwd = D["fwd"].to_numpy()
    n = len(D)
    edges = np.linspace(0, n, folds + 1).astype(int)
    res = []
    for k in range(1, folds):
        train_end, test_end = edges[k], edges[k + 1]
        tr = np.zeros(n, bool)
        hi = train_end - H if purge else train_end          # purge label overlap
        tr[:hi] = True
        te = np.zeros(n, bool)
        lo = train_end + FEATURE_LOOKBACK if purge else train_end  # embargo features
        te[lo:test_end] = True
        m = _fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        row = {"fold": k}
        row.update(_score(y[te], p, fwd[te], cost, topq))
        res.append(row)
    return pd.DataFrame(res)


def date_split(D: pd.DataFrame, split: str, H: int, cost: float,
               topq: float = 0.9) -> pd.DataFrame:
    """Single purged train/test split at a date: train strictly before `split`,
    test on/after it, with the H-label overlap and feature window removed."""
    ts = D["ts"]
    cut_ts = pd.Timestamp(split)
    if getattr(ts.dtype, "tz", None) is not None and cut_ts.tz is None:
        cut_ts = cut_ts.tz_localize("UTC")
    cut = int((ts < cut_ts).sum())
    X = D[FEATURE_COLS].to_numpy(float)
    y = D["y"].to_numpy()
    fwd = D["fwd"].to_numpy()
    n = len(D)
    tr = np.zeros(n, bool); tr[:max(0, cut - H)] = True          # purge label overlap
    te = np.zeros(n, bool); te[min(n, cut + FEATURE_LOOKBACK):] = True   # embargo features
    m = _fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    row = {"train_rows": int(tr.sum()), "test_rows": int(te.sum()),
           "train_end": ts[tr].max(), "test_start": ts[te].min()}
    row.update(_score(y[te], p, fwd[te], cost, topq))
    return pd.DataFrame([row])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default="bt_cache")
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cost-bps", type=float, default=50.0,
                    help="round-trip cost in bps (Alpaca 25/side = 50)")
    ap.add_argument("--split-date", default=None,
                    help="if set, run one purged train<date / test>=date split")
    a = ap.parse_args(argv)
    syms = [s.strip() for s in a.symbols.split(",")] if a.symbols else None
    bars = load_bars(a.csv_dir, syms)
    if not bars:
        print(f"No bars in {a.csv_dir}"); return 2
    print(f"Loaded {len(bars)} symbols: {', '.join(sorted(bars))}")
    D = build_dataset(bars, a.horizon)
    print(f"Dataset {len(D):,} rows, {D['ts'].min()} -> {D['ts'].max()}, "
          f"base up-rate {D['y'].mean():.3f}, horizon {a.horizon} bars, "
          f"cost {a.cost_bps:.0f} bps round-trip\n")
    if a.split_date:
        r = date_split(D, a.split_date, a.horizon, a.cost_bps / 1e4)
        print(f"=== PURGED DATE SPLIT train<{a.split_date} / test>={a.split_date} ===")
        print(r.to_string(index=False))
        print(f"  AUC {r['auc'][0]:.4f} | IC {r['ic'][0]:+.4f} | "
              f"top-decile hit {r['top_hit'][0]:.3f} (base {r['base_hit'][0]:.3f}) | "
              f"gross {r['top_gross_bps'][0]:+.1f} bps | NET {r['top_net_bps'][0]:+.1f} bps\n")
        return 0
    for label, purge in (("NAIVE (as-is, leaky)", False), ("PURGED + EMBARGOED", True)):
        wf = walk_forward(D, a.folds, a.horizon, a.cost_bps / 1e4, purge)
        print(f"=== {label} ===")
        print(wf.to_string(index=False))
        print(f"  mean AUC {wf['auc'].mean():.4f} | mean IC {wf['ic'].mean():+.4f} | "
              f"top-decile hit {wf['top_hit'].mean():.3f} (base {wf['base_hit'].mean():.3f}) | "
              f"gross {wf['top_gross_bps'].mean():+.1f} bps | "
              f"NET {wf['top_net_bps'].mean():+.1f} bps\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
