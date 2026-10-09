"""Train a drop-in Apex model from real Alpaca 15m bars.

Pipeline mirrors the repo's inference EXACTLY:
  bars -> add_features(df)[FEATURE_COLS]  (this row is what SafeMLPredictor's
  sklearn path / backtest_apex's infer_sk consumes as the "last bar")
  label -> forward return over HORIZON bars > 0
Model is a StandardScaler + HistGradientBoostingClassifier pipeline saved with
joblib, which SafeMLPredictor._load accepts as a sklearn champion (a .joblib
file), and which backtest_apex.make_infer runs through infer_sk.

Usage:
  python research/train_model.py --csv-dir bt_cache --out apex_gbdt.joblib \
      --horizon 8 --walk-forward
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_engineering import FEATURE_COLS, add_features  # noqa: E402

UPCAP = 8.0  # winsorize feature z-scores before fitting


def load_bars(csv_dir: str, symbols: list[str] | None) -> dict[str, pd.DataFrame]:
    files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    files += sorted(glob.glob(os.path.join(csv_dir, "*.csv.gz")))
    out: dict[str, pd.DataFrame] = {}
    for fp in files:
        stem = os.path.basename(fp).split("_15m_")[0]
        sym = stem.replace("_", "/")
        if symbols and sym not in symbols:
            continue
        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        # the cache can hold several date-ranges per symbol; keep the longest
        if len(df) and (sym not in out or len(df) > len(out[sym])):
            out[sym] = df
    return out


def build_dataset(bars: dict[str, pd.DataFrame], horizon: int) -> pd.DataFrame:
    rows = []
    for sym, df in bars.items():
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
    m = make_pipeline(
        StandardScaler(),
        HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3,
                                       l2_regularization=1.0, early_stopping=True,
                                       validation_fraction=0.15, random_state=0),
    )
    m.fit(np.clip(X, -UPCAP, UPCAP), y)
    return m


def walk_forward(D: pd.DataFrame, folds: int = 5, topq: float = 0.9, cost: float = 0.005):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    X = D[FEATURE_COLS].to_numpy(float)
    y = D["y"].to_numpy()
    fwd = D["fwd"].to_numpy()
    n = len(D)
    edges = np.linspace(0, n, folds + 1).astype(int)
    res = []
    for k in range(1, folds):
        tr = np.zeros(n, bool); tr[:edges[k]] = True
        te = np.zeros(n, bool); te[edges[k]:edges[k + 1]] = True
        m = _fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        top = p >= np.quantile(p, topq)
        res.append({
            "fold": k,
            "auc": roc_auc_score(y[te], p),
            "ic": spearmanr(p, fwd[te]).statistic,
            "top_hit": y[te][top].mean(),
            "top_gross_bps": fwd[te][top].mean() * 1e4,
            "top_net_bps": (fwd[te][top].mean() - cost) * 1e4,
            "n_te": int(te.sum()),
        })
    return pd.DataFrame(res)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default="bt_cache")
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--out", default="apex_gbdt.joblib")
    ap.add_argument("--walk-forward", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-train-frac", type=float, default=0.7)
    ap.add_argument("--as-of", default=None,
                    help="drop rows at/after this date before the final fit (for an honest OOS backtest)")
    a = ap.parse_args(argv)

    syms = [s.strip() for s in a.symbols.split(",")] if a.symbols else None
    bars = load_bars(a.csv_dir, syms)
    if not bars:
        print(f"No bars found in {a.csv_dir}")
        return 2
    print(f"Loaded {len(bars)} symbols: {', '.join(bars)}")
    D = build_dataset(bars, a.horizon)
    print(f"Dataset: {len(D):,} rows, {D['ts'].min()} -> {D['ts'].max()}, "
          f"base up-rate {D['y'].mean():.3f}")

    if a.as_of:
        asof = pd.Timestamp(a.as_of)
        D = D[D["ts"] < asof].reset_index(drop=True)
        print(f"--as-of {asof}: training set trimmed to {len(D):,} rows "
              f"({D['ts'].min()} -> {D['ts'].max()})")

    if a.walk_forward:
        wf = walk_forward(D, folds=a.folds)
        print(wf.to_string(index=False))
        print(f"\nmean AUC {wf['auc'].mean():.4f} | mean IC {wf['ic'].mean():+.4f} | "
              f"mean top-decile gross {wf['top_gross_bps'].mean():+.1f} bps | "
              f"net(-50bps) {wf['top_net_bps'].mean():+.1f} bps")

    n = len(D)
    cut = int(n * a.min_train_frac)
    Xtr = D[FEATURE_COLS].to_numpy(float)[:cut]
    ytr = D["y"].to_numpy()[:cut]
    Xte = D[FEATURE_COLS].to_numpy(float)[cut:]
    yte = D["y"].to_numpy()[cut:]
    fwdte = D["fwd"].to_numpy()[cut:]
    model = _fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    top = p >= np.quantile(p, 0.9)
    print(f"\nFinal holdout (last {1-a.min_train_frac:.0%}, {len(yte):,} rows): "
          f"AUC {roc_auc_score(yte, p):.4f} | IC {spearmanr(p, fwdte).statistic:+.4f} | "
          f"top-decile hit {yte[top].mean():.3f} (base {yte.mean():.3f}) | "
          f"gross {fwdte[top].mean()*1e4:+.1f} bps")
    import joblib
    joblib.dump(model, a.out)
    print(f"Saved {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
