"""Funding / positioning features: does derivative funding add a tradeable,
beta-free edge on top of the 11 microstructure features?

Data: Binance USDT-M monthly fundingRate archives (data.binance.vision), which
return full hourly history (OKX's funding endpoint only exposes ~100 days).
Fetch once with `fetch_funding()` into bt_cache/funding/ (gitignored).

Findings (see research/RESULTS.md "Funding / better-inputs investigation"):
  - Funding helps prediction most at MULTI-DAY horizons, not at 2h.
  - The top-decile "alpha" at 72h (+117 bps) is BETA, not timing: a circular
    timing-alpha null gives p=0.75.
  - The beta-free arbiter (non-overlapping market-neutral long-short) shows a
    +90 bps gross edge at 7d with funding -- the best candidate found -- but
    only ~38 observations (gross t~1.3) and it is eaten by costs.

Run: python research/funding_probe.py
"""
from __future__ import annotations

import io
import os
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_model as T  # noqa: E402
from feature_engineering import FEATURE_COLS  # noqa: E402

FUND_DIR = os.environ.get("FUNDING_DIR", "bt_cache/funding")
FCOLS = ["funding", "funding_z", "funding_cum_7d", "funding_cum_14d", "xs_fund_rank", "xs_fund_z"]


def fetch_funding(symbols, months=None, out_dir=FUND_DIR):
    os.makedirs(out_dir, exist_ok=True)
    months = months or [f"{y}-{m:02d}" for y in (2025, 2026) for m in range(1, 13)]
    months = [m for m in months if "2025-04" <= m <= "2026-09"]

    def _url(sym, mon):
        return (f"https://data.binance.vision/data/futures/um/monthly/fundingRate/"
                f"{sym}USDT/{sym}USDT-fundingRate-{mon}.zip")

    for sym in symbols:
        frames = []
        for mon in months:
            try:
                raw = urllib.request.urlopen(_url(sym, mon), timeout=30).read()
                zf = zipfile.ZipFile(io.BytesIO(raw))
                frames.append(pd.read_csv(zf.open(zf.namelist()[0])))
            except Exception:
                continue
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(
                os.path.join(out_dir, f"funding_{sym}.csv"), index=False)


def _load_panel(bars, out_dir=FUND_DIR):
    """Return (per-symbol funding frame dict, cross-sectional funding panel)."""
    fund = {}
    for sym in bars:
        s = sym.split("/")[0]
        fp = os.path.join(out_dir, f"funding_{s}.csv")
        f = pd.read_csv(fp)
        f.index = pd.to_datetime(f["calc_time"], unit="ms").dt.tz_localize(None)
        fund[sym] = f["last_funding_rate"].sort_index()
    return fund, pd.DataFrame(fund)


def funding_features(fund, panel, sym):
    f = fund[sym]
    out = pd.DataFrame(index=f.index)
    out["funding"] = f
    out["funding_z"] = (f - f.rolling(30).mean()) / f.rolling(30).std()
    out["funding_cum_7d"] = f.rolling(21).sum()
    out["funding_cum_14d"] = f.rolling(42).sum()
    out["xs_fund_rank"] = panel.rank(axis=1, pct=True)[sym]
    out["xs_fund_z"] = (panel.sub(panel.mean(axis=1), axis=0)).div(panel.std(axis=1), axis=0)[sym]
    return out


def build(bars, fund, panel, H, use_f):
    frames = []
    for sym, df in bars.items():
        F = pd.DataFrame(T.add_features(df)[FEATURE_COLS].to_numpy(float),
                         index=df.index, columns=FEATURE_COLS)
        if use_f:
            ff = funding_features(fund, panel, sym)
            F = F.join(ff.reindex(df.index.union(ff.index)).sort_index().ffill().reindex(df.index))
        C = df["close"].to_numpy(float)
        fwd = np.full(len(df), np.nan)
        fwd[:-H] = C[H:] / C[:-H] - 1.0
        cols = FEATURE_COLS + (FCOLS if use_f else [])
        F = F[cols].assign(sym=sym, fwd=fwd)
        frames.append(F[np.isfinite(F[cols].to_numpy(float)).all(axis=1) & np.isfinite(F["fwd"])])
    return pd.concat(frames), (FEATURE_COLS + (FCOLS if use_f else []))


def market_neutral_ls(P, cols, H, K=3, cost_bps=50, split=0.5):
    """Non-overlapping, market-neutral long-short: every H bars, rank symbols by
    model prob, long top-K / short bottom-K. Beta cancels, so a positive result
    is a timing edge."""
    times = np.sort(P.index.unique())
    cut_t = times[int(len(times) * split)]
    tr = P[P.index < cut_t]
    m = T._fit(np.clip(tr[cols].to_numpy(float), -8, 8), (tr["fwd"] > 0).astype(int))
    te = P[P.index >= cut_t].copy()
    te["p"] = m.predict_proba(np.clip(te[cols].to_numpy(float), -8, 8))[:, 1]
    picks = np.sort(te.index.unique())[::H]
    gross = []
    for t in picks:
        g = te.loc[[t]].sort_values("p")
        if len(g) >= 2 * K:
            gross.append(g["fwd"].iloc[-K:].mean() - g["fwd"].iloc[:K].mean())
    gross = np.array(gross)
    net = gross - 2 * cost_bps / 1e4
    return gross.mean() * 1e4, net.mean() * 1e4, len(net), net.mean() / (net.std(ddof=1) / np.sqrt(len(net)))


def main():
    bars = T.load_bars("bt_cache", None)
    try:
        fund, panel = _load_panel(bars)
    except FileNotFoundError:
        print("Funding data missing. Fetching...")
        fetch_funding([s.split("/")[0] for s in bars])
        fund, panel = _load_panel(bars)
    print(f"{'H':>4s} {'cfg':>6s} {'n':>4s} {'gross_bps':>10s} {'net_bps':>9s} {'t':>7s}")
    for H in (288, 672):
        for uf in (False, True):
            P, cols = build(bars, fund, panel, H, uf)
            g, n, cnt, t = market_neutral_ls(P, cols, H)
            print(f"{H:>4d} {'+fund' if uf else 'base':>6s} {cnt:>4d} {g:+10.1f} {n:+9.1f} {t:+7.2f}")


if __name__ == "__main__":
    main()
