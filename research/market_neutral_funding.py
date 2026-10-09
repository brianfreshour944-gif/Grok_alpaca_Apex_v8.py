"""Market-neutral, multi-day funding/basis strategy -- a genuinely different
"shape" from the repo's 2-hour, long-only, single-symbol OHLCV bot.

Motivation (see research/RESULTS.md): every lever applied to the *current* bot
(better inputs, lower costs, roll retrain, config tuning) leaves it net-negative
because its per-trade edge (~+5 bps) is far below a ~50 bps taker round trip.
The only candidate with a pulse was a 7-day, market-neutral, funding-driven
long-short (+90 bps gross, beta-free). This module builds that strategy
properly:

  * **Multi-day horizon** (H hours), not 2h -- funding is a slow signal.
  * **Market-neutral long-short** across the universe, so the P&L is a *timing*
    edge with crypto beta cancelled (verified with a circular-shift null).
  * **Cost-aware**: P&L is reported gross and net at taker and maker fees.
  * **Honest OOS**: expanding-window walk-forward; every number is out of sample.

Data: Binance Vision monthly archives (data.binance.vision), which are public
and return full history:
  * perp klines:  futures/um/monthly/klines/<SYM>USDT/1h/...
  * funding rate: futures/um/monthly/fundingRate/<SYM>USDT/...
  * spot klines:  spot/monthly/klines/<SYM>USDT/1h/...   (for the perp-spot basis)
Everything is cached under bt_cache/research/ (gitignored).

Run: python research/market_neutral_funding.py            # full report
     python research/market_neutral_funding.py --quick    # smaller universe
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CACHE = os.environ.get("MN_CACHE", os.path.join(ROOT, "bt_cache", "research"))

# Liquid, long-lived Binance USDT perps (broad universe -> cross-sectional width).
DEFAULT_SYMBOLS = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX",
    "LINK", "DOT", "LTC", "BCH", "MATIC", "TRX", "ATOM", "UNI",
]
# 2025-01 .. 2026-09 (verified available on Binance Vision).
DEFAULT_MONTHS = [f"{y}-{m:02d}" for y in (2025, 2026) for m in range(1, 13)]
DEFAULT_MONTHS = [m for m in DEFAULT_MONTHS if "2025-01" <= m <= "2026-09"]

KLINES_HEADER = ["open_time", "open", "high", "low", "close", "volume",
                 "close_time", "quote_volume", "count", "taker_buy_volume",
                 "taker_buy_quote_volume", "ignore"]


# ─────────────────────────────────────────────────────────────────────────────
#  Fetch + cache
# ─────────────────────────────────────────────────────────────────────────────
def _vision_url(kind: str, sym: str, mon: str, interval: str = "1h") -> str:
    if kind == "funding":
        return (f"https://data.binance.vision/data/futures/um/monthly/fundingRate/"
                f"{sym}USDT/{sym}USDT-fundingRate-{mon}.zip")
    base = "futures/um" if kind == "perp" else "spot"
    return (f"https://data.binance.vision/data/{base}/monthly/klines/"
            f"{sym}USDT/{interval}/{sym}USDT-{interval}-{mon}.zip")


def _read_zip_csv(url: str) -> pd.DataFrame | None:
    """Read a Vision archive CSV. Binance occasionally ships a header row and
    occasionally does not, so parse header-less and drop a header row if present."""
    try:
        raw = urllib.request.urlopen(url, timeout=30).read()
        zf = zipfile.ZipFile(io.BytesIO(raw))
        d = pd.read_csv(zf.open(zf.namelist()[0]), header=None)
        if len(d) and isinstance(d.iloc[0, 0], str):
            d = d.iloc[1:].reset_index(drop=True)
        return d
    except (urllib.error.URLError, zipfile.BadZipFile, OSError, ValueError, IndexError):
        return None


def _to_datetime(col) -> pd.DatetimeIndex:
    """Vision mixes epoch units: perp ms (13 digits), spot us (16 digits)."""
    v = pd.to_numeric(col)
    unit = "us" if v.dropna().abs().median() > 1e14 else "ms"
    return pd.to_datetime(v.astype("int64"), unit=unit, utc=True).dt.tz_localize(None)


def _fetch_series(kind: str, sym: str, months, interval="1h") -> pd.DataFrame:
    """Download and concatenate one dataset for one symbol; cache to parquet-free CSV."""
    cache_fp = os.path.join(CACHE, f"{kind}_{sym}_{interval}.csv")
    if os.path.exists(cache_fp):
        df = pd.read_csv(cache_fp)
        if len(df):
            return df
    os.makedirs(CACHE, exist_ok=True)
    frames = []
    for mon in months:
        d = _read_zip_csv(_vision_url(kind, sym, mon, interval))
        if d is None or not len(d):
            continue
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(cache_fp, index=False)
    return df


@dataclass
class Panel:
    close: pd.DataFrame        # perp close, hourly, wide (index=ts, cols=sym)
    funding: pd.DataFrame      # 8h funding settlements, wide, reindexed hourly (ffill)
    basis: pd.DataFrame        # perp/spot - 1, hourly, wide
    ret: pd.DataFrame          # hourly log returns of perp close


def load_panel(symbols, months, use_basis=True, verbose=True) -> Panel:
    close, funding, basis = {}, {}, {}
    for sym in symbols:
        k = _fetch_series("perp", sym, months)
        if not len(k):
            if verbose:
                print(f"  [data] {sym}: no perp klines -- skipped")
            continue
        k = k.iloc[:, :len(KLINES_HEADER)]
        k.columns = KLINES_HEADER
        ts = _to_datetime(k["open_time"])
        s = pd.Series(k["close"].astype(float).to_numpy(), index=ts).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        close[sym] = s

        f = _fetch_series("funding", sym, months)
        if len(f):
            f.columns = [str(c) for c in f.columns][:3]
            fts = _to_datetime(f.iloc[:, 0])
            fr = pd.Series(f.iloc[:, 2].astype(float).to_numpy(), index=fts).sort_index()
            fr = fr[~fr.index.duplicated(keep="last")]
            funding[sym] = fr

        if use_basis:
            sp = _fetch_series("spot", sym, months)
            if len(sp):
                sp = sp.iloc[:, :len(KLINES_HEADER)]
                sp.columns = KLINES_HEADER
                sts = _to_datetime(sp["open_time"])
                ss = pd.Series(sp["close"].astype(float).to_numpy(), index=sts).sort_index()
                ss = ss[~ss.index.duplicated(keep="last")]
                basis[sym] = ss

    C = pd.DataFrame(close).sort_index()
    C = C.loc[:, C.notna().sum() > 1000]
    idx = C.index

    F = pd.DataFrame(funding).sort_index()
    F = F.reindex(idx, method="ffill")  # hourly view of the 8h funding rate

    B = pd.DataFrame()
    if use_basis and basis:
        S = pd.DataFrame(basis).reindex(idx, method="ffill")
        B = (C / S - 1.0).replace([np.inf, -np.inf], np.nan)

    R = np.log(C).diff()
    if verbose:
        print(f"  [data] panel: {C.shape[1]} symbols x {C.shape[0]:,} hours "
              f"({idx.min()} -> {idx.max()})")
    return Panel(close=C, funding=F, basis=B, ret=R)


# ─────────────────────────────────────────────────────────────────────────────
#  Features (cross-sectional + time-series), all lag-safe at time t
# ─────────────────────────────────────────────────────────────────────────────
def xs_rank(W: pd.DataFrame) -> pd.DataFrame:
    return W.rank(axis=1, pct=True)


def xs_z(W: pd.DataFrame) -> pd.DataFrame:
    return W.sub(W.mean(axis=1), axis=0).div(W.std(axis=1).replace(0, np.nan), axis=0)


def build_features(P: Panel, use_basis: bool) -> dict[str, pd.DataFrame]:
    """Return a dict of aligned feature panels (each = one feature)."""
    F = P.funding
    R = P.ret
    feats: dict[str, pd.DataFrame] = {}

    feats["funding"] = F
    feats["funding_z"] = (F - F.rolling(90, min_periods=30).mean()) / F.rolling(90, min_periods=30).std()
    feats["funding_cum_3d"] = F.rolling(72).sum()
    feats["funding_cum_7d"] = F.rolling(168).sum()
    feats["xs_funding"] = xs_rank(F)
    feats["xs_funding_z"] = xs_z(F)

    # Slow price factors (multi-day momentum / trend), a second information source.
    feats["ret_3d"] = np.log(P.close / P.close.shift(72))
    feats["ret_7d"] = np.log(P.close / P.close.shift(168))
    feats["ret_14d"] = np.log(P.close / P.close.shift(336))
    feats["vol_7d"] = R.rolling(168).std()
    feats["xs_ret_7d"] = xs_rank(feats["ret_7d"])
    feats["xs_vol_7d"] = xs_rank(feats["vol_7d"])

    if use_basis and len(P.basis):
        B = P.basis
        feats["basis"] = B
        feats["basis_z"] = (B - B.rolling(90, min_periods=30).mean()) / B.rolling(90, min_periods=30).std()
        feats["xs_basis"] = xs_rank(B)

    return feats


# ─────────────────────────────────────────────────────────────────────────────
#  Walk-forward model + evaluation
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Cov:
    """Cost model in bps, per fill. This is a Binance USDT-M perpetual strategy, so
    the venue's own fees apply, NOT Alpaca's crypto tiers: Binance USDⓈ-M
    maker/taker are 0.02%/0.05% (2/5 bps) at VIP0. `tier1` keeps the old Alpaca
    taker (25 bps) as a conservative reference column."""
    taker_bps: float = 5.0
    maker_bps: float = 2.0
    tier1_bps: float = 25.0


FEATURE_ORDER = [
    "funding", "funding_z", "funding_cum_3d", "funding_cum_7d", "xs_funding",
    "xs_funding_z", "ret_3d", "ret_7d", "ret_14d", "vol_7d", "xs_ret_7d", "xs_vol_7d",
    "basis", "basis_z", "xs_basis",
]


def _stack(feats: dict[str, pd.DataFrame], cols: list[str], fwd: np.ndarray):
    """Long-form (feature rows, target) aligned across symbols."""
    syms = list(next(iter(feats.values())).columns)
    X = []
    for sym in syms:
        block = np.column_stack([feats[c][sym].to_numpy(float) for c in cols])
        X.append(block)
    X = np.concatenate(X, axis=0)
    y = fwd.reshape(-1)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X, y, ok


def _fit(X, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(
        StandardScaler(),
        HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_depth=3,
                                       l2_regularization=1.0, early_stopping=True,
                                       validation_fraction=0.15, random_state=0),
    )
    m.fit(np.clip(X, -8, 8), y)
    return m


def forward_returns(C: pd.DataFrame, H: int) -> pd.DataFrame:
    """H-bar forward simple return, aligned at t (no look-ahead in features)."""
    return C.shift(-H) / C - 1.0


def long_short_book(pred: pd.DataFrame, fwd: pd.DataFrame, H: int, K: int,
                    long_only: bool = False, funding: pd.DataFrame | None = None) -> pd.DataFrame:
    """Non-overlapping (every H bars) long-top-K / short-bottom-K book.

    `pred` is a per-symbol score panel (higher = more attractive to hold long).
    Returns per-period: price long-short return, the *funding cashflow* of the
    book, turnover, and names traded.

    Funding sign (the part the earlier probe omitted): when funding > 0, longs
    PAY shorts, so a short leg EARNS the funding rate and a long leg PAYS it.
    A perp long-short's true P&L is the price leg PLUS this funding leg; ignoring
    it makes a funding-carry strategy look like pure price betting, which is
    exactly the mistake that hid the carry edge.
    """
    rows = []
    prev = None
    times = list(pred.index)[::H]
    for t in times:
        p = pred.loc[t].dropna()
        r = fwd.loc[t]
        valid = p.index.intersection(r.dropna().index)
        if len(valid) < 2 * K:
            continue
        p = p[valid]
        order = p.sort_values()
        shorts = list(order.index[:K])
        longs = list(order.index[-K:])
        gl = r[longs].mean()
        gs = r[shorts].mean() if not long_only else 0.0
        price = gl - gs
        names = set(longs) | (set() if long_only else set(shorts))
        turn = 1.0 if prev is None else len(names - prev) / len(names)
        prev = names

        # Funding cashflow over the holding window (8h settlements).
        f_cash = np.nan
        if funding is not None and t in funding.index:
            loc = funding.index.get_indexer([t])[0]
            window = funding.iloc[loc:loc + H]
            if len(window):
                short_sum = window[shorts].sum().mean() if not long_only else 0.0
                long_sum = window[longs].sum().mean()
                f_cash = (short_sum - long_sum) / 8.0  # hourly view -> 8h settlements

        rows.append({"t": t, "price": price, "gross": price + (0.0 if np.isnan(f_cash) else f_cash),
                     "funding_cash": f_cash, "turnover": turn,
                     "n_long": len(longs), "n_short": 0 if long_only else len(shorts)})
    return pd.DataFrame(rows).set_index("t") if rows else pd.DataFrame()


def summarize_periods(book: pd.DataFrame, cov: Cov, H: int) -> dict:
    """Convert a per-period book into annualised net metrics at both fee tiers.

    Cost per period = turnover * legs * fee, where legs = (#longs + #shorts).
    """
    g = book["gross"].to_numpy(float)
    turn = book["turnover"].to_numpy(float)
    legs = book["n_long"].to_numpy(float) + book["n_short"].to_numpy(float)
    n = len(g)
    per_year = 8760.0 / H

    def _net(fee_bps):
        cost = turn * legs * fee_bps / 1e4
        return g - cost

    gross_bps = g.mean() * 1e4
    gross_sharpe = (g.mean() / g.std(ddof=1)) * np.sqrt(per_year) if n > 2 and g.std() else np.nan
    out = {"n": n, "gross_bps": gross_bps, "gross_sharpe": gross_sharpe,
           "turnover": float(turn.mean())}
    if "price" in book:
        out["price_bps"] = book["price"].mean() * 1e4
    if "funding_cash" in book:
        out["funding_bps"] = float(np.nanmean(book["funding_cash"]) * 1e4)
    for label, fee in (("taker", cov.taker_bps), ("maker", cov.maker_bps), ("tier1", cov.tier1_bps)):
        net = _net(fee)
        sd = net.std(ddof=1)
        out[f"net_bps_{label}"] = net.mean() * 1e4
        out[f"net_ret_{label}"] = float(np.prod(1 + net) - 1)
        out[f"net_sharpe_{label}"] = (net.mean() / sd) * np.sqrt(per_year) if n > 2 and sd else np.nan
        out[f"t_{label}"] = net.mean() / (sd / np.sqrt(n)) if n > 2 and sd else np.nan
    return out


def walk_forward_signal(P: Panel, feats: dict[str, pd.DataFrame], cols: list[str],
                        H: int, folds: int = 5, min_train_frac: float = 0.4):
    """Expanding-window walk-forward; returns OOS predictions panel."""
    C = P.close
    fwd = forward_returns(C, H)
    syms = C.columns
    X, y, ok = _stack(feats, cols, fwd.to_numpy(float))
    y = (y > 0).astype(int)
    ts = np.repeat(next(iter(feats.values())).index.to_numpy(), len(syms))
    sym_rep = np.tile(np.array(syms, dtype=object), len(next(iter(feats.values()))))
    X, y, ts, sym_rep = X[ok], y[ok], ts[ok], sym_rep[ok]
    order = np.argsort(ts, kind="stable")
    X, y, ts, sym_rep = X[order], y[order], ts[order], sym_rep[order]

    n = len(y)
    start = int(n * min_train_frac)
    edges = np.linspace(start, n, folds + 1).astype(int)
    preds = {}
    for k in range(folds):
        tr_ts = ts[:edges[k]]
        te_ts = ts[edges[k]:edges[k + 1]]
        if len(te_ts) == 0 or len(tr_ts) == 0:
            continue
        m = _fit(X[:edges[k]], y[:edges[k]])
        p = m.predict_proba(X[edges[k]:edges[k + 1]])[:, 1]
        df = pd.DataFrame({"ts": te_ts, "sym": sym_rep[edges[k]:edges[k + 1]], "p": p})
        preds[k] = df
    if not preds:
        return pd.DataFrame()
    allp = pd.concat(preds.values(), ignore_index=True)
    panel = allp.pivot_table(index="ts", columns="sym", values="p", aggfunc="last")
    return panel


def _xs_scale(W: pd.DataFrame) -> pd.DataFrame:
    return W.sub(W.mean(axis=1), axis=0).div(W.std(axis=1).replace(0, np.nan), axis=0)


def signal_panel(mode: str, pred_ml: pd.DataFrame, feats: dict[str, pd.DataFrame],
                 P: Panel | None = None) -> pd.DataFrame:
    """A per-symbol score panel (higher = better long). `ml` uses the walk-forward
    model; the others are model-free and interpretable."""
    if mode == "ml":
        return pred_ml
    if mode == "carry":            # long low funding, short high funding (funding-z scored)
        return -_xs_scale(feats["funding_cum_7d"])
    if mode == "xs_carry":         # cross-sectional carry
        return -feats["xs_funding_z"]
    if mode == "rev":              # 7-day reversal
        return -_xs_scale(feats["ret_7d"])
    if mode == "mom":              # 7-day momentum
        return _xs_scale(feats["ret_7d"])
    if mode == "carry_rev":        # carry blended with reversal
        return -_xs_scale(feats["funding_cum_7d"]) - _xs_scale(feats["ret_7d"])
    if mode == "carry_mixed":      # carry from several windows + reversal
        c = (-_xs_scale(feats["funding_cum_3d"]) - _xs_scale(feats["funding_cum_7d"])
             - feats["xs_funding_z"] * 0.5)
        return c - _xs_scale(feats["ret_7d"])
    if mode == "carry_cond":       # carry, but only when funding is dispersed enough to pay
        disp = feats["funding_cum_7d"].std(axis=1)
        gate = disp > disp.rolling(720, min_periods=120).median()
        return (-_xs_scale(feats["funding_cum_7d"])).where(gate, np.nan)
    if mode == "carry_disp":       # carry scaled continuously by dispersion z-score
        disp = feats["funding_cum_7d"].std(axis=1)
        scale = (disp - disp.rolling(720, min_periods=120).mean()) / disp.rolling(720, min_periods=120).std()
        return (-_xs_scale(feats["funding_cum_7d"])).mul(scale.clip(lower=0), axis=0)
    if mode == "carry_bull":       # carry, gated on broad-market uptrend (BTC > 30d MA)
        btc = None
        for c in ("BTC", "BTCUSDT"):
            if c in feats["ret_7d"].columns:
                btc = c
                break
        if btc is None:
            return -_xs_scale(feats["funding_cum_7d"])
        px = P.close[btc]
        gate = (px > px.rolling(720, min_periods=168).mean()).reindex(feats["ret_7d"].index).ffill()
        return (-_xs_scale(feats["funding_cum_7d"])).where(gate, np.nan)
    raise ValueError(mode)


def null_alpha(pred: pd.DataFrame, fwd: pd.DataFrame, H: int, K: int,
               trials: int = 200, seed: int = 0, long_only: bool = False,
               funding: pd.DataFrame | None = None) -> float:
    """Circular-shift null: shift the return panel vs the prediction panel by a
    random >=3-day offset and recompute the book. p = fraction of nulls >= real."""
    real = long_short_book(pred, fwd, H, K, long_only=long_only, funding=funding)["gross"].mean()
    rng = np.random.default_rng(seed)
    n = len(fwd)
    shift_min = max(1, 72 // H)
    vals = []
    fwd_arr = fwd.to_numpy()
    for _ in range(trials):
        sh = int(rng.integers(shift_min, max(shift_min + 1, n - shift_min)))
        shifted = pd.DataFrame(np.roll(fwd_arr, sh, axis=0), index=fwd.index, columns=fwd.columns)
        b = long_short_book(pred, shifted, H, K, long_only=long_only, funding=funding)
        if len(b):
            vals.append(b["gross"].mean())
    vals = np.array(vals)
    return float((vals >= real).mean()) if len(vals) else np.nan


# ─────────────────────────────────────────────────────────────────────────────
#  Report
# ─────────────────────────────────────────────────────────────────────────────
def run(symbols, months, H_list, K=3, use_basis=True, long_only=False,
        modes=("ml", "carry", "xs_carry", "rev", "carry_rev")):
    P = load_panel(symbols, months, use_basis=use_basis)
    feats = build_features(P, use_basis)
    cols = [c for c in FEATURE_ORDER if c in feats]
    cov = Cov()
    print(f"\n  features ({len(cols)}): {', '.join(cols)}")
    print(f"  book: {'long-only top-' + str(K) if long_only else 'long top-' + str(K) + ' / short bottom-' + str(K)}"
          f" | taker {cov.taker_bps} bps, maker {cov.maker_bps} bps per fill")
    print("  P&L includes the funding cashflow of the book (longs pay shorts when funding > 0)\n")
    print(f"  {'mode':>11} {'H(h)':>5} {'n':>4} {'turn':>5} {'gross':>8} {'price':>8} {'fund':>7} "
          f"{'gShp':>6} {'netT':>8} {'tT':>6} {'netM':>8} {'tM':>6} {'net25':>8} {'p_null':>7}")
    results = {}
    for H in H_list:
        pred_ml = walk_forward_signal(P, feats, cols, H)
        fwd = forward_returns(P.close, H)
        for mode in modes:
            pred = signal_panel(mode, pred_ml, feats, P)
            pred = pred.reindex(fwd.index).dropna(how="all")
            book = long_short_book(pred, fwd, H, K, long_only=long_only, funding=P.funding)
            if book.empty:
                continue
            s = summarize_periods(book, cov, H)
            fwd_oos = fwd.loc[pred.index]
            p_null = null_alpha(pred, fwd_oos, H, K, trials=120, long_only=long_only)
            results[(mode, H)] = s
            print(f"  {mode:>11} {H:>5} {s['n']:>4} {s['turnover']:>5.2f} {s['gross_bps']:>+8.1f} "
                  f"{s.get('price_bps', float('nan')):>+8.1f} {s.get('funding_bps', float('nan')):>+7.1f} "
                  f"{s['gross_sharpe']:>6.2f} {s['net_bps_taker']:>+8.1f} {s['t_taker']:>+6.2f} "
                  f"{s['net_bps_maker']:>+8.1f} {s['t_maker']:>+6.2f} {s['net_bps_tier1']:>+8.1f} {p_null:>7.2f}")
        print()
    return results, P, feats, cols


def robustness(P: Panel, feats: dict[str, pd.DataFrame], cols: list[str],
               mode: str = "carry", H: int = 168, K: int = 3, chunks: int = 3):
    """Sub-period stability + K sensitivity for the chosen strategy."""
    cov = Cov()
    fwd = forward_returns(P.close, H)
    pred = signal_panel(mode, walk_forward_signal(P, feats, cols, H), feats, P)
    pred = pred.reindex(fwd.index)
    print(f"\n  === robustness: mode={mode} H={H}h ===")
    print(f"  {'K':>3} {'n':>4} {'gross':>8} {'netT':>8} {'tT':>6} {'netM':>8} {'net25':>8}")
    for k in (2, 3, 4, 5):
        book = long_short_book(pred, fwd, H, k, funding=P.funding)
        if book.empty:
            continue
        s = summarize_periods(book, cov, H)
        print(f"  {k:>3} {s['n']:>4} {s['gross_bps']:>+8.1f} {s['net_bps_taker']:>+8.1f} "
              f"{s['t_taker']:>+6.2f} {s['net_bps_maker']:>+8.1f} {s['net_bps_tier1']:>+8.1f}")
    book = long_short_book(pred, fwd, H, K, funding=P.funding)
    if book.empty:
        return
    t = book.index
    edges = np.linspace(0, len(t), chunks + 1).astype(int)
    print(f"\n  sub-periods (K={K}):")
    print(f"  {'chunk':>5} {'from':>10} {'to':>10} {'n':>4} {'gross':>8} {'netT':>8} {'netM':>8}")
    for c in range(chunks):
        b = book.iloc[edges[c]:edges[c + 1]]
        if b.empty:
            continue
        s = summarize_periods(b, cov, H)
        print(f"  {c:>5} {str(t[edges[c]])[:10]:>10} {str(t[edges[c + 1] - 1])[:10]:>10} "
              f"{s['n']:>4} {s['gross_bps']:>+8.1f} {s['net_bps_taker']:>+8.1f} {s['net_bps_maker']:>+8.1f}")


def verify_deployable(P: Panel):
    """Cross-check: the ship-ready module (funding_carry.py) must reproduce the
    research harness's carry book. Guards against research/ship divergence."""
    sys.path.insert(0, ROOT)
    import funding_carry as fc
    fwd = forward_returns(P.close, fc.HOLD_HOURS)
    score = fc.funding_carry_score(P.funding, lookback=fc.LOOKBACK_HOURS)
    W = fc.target_weights(score, K=fc.TOP_K)
    m = fc.book_metrics(P.close, W, H=fc.HOLD_HOURS)
    book = long_short_book(score, fwd, fc.HOLD_HOURS, fc.TOP_K, funding=P.funding)
    f_cash = book["funding_cash"].mean() * 1e4
    print("\n  === deployable parity (funding_carry.py vs harness) ===")
    print(f"  price leg:  module {m['gross_bps']:+.2f} bps | harness {book['price'].mean()*1e4:+.2f} bps")
    print(f"  +funding:   module {m['gross_bps']+f_cash:+.1f} gross | harness {book['gross'].mean()*1e4:+.1f} gross")
    per_year = 8760 / fc.HOLD_HOURS
    print(f"  annualised (taker {fc.TAKER_BPS:.0f}bps): {((m['net_bps_taker']+f_cash)/1e4)*per_year*100:+.1f}%")
    print(f"  annualised (maker {fc.MAKER_BPS:.0f}bps): {((m['net_bps_maker']+f_cash)/1e4)*per_year*100:+.1f}%")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller universe")
    ap.add_argument("--months", default=None, help="comma list YYYY-MM")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--no-basis", action="store_true")
    ap.add_argument("--H", default="72,168,336", help="horizons in hours")
    ap.add_argument("--modes", default="ml,carry,xs_carry,rev,carry_rev")
    ap.add_argument("--robust", action="store_true", help="sub-period + K stability")
    a = ap.parse_args(argv)

    symbols = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX"] if a.quick else DEFAULT_SYMBOLS
    months = [m.strip() for m in a.months.split(",")] if a.months else DEFAULT_MONTHS
    H_list = [int(x) for x in a.H.split(",")]
    modes = tuple(m.strip() for m in a.modes.split(","))
    print(f"Market-neutral funding strategy | {len(symbols)} symbols | {len(months)} months "
          f"({months[0]}..{months[-1]})")
    _results, P, feats, cols = run(symbols, months, H_list, K=a.K, use_basis=not a.no_basis,
                                   long_only=a.long_only, modes=modes)
    if a.robust:
        for H in H_list:
            robustness(P, feats, cols, mode="carry", H=H, K=a.K)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
