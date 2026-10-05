#!/usr/bin/env python3
"""
backtest_apex.py -- live-parity backtest for Grok_Alpaca_Apex (main_bot.py).

WHAT THIS IS
    A bar-close-resolution portfolio simulation that re-uses the repo's OWN
    decision code wherever possible (add_features, the champion model +
    scaler, compute_regime_and_trend, evaluate_exit, get_regime_params,
    calculate_adjusted_risk) so the backtest cannot silently drift away from
    the bot.  Everything the bot does that cannot be reproduced offline is
    listed under "NOT MODELLED" in the generated report.

PIPELINE
    Stage 1  signals    : per (symbol, 15m bar) -> model signal, ATR%, regime,
                          EMA trend.  Features are computed on the same
                          64-bar window the live bot fetches (see _rpz_windows
                          for why this needs special care).  Cached on disk.
    Stage 2  simulation : cycle-by-cycle replay of main_bot.run_trading_mode()
                          (universe rotation, exits, entries, Kelly sizing,
                          20% exposure cap, forced sells, swaps, kill-switch,
                          cooldowns, limit-order fills, Alpaca fees).
    Stage 3  analytics  : performance, trade breakdowns, benchmarks, signal
                          IC diagnostics, circular-shift null test, cost and
                          threshold sensitivity, in-sample / out-of-sample.

TIMING MODEL (the part that matters most)
    main_bot.py only ever sees the last COMPLETED 15m bar (data_feeds.py drops
    the in-progress bar), so every decision is made on a bar CLOSE.  The order
    it sends is a marketable limit (buy: close*1.001, sell: close*0.999) that
    reaches the market ~seconds after that close, i.e. at roughly the NEXT
    bar's open.  The simulator therefore decides at bar close T and fills at
    the open of the bar that starts at T.  Stops are evaluated on closes only
    (the bot has no resting stop orders), exactly like production.

USAGE
    # real data (Alpaca crypto history needs no API key):
    python backtest_apex.py --source alpaca --start 2026-01-01 --end 2026-09-30
    # your own 15m CSVs (one file per symbol, e.g. BTC_USD.csv):
    python backtest_apex.py --source csv --csv-dir historical_data/
    # plumbing smoke test only -- synthetic data proves NOTHING about edge:
    python backtest_apex.py --source synthetic --start 2026-01-01 --end 2026-03-01
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# ── Repo wiring ───────────────────────────────────────────────────────────────
# config.py demands Alpaca keys at import time (it builds API clients, no
# network call).  A backtest never trades, so dummy keys are fine.  LOG_LEVEL
# is forced to WARNING so the repo's per-cycle INFO logging doesn't flood the
# console, and DATABASE_URL is removed so nothing here can ever write to a DB.
REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))
os.environ.setdefault("APCA_API_KEY_ID", "backtest-offline")
os.environ.setdefault("APCA_API_SECRET_KEY", "backtest-offline")
os.environ["LOG_LEVEL"] = "WARNING"
os.environ.pop("DATABASE_URL", None)

import config as cfg  # noqa: E402
from config import get_regime_params  # noqa: E402
from exit_logic import evaluate_exit  # noqa: E402
from feature_engineering import FEATURE_COLS, add_features  # noqa: E402
from money import qty as money_qty  # noqa: E402
from regime import calculate_adjusted_risk, compute_regime_and_trend  # noqa: E402

try:  # the repo's own sizing function; portfolio.py pulls in heavy deps
    from portfolio import calculate_kelly_multiplier as _repo_kelly
except Exception:  # pragma: no cover - environment dependent
    _repo_kelly = None

STAGE1_VERSION = "1"
BAR_SECONDS = 900          # 15-minute bars
WIN = 64                   # bars the live bot fetches (data_feeds.py limit=64)
SEQ = cfg.SEQUENCE_LEN     # 32
RPZ_IDX = FEATURE_COLS.index("range_position_z")
REGIME_NAMES = ("quiet", "normal", "wild")
REGIME_CODE = {n: i for i, n in enumerate(REGIME_NAMES)}


# ══════════════════════════════════════════════════════════════════════════════
# 0.  Small shared helpers
# ══════════════════════════════════════════════════════════════════════════════
def kelly_multiplier(signal_prob: float, profit_target_pct: float, stop_loss_pct: float) -> float:
    """Half-Kelly multiplier.  Uses portfolio.calculate_kelly_multiplier when
    importable; otherwise a verbatim copy (tests assert the two agree)."""
    if _repo_kelly is not None:
        return _repo_kelly(signal_prob, profit_target_pct, stop_loss_pct)
    return _kelly_vendored(signal_prob, profit_target_pct, stop_loss_pct)


def _kelly_vendored(signal_prob, profit_target_pct, stop_loss_pct) -> float:
    if stop_loss_pct <= 0 or profit_target_pct <= 0:
        return 1.0
    r = profit_target_pct / stop_loss_pct
    k = signal_prob - ((1.0 - signal_prob) / r)
    if k <= 0:
        return 0.5
    return max(0.5, min(0.5 + k * 15.0, 3.0))


def sanitize_price(price: float) -> float:
    """orders._sanitize_price: floor to a magnitude-dependent tick (this
    matters for sub-$1 coins, where the tick is a visible fraction of 0.1%)."""
    if price >= 1.0:
        step = 0.01
    elif price >= 0.01:
        step = 0.0001
    elif price >= 0.0001:
        step = 0.000001
    else:
        step = 0.00000001
    return math.floor(price / step + 1e-9) * step


def to_sec(ts) -> int:
    return int(pd.Timestamp(ts).value // 10**9)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(repo: Path) -> dict:
    def _run(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=repo, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return ""
    return {
        "commit": _run("rev-parse", "HEAD"),
        "dirty": bool(_run("status", "--porcelain", "--untracked-files=no")),
        "model_last_commit_date": _run("log", "-1", "--format=%cI", "--", cfg.MODEL_PATH),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Data layer
# ══════════════════════════════════════════════════════════════════════════════
BAR_COLS = ["open", "high", "low", "close", "volume", "vwap", "trade_count"]


def clean_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror data_feeds.get_clean_ohlcv_dataframe's cleaning: tz-naive UTC
    index, vwap fallback to close, drop non-positive closes.  Also sorts and
    de-duplicates, which the live API does server-side."""
    df = df.copy()
    for c in BAR_COLS:
        if c not in df.columns:
            df[c] = 0.0 if c == "trade_count" else np.nan
    df = df[BAR_COLS].astype(float)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.index = pd.DatetimeIndex(df.index).astype("datetime64[ns]")
    df.index.name = "timestamp"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df["vwap"] = df["vwap"].where(df["vwap"] > 0, df["close"])
    df = df[df["close"] > 0]
    return df


def _sym_file_stems(symbol: str) -> List[str]:
    base, quote = symbol.split("/")
    return [f"{base}_{quote}", f"{base}{quote}", f"{base}-{quote}", f"{base}_{quote}_15m", f"{base}{quote}_15m"]


def load_csv_dir(csv_dir: Path, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        path = None
        for stem in _sym_file_stems(sym):
            for ext in (".csv", ".csv.gz"):
                cand = csv_dir / f"{stem}{ext}"
                if cand.exists():
                    path = cand
                    break
            if path:
                break
        if path is None:
            print(f"  [data] {sym}: no CSV found in {csv_dir} (tried {_sym_file_stems(sym)}) -- skipped")
            continue
        raw = pd.read_csv(path)
        raw.columns = [str(c).strip().lower() for c in raw.columns]
        tcol = next((c for c in ("timestamp", "time", "datetime", "date", "t", "open_time") if c in raw.columns), raw.columns[0])
        if pd.api.types.is_numeric_dtype(raw[tcol]):
            unit = "ms" if raw[tcol].iloc[0] > 1e11 else "s"
            ts = pd.to_datetime(raw[tcol], unit=unit, utc=True)
        else:
            ts = pd.to_datetime(raw[tcol], utc=True)
        raw.index = ts
        raw = raw.rename(columns={"trades": "trade_count", "n": "trade_count"})
        out[sym] = clean_bars(raw)
    return out


def fetch_alpaca(symbols: List[str], start: pd.Timestamp, end: pd.Timestamp, cache_dir: Path) -> Dict[str, pd.DataFrame]:
    """Native 15-minute bars from Alpaca (same endpoint/venue the live bot uses).
    Crypto market data needs no API key.  Results are cached on disk."""
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    cache_dir.mkdir(parents=True, exist_ok=True)
    client = CryptoHistoricalDataClient()
    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        fp = cache_dir / f"{sym.replace('/', '_')}_15m_{start:%Y%m%d}_{end:%Y%m%d}.csv.gz"
        if fp.exists():
            df = pd.read_csv(fp, index_col=0, parse_dates=True)
            print(f"  [data] {sym}: {len(df):,} bars (cache)")
        else:
            req = CryptoBarsRequest(
                symbol_or_symbols=sym, timeframe=TimeFrame(15, TimeFrameUnit.Minute),
                start=start.to_pydatetime().replace(tzinfo=timezone.utc),
                end=end.to_pydatetime().replace(tzinfo=timezone.utc),
            )
            res = client.get_crypto_bars(req).df
            if res is None or len(res) == 0:
                print(f"  [data] {sym}: no bars returned -- skipped")
                continue
            if isinstance(res.index, pd.MultiIndex):
                res = res.xs(sym, level="symbol")
            df = clean_bars(res)
            df.to_csv(fp)
            print(f"  [data] {sym}: {len(df):,} bars (downloaded)")
        out[sym] = clean_bars(df)
    return out


def make_synthetic(symbols: List[str], start, end, seed: int = 7, drift: float = 0.0,
                   gap_prob: Optional[List[float]] = None) -> Dict[str, pd.DataFrame]:
    """Regime-switching, jumpy, cross-correlated OHLCV with volume/vwap/trade
    counts.  ONLY for testing the machinery -- by construction there is no
    exploitable pattern here, so any performance number is meaningless."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="15min", inclusive="left")
    n, k = len(idx), len(symbols)
    mult_levels = np.array([0.6, 1.0, 2.2])
    st = np.zeros(n, dtype=int)
    for i in range(1, n):
        st[i] = rng.integers(0, 3) if rng.random() < 1 / 400 else st[i - 1]
    vm = mult_levels[st]
    mkt = rng.standard_normal(n) * 0.0015 * vm
    out = {}
    gap_prob = gap_prob or [0.0] * k
    for j, sym in enumerate(symbols):
        bvol = 0.0012 + 0.0025 * j / max(1, k - 1)
        idio = rng.standard_normal(n) * bvol * vm
        jumps = (rng.random(n) < 0.002) * rng.standard_normal(n) * 0.012
        r = 0.7 * mkt + idio + jumps + drift
        close = (100.0 * (j + 1)) * np.exp(np.cumsum(r))
        open_ = np.r_[close[0], close[:-1]]
        rng_hi = np.abs(rng.standard_normal(n)) * bvol * vm * 0.6
        rng_lo = np.abs(rng.standard_normal(n)) * bvol * vm * 0.6
        high = np.maximum(open_, close) * (1 + rng_hi)
        low = np.minimum(open_, close) * (1 - rng_lo)
        volume = rng.lognormal(5.0 + 0.1 * j, 0.6, n) * (1 + 30 * np.abs(r) / bvol / 10)
        vwap = low + (high - low) * rng.uniform(0.3, 0.7, n)
        tcount = np.maximum(1, rng.poisson(volume / 3.0)).astype(float)
        df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                           "volume": volume, "vwap": vwap, "trade_count": tcount}, index=idx)
        if gap_prob[j] > 0:
            df = df[rng.random(n) >= gap_prob[j]]
        out[sym] = clean_bars(df)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Stage 1 -- per-bar signal / regime / trend (live-parity, vectorised)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Stage1:
    signal: np.ndarray    # float64, NaN where the live bot could not produce one
    atr_pct: np.ndarray   # rounded to 2dp exactly like compute_regime_and_trend
    regime: np.ndarray    # int8: 0 quiet / 1 normal / 2 wild
    trend_up: np.ndarray  # bool
    valid: np.ndarray     # bool: the bot would have data AND a finite signal here
    # Price the bot would DECIDE on at this bar (== close unless the data window
    # is frozen, see compute_stage1_day_anchored) and whether that data is fresh.
    dec_close: Optional[np.ndarray] = None
    fresh: Optional[np.ndarray] = None

    def decision_price(self, close: np.ndarray) -> np.ndarray:
        return close if self.dec_close is None else self.dec_close

    def fresh_mask(self) -> np.ndarray:
        return self.valid if self.fresh is None else (self.valid & self.fresh)


def _rpz_windows(Hw: np.ndarray, Lw: np.ndarray, Cw: np.ndarray, seq: int = SEQ) -> np.ndarray:
    """range_position_z exactly as add_features() computes it on each 64-bar
    live window, for many windows at once.

    WHY THIS EXISTS: of the 11 model features, 10 depend on at most ~20 prior
    bars and are therefore IDENTICAL whether computed on full history or on the
    live 64-bar window (verified bit-for-bit in the test-suite).  range_position_z
    chains a 20-bar high/low range into a 20-bar z-score (39 bars deep), so the
    oldest ~7 rows of the model's 32-row input differ -- by up to ~0.8 z -- if
    computed on full history.  Calling add_features() per window would be exact
    but ~30ms each (~80 minutes for 10 symbols x 6 months), so instead the
    other 10 features come from one full-history call and only this feature is
    recomputed per window, using the same pandas rolling kernels (columns =
    windows) so results are identical, not approximately equal.
    """
    dH, dL, dC = pd.DataFrame(Hw.T), pd.DataFrame(Lw.T), pd.DataFrame(Cw.T)
    hi = dH.rolling(20, min_periods=2).max()
    lo = dL.rolling(20, min_periods=2).min()
    rng = (hi - lo).replace(0.0, np.nan)
    raw = ((dC - lo) / rng).replace([np.inf, -np.inf], 0.5).fillna(0.5)
    mu = raw.rolling(20, min_periods=2).mean()
    sig = raw.rolling(20, min_periods=2).std().replace(0.0, np.nan)
    z = ((raw - mu) / sig).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return z.to_numpy()[-seq:].T


def _san(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return np.where(np.isfinite(a), a, 0.0)


def compute_stage1(df: pd.DataFrame, infer: Callable[[np.ndarray], np.ndarray], batch: int = 2048) -> Stage1:
    """Signal/regime/trend for every bar that has a full live-style window.

    `infer` maps RAW (unscaled) feature windows (m, 32, 11) -> probabilities
    (m,).  Injecting it keeps this function testable without torch.
    """
    n = len(df)
    H, L, C = _san(df["high"]), _san(df["low"]), _san(df["close"])
    signal = np.full(n, np.nan)
    atr_pct_out = np.full(n, np.nan)
    regime = np.full(n, -1, dtype=np.int8)
    trend_up = np.zeros(n, dtype=bool)
    if n < WIN:
        return Stage1(signal, atr_pct_out, regime, trend_up, np.zeros(n, dtype=bool))

    ends = np.arange(WIN - 1, n)
    # --- model signal -------------------------------------------------------
    F = add_features(df)[FEATURE_COLS].to_numpy(dtype=float)
    swF = sliding_window_view(F, SEQ, axis=0)           # (n-SEQ+1, 11, SEQ)
    swH, swL, swC = (sliding_window_view(a, WIN) for a in (H, L, C))
    for lo_ in range(0, len(ends), batch):
        e = ends[lo_:lo_ + batch]
        X = swF[e - (SEQ - 1)].transpose(0, 2, 1).copy()   # (m, SEQ, 11)
        s = e - (WIN - 1)
        X[:, :, RPZ_IDX] = _rpz_windows(swH[s], swL[s], swC[s])
        p = np.asarray(infer(X), dtype=float)
        p[~np.isfinite(X).all(axis=(1, 2))] = 0.5          # predict_batch: non-finite -> 0.5
        signal[e] = p

    # --- ATR% / regime (regime.compute_regime_and_trend, vectorised) --------
    prev = np.r_[np.nan, C[:-1]]
    tr = np.nanmax(np.vstack([np.abs(H - L), np.abs(H - prev), np.abs(L - prev)]), axis=0)
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    atr_pct = np.where(C > 0, atr / C * 100.0, 0.0)
    regime_all = np.where(atr_pct > 4.0, 2, np.where(atr_pct > 2.0, 1, 0)).astype(np.int8)

    # --- adaptive-MA trend: pandas ewm(adjust=True) over a 64-row window is a
    # truncated geometric weighting, which is reproduced exactly here --------
    vol_ratio = np.clip(np.where(np.isfinite(atr_pct), atr_pct, 0.0) / 1.5, 0.5, 5.0)
    spans = np.where(atr_pct > 0, np.maximum(10, np.floor(50.0 / vol_ratio)).astype(int), 50)
    spans = np.where(np.isfinite(atr_pct), spans, 50)
    ma = np.full(n, np.nan)
    for sp in np.unique(spans[ends]):
        sel = ends[spans[ends] == sp]
        alpha = 2.0 / (sp + 1.0)
        w = (1.0 - alpha) ** np.arange(WIN - 1, -1, -1)       # oldest..newest
        w /= w.sum()
        ma[sel] = swC[sel - (WIN - 1)] @ w
    tu = C > ma

    ok = np.isfinite(atr) & np.isfinite(ma)
    valid = np.zeros(n, dtype=bool)
    valid[ends] = ok[ends] & np.isfinite(signal[ends])
    atr_pct_out[ends] = np.round(atr_pct[ends], 2)
    regime[ends] = regime_all[ends]
    trend_up[ends] = tu[ends]
    return Stage1(signal, atr_pct_out, regime, trend_up, valid)


def compute_stage1_day_anchored(df: pd.DataFrame, infer: Callable[[np.ndarray], np.ndarray],
                                batch: int = 2048) -> Stage1:
    """What the bot would see if data_feeds.py's request behaves as Alpaca's docs say.

    data_feeds.get_clean_ohlcv_dataframe asks for `limit=64` with no `start`.
    Alpaca documents the default `start` as the beginning of the current UTC
    day and the default sort as ascending, and alpaca-py 0.33 injects neither.
    That returns the FIRST 64 bars of today, not the latest 64.  Consequences,
    all reproduced here:
      * fewer than 32 completed bars today (00:00-08:00 UTC) -> get_clean_ohlcv
        returns None -> the symbol is skipped, INCLUDING held positions (their
        exits are never evaluated);
      * 32..64 bars (08:00-16:00 UTC) -> fresh data, windows anchored at 00:00;
      * more than 64 bars (16:00-24:00 UTC) -> the window stops growing: signal,
        regime and the decision price are frozen at the 16:00 close.
    This is a HYPOTHESIS derived from documentation and code, not something this
    script could observe (Alpaca is unreachable from where it was written);
    verify with the snippet in the report before relying on it.
    """
    n = len(df)
    H, L, C = _san(df["high"]), _san(df["low"]), _san(df["close"])
    signal = np.full(n, np.nan)
    atr_out = np.full(n, np.nan)
    regime = np.full(n, -1, dtype=np.int8)
    trend_up = np.zeros(n, dtype=bool)
    valid = np.zeros(n, dtype=bool)
    fresh = np.zeros(n, dtype=bool)
    dec_close = C.copy()
    if n == 0:
        return Stage1(signal, atr_out, regime, trend_up, valid, dec_close, fresh)

    days = df.index.normalize().values
    _, first = np.unique(days, return_index=True)       # index is sorted -> contiguous day blocks
    bounds = list(first) + [n]
    X_list: List[np.ndarray] = []
    tgt: List[int] = []
    frozen_src: List[Tuple[int, int]] = []              # (dst bar, src bar)

    for g in range(len(bounds) - 1):
        a, b = bounds[g], bounds[g + 1]
        nd = b - a
        if nd < SEQ:
            continue
        Lw = min(nd, WIN)
        F = add_features(df.iloc[a:a + Lw])[FEATURE_COLS].to_numpy(dtype=float)
        Hs, Ls, Cs = H[a:a + Lw], L[a:a + Lw], C[a:a + Lw]
        prev = np.r_[np.nan, Cs[:-1]]
        tr = np.nanmax(np.vstack([np.abs(Hs - Ls), np.abs(Hs - prev), np.abs(Ls - prev)]), axis=0)
        atr = pd.Series(tr).rolling(14).mean().to_numpy()
        for c in range(SEQ, Lw + 1):                    # c = completed bars today
            i = a + c - 1
            X_list.append(F[c - SEQ:c])
            tgt.append(i)
            price = Cs[c - 1]
            ap = atr[c - 1] / price * 100.0 if price > 0 else 0.0
            vr = min(max(ap / 1.5, 0.5), 5.0)
            span = max(10, int(50.0 / vr)) if ap > 0 else 50
            alpha = 2.0 / (span + 1.0)
            w = (1.0 - alpha) ** np.arange(c - 1, -1, -1)
            ma = float(w @ Cs[:c] / w.sum())
            atr_out[i] = round(ap, 2)
            regime[i] = 2 if ap > 4.0 else 1 if ap > 2.0 else 0
            trend_up[i] = price > ma
            fresh[i] = True
        if nd > WIN:                                    # window frozen after the 64th bar
            src = a + WIN - 1
            for i in range(a + WIN, b):
                frozen_src.append((i, src))

    for lo_ in range(0, len(X_list), batch):
        Xb = np.stack(X_list[lo_:lo_ + batch])
        p = np.asarray(infer(Xb), dtype=float)
        p[~np.isfinite(Xb).all(axis=(1, 2))] = 0.5
        signal[np.array(tgt[lo_:lo_ + batch])] = p
    valid[np.array(tgt, dtype=int)] = True
    for dst, src in frozen_src:
        signal[dst], atr_out[dst], regime[dst], trend_up[dst] = signal[src], atr_out[src], regime[src], trend_up[src]
        dec_close[dst] = C[src]
        valid[dst] = True
        fresh[dst] = False
    valid &= np.isfinite(signal)
    return Stage1(signal, atr_out, regime, trend_up, valid, dec_close, fresh)


def make_infer(model_path: str) -> Tuple[Callable[[np.ndarray], np.ndarray], dict]:
    """Wrap the repo's champion (torch transformer or promoted sklearn model)
    exactly as SafeMLPredictor.predict_batch preprocesses it."""
    from ml_predictor import SafeMLPredictor  # lazy: imports torch
    pred = SafeMLPredictor(model_path)
    sig = {"model_sha256": sha256_file(Path(model_path)), "kind": pred._kind}
    scaler_path = Path(pred._scaler_path())
    if scaler_path.exists():
        sig["scaler_sha256"] = sha256_file(scaler_path)

    if pred._kind == "sklearn":
        def infer_sk(X: np.ndarray) -> np.ndarray:
            rows = X[:, -1, :].astype(np.float64)
            proba = pred.clf.predict_proba(rows)
            classes = list(getattr(pred.clf, "classes_", [0, 1]))
            return proba[:, classes.index(1)] if 1 in classes else proba[:, -1]
        return infer_sk, sig

    import torch
    pred.model.eval()

    def infer_torch(X: np.ndarray) -> np.ndarray:
        d = np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        d = np.clip(d, -1e6, 1e6)
        if pred.scaler is not None:
            shp = d.shape
            d = pred.scaler.transform(d.reshape(-1, shp[-1])).astype(np.float32).reshape(shp)
            d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.empty(len(d), dtype=np.float64)
        with torch.no_grad():
            for i in range(0, len(d), 4096):
                t = torch.from_numpy(d[i:i + 4096]).to(pred.device)
                out[i:i + 4096] = torch.sigmoid(pred.model(t)).squeeze(-1).cpu().numpy()
        return out

    infer_torch.predictor = pred  # exposed for the parity check
    return infer_torch, sig


def stage1_cached(sym: str, df: pd.DataFrame, infer, model_sig: dict, cache_dir: Path,
                  mode: str = "latest64") -> Stage1:
    key = hashlib.sha1(json.dumps({
        "v": STAGE1_VERSION, "mode": mode, "sym": sym, "n": len(df), "t0": str(df.index[0]), "t1": str(df.index[-1]),
        "c": round(float(df["close"].sum()), 6), "v_": round(float(df["volume"].sum()), 6),
        "h": round(float(df["high"].sum()), 6), "model": model_sig,
    }, sort_keys=True).encode()).hexdigest()[:20]
    fp = cache_dir / f"stage1_{mode}_{sym.replace('/', '_')}_{key}.npz"
    if fp.exists():
        z = np.load(fp)
        dc = z["dec_close"] if "dec_close" in z.files else None
        fr = z["fresh"] if "fresh" in z.files else None
        return Stage1(z["signal"], z["atr_pct"], z["regime"], z["trend_up"], z["valid"], dc, fr)
    s1 = compute_stage1(df, infer) if mode == "latest64" else compute_stage1_day_anchored(df, infer)
    cache_dir.mkdir(parents=True, exist_ok=True)
    extra = {}
    if s1.dec_close is not None:
        extra = {"dec_close": s1.dec_close, "fresh": s1.fresh}
    np.savez_compressed(fp, signal=s1.signal, atr_pct=s1.atr_pct, regime=s1.regime,
                        trend_up=s1.trend_up, valid=s1.valid, **extra)
    return s1


def live_window(df: pd.DataFrame, i: int, mode: str = "latest64") -> pd.DataFrame:
    """The frame main_bot would have handed to the model for a decision made
    right after bar `i` completes, under each data-window hypothesis."""
    if mode == "latest64":
        return df.iloc[i - WIN + 1:i + 1]
    day = df.index[i].normalize()
    sub = df[(df.index >= day) & (df.index < day + pd.Timedelta(days=1))]
    c = sub.index.get_loc(df.index[i]) + 1
    return sub.iloc[:min(c, WIN)]


def verify_stage1_parity(bars: Dict[str, pd.DataFrame], s1s: Dict[str, Stage1], infer,
                         n_samples: int = 24, seed: int = 0, atol_signal: float = 1e-4,
                         mode: str = "latest64") -> List[str]:
    """Spot-check the vectorised pipeline against the repo's own per-window
    functions (compute_regime_and_trend and, if available, predict_batch).
    Returns a list of failure strings (empty == pass)."""
    rng = np.random.default_rng(seed)
    fails: List[str] = []
    pred = getattr(infer, "predictor", None)
    syms = [s for s in bars if s1s[s].valid.any()]
    for _ in range(n_samples):
        sym = syms[rng.integers(len(syms))]
        v = np.flatnonzero(s1s[sym].valid)
        i = int(v[rng.integers(len(v))])
        w = live_window(bars[sym], i, mode)
        tag = f"{sym}@{bars[sym].index[i]}"
        reg, trend, atr = compute_regime_and_trend(w)
        s = s1s[sym]
        if REGIME_NAMES[s.regime[i]] != reg:
            fails.append(f"{tag}: regime {REGIME_NAMES[s.regime[i]]} != {reg}")
        if bool(s.trend_up[i]) != (trend == "up"):
            fails.append(f"{tag}: trend_up {bool(s.trend_up[i])} != {trend}")
        if abs(s.atr_pct[i] - atr) > 0.011:
            fails.append(f"{tag}: atr_pct {s.atr_pct[i]} != {atr}")
        if s.dec_close is not None and abs(s.dec_close[i] - float(w["close"].iloc[-1])) > 1e-9:
            fails.append(f"{tag}: decision price {s.dec_close[i]} != {float(w['close'].iloc[-1])}")
        if pred is not None:
            live = pred.predict_batch({sym: w})[sym]
            if abs(live - s.signal[i]) > atol_signal:
                fails.append(f"{tag}: signal {s.signal[i]:.6f} != predict_batch {live:.6f}")
    return fails


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Stage 2 -- the simulator
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Params:
    equity0: float = float(cfg.ACCOUNT_BASE)
    fee_bps: float = 25.0              # Alpaca tier-1 TAKER; applied to every fill
    slippage_bps: float = 5.0          # added to the next-bar open (ask for buys, bid for sells)
    fill_model: str = "touch"          # "touch" | "strict"  (see Simulator._fill)
    killswitch: str = "live"           # "live" | "daily" | "off"
    universe_mode: str = "live"        # "live" | "trailing24h" | "all"
    buy_signal: float = float(cfg.BUY_SIGNAL)
    sell_signal: float = float(cfg.SELL_SIGNAL)
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None


@dataclass
class Pos:
    sym: str
    qty: float
    avg_entry: float
    cost_usd: float
    entry_fee_usd: float
    entry_ts: int          # decision time (the bot's state.entry_time)
    fill_ts: int
    fill_idx: int
    highest: float
    signal0: float
    regime0: str
    atr0: float
    rest_limit: Optional[float] = None   # GTC sell left on the book after a halt


class MarketData:
    """Per-symbol arrays aligned to a regular 15-minute decision grid."""

    def __init__(self, bars: Dict[str, pd.DataFrame], s1: Dict[str, Stage1], start, end,
                 universe_mode: str, candidates: List[str]):
        self.syms = list(bars)
        self.cands = [c for c in candidates if c in bars]
        self.o_ts, self.c_ts = {}, {}
        self.O, self.H, self.L, self.C, self.D = {}, {}, {}, {}, {}
        self.sig, self.atr, self.reg, self.tu, self.valid = {}, {}, {}, {}, {}
        self.dollar_vol: Dict[str, np.ndarray] = {}
        for s, df in bars.items():
            ots = (df.index.values.astype("datetime64[s]").astype(np.int64))
            self.o_ts[s], self.c_ts[s] = ots, ots + BAR_SECONDS
            self.O[s], self.H[s], self.L[s], self.C[s] = (df[c].to_numpy(float) for c in ("open", "high", "low", "close"))
            st = s1[s]
            self.D[s] = st.decision_price(self.C[s])
            self.sig[s], self.atr[s], self.reg[s], self.tu[s], self.valid[s] = st.signal, st.atr_pct, st.regime, st.trend_up, st.valid
            vol = df["volume"]
            if universe_mode == "trailing24h":
                dv = vol.rolling("24h").sum().to_numpy() * df["close"].to_numpy()
            else:  # "live": data_feeds.scan_stable_assets reads the CURRENT UTC day's partial daily bar
                dv = vol.groupby(df.index.normalize()).cumsum().to_numpy() * df["close"].to_numpy()
            self.dollar_vol[s] = dv
        t0 = to_sec(start) if start is not None else min(a[0] for a in self.c_ts.values())
        t1 = to_sec(end) if end is not None else max(a[-1] for a in self.c_ts.values())
        g0 = -(-t0 // BAR_SECONDS) * BAR_SECONDS
        g1 = (t1 // BAR_SECONDS) * BAR_SECONDS
        self.grid = np.arange(g0, g1 + 1, BAR_SECONDS, dtype=np.int64)
        self.last_idx = {s: np.searchsorted(self.c_ts[s], self.grid, side="right") - 1 for s in self.syms}
        self.next_idx = {s: np.searchsorted(self.o_ts[s], self.grid, side="left") for s in self.syms}

    def with_signals(self, sig: Dict[str, np.ndarray]) -> "MarketData":
        m = copy.copy(self)
        m.sig = sig
        return m


@dataclass
class SimResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    counters: Dict[str, int]
    halted_at: Optional[pd.Timestamp]
    halt_reason: str
    params: Params


_REASON_CODES = (("Trailing", "TRAIL"), ("Time-Decay", "STOP"), ("Slow-bleed", "SLOW_BLEED"),
                 ("Max hold", "MAX_HOLD"), ("Signal weak", "SIGNAL"))


def _reason_code(text: str) -> str:
    for needle, code in _REASON_CODES:
        if needle in text:
            return code
    return text


class Simulator:
    """Replays main_bot.run_trading_mode() one 15-minute cycle at a time.

    Differences from the live loop that are deliberate and documented:
      * one cycle per bar close (live loops every ~40s but sees identical
        data/price inside a bar; only sub-minute timing of time-based
        thresholds is lost);
      * orderbook "whale filter" is skipped (no historical L2 data);
      * orders are filled by `_fill`, not a broker.
    """

    def __init__(self, md: MarketData, params: Params):
        self.md, self.p = md, params
        self.fee = params.fee_bps / 1e4
        self.slip = params.slippage_bps / 1e4
        self._rp = {n: get_regime_params(n, params.buy_signal, params.sell_signal) for n in REGIME_NAMES}

    # ── order fills ──────────────────────────────────────────────────────────
    def _fill(self, side: str, sym: str, k: int, limit: float) -> Optional[Tuple[float, str, int]]:
        """Try to fill a limit order submitted at decision time grid[k].

        Fills only against the bar that STARTS at the decision time (no bar =
        no trades = no fill).  Marketable limit -> fills at the slipped open
        (price improvement vs the limit is kept).  "touch" additionally fills
        at the limit if the bar's range reaches it, modelling the bot's
        repeated re-submission of the same stale-price limit order within the
        bar; "strict" cancels instead (the conservative bound).
        """
        md = self.md
        nb = int(md.next_idx[sym][k])
        if nb >= len(md.o_ts[sym]) or md.o_ts[sym][nb] != md.grid[k]:
            return None
        o, h, l = md.O[sym][nb], md.H[sym][nb], md.L[sym][nb]
        if side == "B":
            ask = o * (1 + self.slip)
            if ask <= limit:
                return ask, "mkt", nb
            if self.p.fill_model == "touch" and l <= limit:
                return limit, "touch", nb
        else:
            bid = o * (1 - self.slip)
            if bid >= limit:
                return bid, "mkt", nb
            if self.p.fill_model == "touch" and h >= limit:
                return limit, "touch", nb
        return None

    # ── main loop ────────────────────────────────────────────────────────────
    def run(self) -> SimResult:
        md, p = self.md, self.p
        syms = md.syms
        cash = p.equity0
        positions: Dict[str, Pos] = {}
        cooldown: Dict[str, int] = {}
        latest_signal: Dict[str, float] = {}
        trades: List[dict] = []
        eq_rows: List[tuple] = []
        cnt: Counter = Counter()
        universe: List[str] = []
        last_scan = -10**12
        start_equity: Optional[float] = None
        session_base: Optional[float] = None
        session_day = None
        blocked_day = None
        halted = False
        halted_at, halt_reason = None, ""
        K = len(md.grid)

        def mtm(k) -> Tuple[float, int]:
            v = 0.0
            for s, ps in positions.items():
                j = md.last_idx[s][k]
                v += ps.qty * (md.C[s][j] if j >= 0 else ps.avg_entry)
            return v, len(positions)

        def close_position(sym: str, k: int, reason: str, limit: Optional[float] = None) -> bool:
            nonlocal cash
            ps = positions[sym]
            j = md.last_idx[sym][k]
            dec_px = md.D[sym][j]
            lim = limit if limit is not None else sanitize_price(dec_px * 0.999)
            res = self._fill("S", sym, k, lim)
            if res is None:
                cnt["sell_unfilled"] += 1
                return False
            px, kind, nb = res
            gross = ps.qty * px
            exit_fee = gross * self.fee
            proceeds = gross - exit_fee
            cash += proceeds
            a, b = ps.fill_idx, max(nb, ps.fill_idx)
            trades.append({
                "symbol": sym,
                "entry_ts": pd.Timestamp(ps.entry_ts, unit="s"), "entry_fill_ts": pd.Timestamp(ps.fill_ts, unit="s"),
                "exit_ts": pd.Timestamp(int(md.grid[k]), unit="s"), "exit_fill_ts": pd.Timestamp(int(md.o_ts[sym][nb]), unit="s"),
                "entry_px": ps.avg_entry, "exit_px": px, "qty": ps.qty, "cost_usd": ps.cost_usd,
                "proceeds_usd": proceeds, "fees_usd": ps.entry_fee_usd + exit_fee,
                "pnl_usd": proceeds - ps.cost_usd, "ret_net": proceeds / ps.cost_usd - 1.0,
                "ret_gross": px / ps.avg_entry - 1.0,
                "hold_h": (int(md.grid[k]) - ps.entry_ts) / 3600.0,
                "reason": _reason_code(reason), "signal0": ps.signal0, "regime0": ps.regime0, "atr0": ps.atr0,
                "mae": float(md.L[sym][a:b + 1].min() / ps.avg_entry - 1.0),
                "mfe": float(md.H[sym][a:b + 1].max() / ps.avg_entry - 1.0),
                "fill_kind": kind,
            })
            del positions[sym]
            return True

        def flatten_all(k: int, reason: str) -> None:
            for s in list(positions):
                if not close_position(s, k, reason):
                    positions[s].rest_limit = sanitize_price(md.D[s][md.last_idx[s][k]] * 0.999)

        for k in range(K):
            T = int(md.grid[k])
            day = T // 86400

            # ── halted bot: only pre-existing GTC sells can still fill ──────
            if halted:
                for s in list(positions):
                    rl = positions[s].rest_limit
                    if rl is not None:
                        close_position(s, k, "KILL", limit=rl)
                mv, npos = mtm(k)
                eq_rows.append((T, cash + mv, cash, mv, npos))
                continue

            mv, npos = mtm(k)
            equity = cash + mv
            eq_rows.append((T, equity, cash, mv, npos))
            if start_equity is None:
                start_equity = equity
            if session_day != day:
                session_day, session_base = day, equity
            base = session_base if p.killswitch == "daily" else start_equity
            drawdown = (equity - start_equity) / start_equity * 100.0 if start_equity else None
            sess_loss = (equity - base) / base * 100.0 if base else None

            # ── kill-switches (order and semantics copied from main_bot.py) ─
            if p.killswitch != "off" and drawdown is not None and drawdown < cfg.MAX_DRAWDOWN_STOP:
                halted, halted_at, halt_reason = True, pd.Timestamp(T, unit="s"), "MAX_DRAWDOWN (no flatten)"
                continue
            risk_taper = stop_taper = 1.0
            if drawdown is not None:
                if drawdown < -7.0:
                    risk_taper, stop_taper = 0.5, 0.5
                elif drawdown < -5.0:
                    risk_taper, stop_taper = 0.7, 0.75
            if p.killswitch != "off" and sess_loss is not None and sess_loss <= cfg.DAILY_LOSS_LIMIT:
                cnt["killswitch_trips"] += 1
                flatten_all(k, "KILL")
                if p.killswitch == "live":
                    halted, halted_at, halt_reason = True, pd.Timestamp(T, unit="s"), f"DAILY_LOSS_LIMIT ({sess_loss:.2f}% vs start)"
                else:
                    blocked_day = day
                continue

            open_count = sum(1 for s, ps in positions.items() if ps.qty * md.C[s][max(md.last_idx[s][k], 0)] >= cfg.MIN_POSITION_USD)
            total_value = mv
            max_pv = equity * cfg.BASE_RISK_PERCENT * cfg.MAX_OPEN_POSITIONS
            buys_allowed = blocked_day != day
            running_pv = total_value
            sold_this_cycle = set()

            if total_value >= max_pv and positions:
                cnt["exposure_cap_hit"] += 1
                big = max(positions, key=lambda s: positions[s].qty * md.C[s][max(md.last_idx[s][k], 0)])
                if close_position(big, k, "CAP_FORCED"):
                    sold_this_cycle.add(big)
                    cnt["forced_sells"] += 1
                buys_allowed = False
            if open_count >= cfg.MAX_OPEN_POSITIONS:
                buys_allowed = False

            # ── universe rotation (hourly) ──────────────────────────────────
            if T - last_scan >= cfg.UNIVERSE_REFRESH_SECONDS:
                universe = self._scan(k, T)
                last_scan = T
            held = sorted(positions)
            to_process = list(dict.fromkeys(universe + held))

            for sym in to_process:
                j = int(md.last_idx[sym][k])
                if j < 0 or not md.valid[sym][j]:
                    continue
                signal = float(md.sig[sym][j])
                regime = REGIME_NAMES[int(md.reg[sym][j])]
                atr_pct = float(md.atr[sym][j])
                trend_up = bool(md.tu[sym][j])
                rp = self._rp[regime]
                price = float(md.D[sym][j])      # price the bot DECIDES on (stale if its data window is frozen)
                latest_signal[sym] = signal
                if price <= 0:
                    continue

                # ── EXIT ────────────────────────────────────────────────────
                if sym in sold_this_cycle:
                    continue
                ps = positions.get(sym)
                if ps is not None and ps.qty * price >= cfg.MIN_POSITION_USD:
                    held_h = (T - ps.entry_ts) / 3600.0
                    dec = evaluate_exit(
                        avg_entry=ps.avg_entry, price=price, highest_seen=ps.highest, held_hours=held_h,
                        signal=signal, regime=regime, atr_pct=atr_pct,
                        profit_target_pct=rp["profit_target_pct"], stop_loss_pct=rp["stop_loss_pct"],
                        sell_signal=rp["sell_signal"], max_hold_hours=cfg.MAX_HOLD_HOURS,
                        min_hold_hours_before_signal=cfg.MIN_HOLD_HOURS_BEFORE_SIGNAL,
                        slow_bleed_pct=cfg.SLOW_BLEED_PCT, slow_bleed_min_hours=cfg.SLOW_BLEED_MIN_HOURS,
                        trailing_stop_atr_multiplier=cfg.TRAILING_STOP_ATR_MULTIPLIER,
                        min_trailing_stop_pct=cfg.MIN_TRAILING_STOP_PCT, max_trailing_stop_pct=cfg.MAX_TRAILING_STOP_PCT,
                    )
                    ps.highest = dec.highest_seen
                    if dec.exit_reason:
                        if close_position(sym, k, dec.exit_reason):
                            cooldown[sym] = T + cfg.COOLDOWN_SECONDS_SELL
                    continue

                # ── ENTRY ───────────────────────────────────────────────────
                if sym not in universe:
                    continue
                if T < cooldown.get(sym, 0):
                    cnt["skip_cooldown"] += 1
                    continue
                if not (trend_up and signal > rp["buy_signal"]):
                    continue
                cnt["entry_candidates"] += 1
                if atr_pct > 6.0:
                    cnt["veto_atr_gt_6"] += 1
                    continue
                if regime == "wild" and signal < 0.70:
                    cnt["veto_wild_low_conviction"] += 1
                    continue

                if not buys_allowed:
                    sold = self._swap_weakest(sym, signal, latest_signal, positions, k, close_position)
                    if sold > 0:
                        cnt["swaps"] += 1
                        buys_allowed = True
                        open_count = max(0, open_count - 1)
                        running_pv = max(0.0, running_pv - sold)
                    else:
                        cnt["suppressed_cap_no_swap"] += 1
                        continue

                kelly = kelly_multiplier(signal, rp["profit_target_pct"], rp["stop_loss_pct"])
                adj = calculate_adjusted_risk(equity, atr_pct) * 1.0 * kelly
                if drawdown is not None:
                    adj *= risk_taper * stop_taper
                qty = money_qty(adj / price)
                trade_value = min(qty * price, cfg.MAX_SINGLE_TRADE_USD)
                pos_cap = equity * cfg.MAX_POSITION_PCT
                if trade_value > pos_cap:
                    trade_value = pos_cap
                    cnt["capped_20pct"] += 1
                qty = trade_value / price
                if trade_value < cfg.MIN_ORDER_USD:
                    cnt["skip_below_min_order"] += 1
                    continue
                if max_pv - running_pv < trade_value:
                    cnt["blocked_headroom"] += 1
                    buys_allowed = False
                    continue
                if cash < trade_value:
                    cnt["blocked_cash"] += 1
                    buys_allowed = False
                    continue

                qty = math.floor(qty * 1e8) / 1e8
                limit = sanitize_price(price * 1.001)
                res = self._fill("B", sym, k, limit)
                if res is None:
                    cnt["buy_unfilled"] += 1
                    continue
                px, kind, nb = res
                notional = qty * px
                if notional > cash:
                    cnt["blocked_cash"] += 1
                    continue
                fee_usd = notional * self.fee
                cash -= notional
                positions[sym] = Pos(
                    sym=sym, qty=qty * (1 - self.fee), avg_entry=px, cost_usd=notional, entry_fee_usd=fee_usd,
                    entry_ts=T, fill_ts=int(md.o_ts[sym][nb]), fill_idx=nb, highest=price,
                    signal0=signal, regime0=regime, atr0=atr_pct,
                )
                cooldown[sym] = T + cfg.COOLDOWN_SECONDS_BUY
                running_pv += trade_value
                open_count += 1
                cnt["entries"] += 1
                if open_count >= cfg.MAX_OPEN_POSITIONS:
                    buys_allowed = False

        # ── end of data: mark remaining positions to market through the same
        # fee/slippage path so open trades are not flattered ---------------
        if positions:
            k = K - 1
            for s in list(positions):
                ps = positions[s]
                j = md.last_idx[s][k]
                px = md.C[s][j] * (1 - self.slip)
                gross = ps.qty * px
                proceeds = gross * (1 - self.fee)
                cash += proceeds
                trades.append({
                    "symbol": s, "entry_ts": pd.Timestamp(ps.entry_ts, unit="s"), "entry_fill_ts": pd.Timestamp(ps.fill_ts, unit="s"),
                    "exit_ts": pd.Timestamp(int(md.grid[k]), unit="s"), "exit_fill_ts": pd.Timestamp(int(md.grid[k]), unit="s"),
                    "entry_px": ps.avg_entry, "exit_px": px, "qty": ps.qty, "cost_usd": ps.cost_usd,
                    "proceeds_usd": proceeds, "fees_usd": ps.entry_fee_usd + gross * self.fee,
                    "pnl_usd": proceeds - ps.cost_usd, "ret_net": proceeds / ps.cost_usd - 1.0, "ret_gross": px / ps.avg_entry - 1.0,
                    "hold_h": (int(md.grid[k]) - ps.entry_ts) / 3600.0, "reason": "END", "signal0": ps.signal0,
                    "regime0": ps.regime0, "atr0": ps.atr0, "mae": float(md.L[s][ps.fill_idx:max(j, ps.fill_idx) + 1].min() / ps.avg_entry - 1.0),
                    "mfe": float(md.H[s][ps.fill_idx:max(j, ps.fill_idx) + 1].max() / ps.avg_entry - 1.0), "fill_kind": "eod",
                })
                del positions[s]
            eq_rows.append((int(md.grid[-1]), cash, cash, 0.0, 0))

        eq = pd.DataFrame(eq_rows, columns=["t", "equity", "cash", "mv", "npos"])
        eq.index = pd.to_datetime(eq.pop("t"), unit="s")
        eq = eq[~eq.index.duplicated(keep="last")]
        tr = pd.DataFrame(trades)
        return SimResult(tr, eq, dict(cnt), halted_at, halt_reason, self.p)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _scan(self, k: int, T: int) -> List[str]:
        """data_feeds.scan_stable_assets over DYNAMIC_UNIVERSE_CANDIDATES."""
        md = self.md
        if self.p.universe_mode == "all":
            return list(md.cands)
        vols = []
        for s in md.cands:
            j = md.last_idx[s][k]
            if j < 0 or T - md.c_ts[s][j] > 2 * 86400:
                continue
            vols.append((s, md.dollar_vol[s][j]))
        if not vols:
            return ["BTC/USD", "ETH/USD", "SOL/USD"]
        vols.sort(key=lambda x: -x[1])
        return [s for s, _ in vols[:cfg.UNIVERSE_SIZE]]

    def _swap_weakest(self, new_sym, new_signal, latest_signal, positions, k, close_position,
                      threshold: float = 0.05) -> float:
        """portfolio.swap_weakest_position: sell the held name with the lowest
        last-seen signal if the new one beats it by >= 0.05."""
        if not positions:
            return 0.0
        weakest, wsig = None, float("inf")
        for s in positions:
            sg = latest_signal.get(s, 0.5)
            if sg < wsig:
                weakest, wsig = s, sg
        if weakest is None or (new_signal - wsig) < threshold:
            return 0.0
        j = self.md.last_idx[weakest][k]
        mv = positions[weakest].qty * self.md.C[weakest][j]
        return mv if close_position(weakest, k, "SWAP") else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Analytics
# ══════════════════════════════════════════════════════════════════════════════
def max_drawdown(eq: np.ndarray) -> Tuple[float, float]:
    """(max drawdown as a negative fraction, longest underwater stretch in 15m bars)."""
    if len(eq) == 0:
        return float("nan"), float("nan")
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    under = dd < 0
    longest = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    return float(dd.min()), float(longest)


def daily_returns(eq: pd.Series) -> pd.Series:
    d = eq.resample("1D").last().dropna()
    return d.pct_change().dropna()


def sharpe(r: np.ndarray, ppy: float = 365.0) -> float:
    r = np.asarray(r, float)
    if len(r) < 2 or np.std(r, ddof=1) == 0:
        return float("nan")
    return float(np.mean(r) / np.std(r, ddof=1) * math.sqrt(ppy))


def block_bootstrap_sharpe(r: np.ndarray, n_boot: int = 2000, block: int = 7, seed: int = 0) -> Tuple[float, float]:
    r = np.asarray(r, float)
    n = len(r)
    if n < 2 * block:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    samp = r[idx]
    sd = samp.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sh = np.where(sd > 0, samp.mean(axis=1) / sd * math.sqrt(365.0), np.nan)
    return float(np.nanpercentile(sh, 2.5)), float(np.nanpercentile(sh, 97.5))


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 5000, seed: int = 0) -> Tuple[float, float]:
    x = np.asarray(x, float)
    if len(x) < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def ols_hac(y: np.ndarray, x: np.ndarray, lags: int = 5) -> dict:
    """y = a + b x with Newey-West (Bartlett) standard errors."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return {"alpha": beta[0], "beta": beta[1], "t_alpha": beta[0] / se[0], "t_beta": beta[1] / se[1],
            "r2": 1 - (e @ e) / (((y - y.mean()) ** 2).sum() or np.nan)}


def equity_stats(eq: pd.Series, label: str = "") -> dict:
    eq = eq.dropna()
    out = {"label": label}
    if len(eq) < 3:
        return out
    days = max((eq.index[-1] - eq.index[0]).total_seconds() / 86400.0, 1e-9)
    tot = eq.iloc[-1] / eq.iloc[0] - 1.0
    dr = daily_returns(eq)
    mdd, dur = max_drawdown(eq.to_numpy())
    lo, hi = block_bootstrap_sharpe(dr.to_numpy())
    out.update({
        "days": days, "total_return": tot,
        "cagr": (1 + tot) ** (365.0 / days) - 1 if days >= 30 and tot > -1 else float("nan"),
        "sharpe": sharpe(dr.to_numpy()), "sharpe_ci_lo": lo, "sharpe_ci_hi": hi,
        "sortino": float(dr.mean() / dr[dr < 0].std(ddof=1) * math.sqrt(365.0)) if (dr < 0).sum() > 1 else float("nan"),
        "max_dd": mdd, "max_dd_days": dur * BAR_SECONDS / 86400.0,
        "calmar": float((1 + tot) ** (365.0 / days) - 1) / abs(mdd) if mdd < 0 and days >= 30 and tot > -1 else float("nan"),
        "n_days": len(dr),
    })
    return out


def trade_stats(tr: pd.DataFrame, seed: int = 0) -> dict:
    if tr is None or len(tr) == 0:
        return {"n_trades": 0}
    r = tr["ret_net"].to_numpy()
    w, l = tr[tr["pnl_usd"] > 0], tr[tr["pnl_usd"] <= 0]
    lo, hi = bootstrap_mean_ci(r, seed=seed)
    sd = r.std(ddof=1) if len(r) > 1 else float("nan")
    return {
        "n_trades": len(tr), "win_rate": float((r > 0).mean()),
        "mean_net_bps": float(r.mean() * 1e4), "median_net_bps": float(np.median(r) * 1e4),
        "mean_net_bps_ci_lo": lo * 1e4, "mean_net_bps_ci_hi": hi * 1e4,
        "t_stat": float(r.mean() / (sd / math.sqrt(len(r)))) if len(r) > 1 and sd > 0 else float("nan"),
        "mean_gross_bps": float(tr["ret_gross"].mean() * 1e4),
        "avg_win_bps": float(w["ret_net"].mean() * 1e4) if len(w) else float("nan"),
        "avg_loss_bps": float(l["ret_net"].mean() * 1e4) if len(l) else float("nan"),
        "profit_factor": float(w["pnl_usd"].sum() / abs(l["pnl_usd"].sum())) if len(l) and l["pnl_usd"].sum() != 0 else float("nan"),
        "total_pnl_usd": float(tr["pnl_usd"].sum()), "total_fees_usd": float(tr["fees_usd"].sum()),
        "avg_hold_h": float(tr["hold_h"].mean()), "avg_notional_usd": float(tr["cost_usd"].mean()),
        "avg_mae_bps": float(tr["mae"].mean() * 1e4), "avg_mfe_bps": float(tr["mfe"].mean() * 1e4),
    }


def summarize(res: SimResult, seed: int = 0, t_from: Optional[pd.Timestamp] = None) -> dict:
    eq = res.equity["equity"]
    tr = res.trades
    if t_from is not None:
        eq = eq[eq.index >= t_from]
        tr = tr[tr["entry_ts"] >= t_from] if len(tr) else tr
    s = equity_stats(eq)
    s.update(trade_stats(tr, seed))
    e = res.equity if t_from is None else res.equity[res.equity.index >= t_from]
    s["avg_exposure"] = float((e["mv"] / e["equity"]).mean()) if len(e) else float("nan")
    s["max_exposure"] = float((e["mv"] / e["equity"]).max()) if len(e) else float("nan")
    return s


def benchmark_daily(bars: Dict[str, pd.DataFrame], symbols: List[str], t0, t1) -> Dict[str, pd.Series]:
    """Daily returns for BTC buy&hold and an equal-weight basket (daily rebalanced)."""
    closes = {}
    for s in symbols:
        if s in bars:
            c = bars[s]["close"]
            closes[s] = c[(c.index >= t0) & (c.index <= t1)].resample("1D").last()
    if not closes:
        return {}
    px = pd.DataFrame(closes).ffill()
    rets = px.pct_change().dropna(how="all")
    out = {"EW basket": rets.mean(axis=1)}
    if "BTC/USD" in rets:
        out["BTC"] = rets["BTC/USD"]
    return out


def signal_diagnostics(bars: Dict[str, pd.DataFrame], s1: Dict[str, Stage1],
                       horizons=(1, 2, 4, 6, 8, 16), t_from=None, t_to=None) -> dict:
    """Does the raw model output predict forward returns at all?  Independent
    of every trading rule -- this is the check KNOWN_ISSUES.md asks for."""
    frames = []
    for s, df in bars.items():
        st = s1[s]
        c = df["close"].to_numpy(float)
        d = pd.DataFrame({"signal": st.signal, "trend_up": st.trend_up, "regime": st.regime, "atr": st.atr_pct}, index=df.index)
        for h in horizons:
            f = np.full(len(c), np.nan)
            f[:-h] = np.log(c[h:] / c[:-h])
            d[f"fwd{h}"] = f
        d["symbol"] = s
        d = d[st.fresh_mask()]
        if t_from is not None:
            d = d[d.index >= t_from]
        if t_to is not None:
            d = d[d.index <= t_to]
        frames.append(d)
    if not frames:
        return {}
    A = pd.concat(frames)
    out: dict = {"n_obs": len(A)}
    if len(A) < 50:
        return out

    rows = []
    for s, g in A.groupby("symbol"):
        row = {"symbol": s, "n": len(g)}
        for h in horizons:
            gg = g[["signal", f"fwd{h}"]].dropna()
            if len(gg) > 30 and gg["signal"].nunique() > 1:
                ic = gg["signal"].rank().corr(gg[f"fwd{h}"].rank())
                n_eff = len(gg) / h
                row[f"IC_{h}"] = ic
                row[f"t_{h}"] = ic * math.sqrt(max(n_eff - 2, 1) / max(1 - ic ** 2, 1e-12))
        rows.append(row)
    ic_tab = pd.DataFrame(rows).set_index("symbol")
    out["ic_by_symbol"] = ic_tab
    out["ic_mean"] = {h: float(ic_tab[f"IC_{h}"].mean()) for h in horizons if f"IC_{h}" in ic_tab}

    # month-by-month IC at the model's training horizon (6 bars = 90 min)
    h = 6 if 6 in horizons else horizons[0]
    mo = []
    for (s, m), g in A.groupby(["symbol", A.index.to_period("M")]):
        gg = g[["signal", f"fwd{h}"]].dropna()
        if len(gg) > 200 and gg["signal"].nunique() > 1:
            mo.append({"symbol": s, "month": str(m), "ic": gg["signal"].rank().corr(gg[f"fwd{h}"].rank())})
    if mo:
        mdf = pd.DataFrame(mo).groupby("month")["ic"].mean()
        out["ic_monthly"] = mdf
        out["ic_monthly_summary"] = {"mean": float(mdf.mean()), "std": float(mdf.std(ddof=1)) if len(mdf) > 1 else float("nan"),
                                     "pct_positive": float((mdf > 0).mean()), "n_months": len(mdf)}

    B = A.dropna(subset=[f"fwd{h}"]).copy()
    B["bin"] = pd.qcut(B["signal"].rank(method="first"), 10, labels=False)
    dec = B.groupby("bin").agg(sig_lo=("signal", "min"), sig_hi=("signal", "max"), n=("signal", "size"),
                               fwd_bps=(f"fwd{h}", lambda x: x.mean() * 1e4),
                               hit=(f"fwd{h}", lambda x: (x > 0).mean()))
    out["deciles"] = dec
    out["decile_monotonicity"] = float(dec.index.to_series().corr(dec["fwd_bps"], method="spearman"))
    out["base_rate_up"] = float((B[f"fwd{h}"] > 0).mean())
    out["horizon"] = h

    thr_rows = []
    for thr in (0.47, 0.51, 0.55, 0.60, 0.65, 0.70):
        g = B[(B["signal"] > thr) & B["trend_up"]]
        thr_rows.append({"signal >": thr, "n": len(g), "mean_fwd_bps": g[f"fwd{h}"].mean() * 1e4 if len(g) else np.nan,
                         "hit_rate": (g[f"fwd{h}"] > 0).mean() if len(g) else np.nan})
    out["thresholds"] = pd.DataFrame(thr_rows).set_index("signal >")
    out["uncond_fwd_bps"] = float(B[f"fwd{h}"].mean() * 1e4)

    cal = B.assign(cb=pd.cut(B["signal"], [0, .4, .45, .5, .55, .6, 1.0])).groupby("cb", observed=True).agg(
        n=("signal", "size"), mean_signal=("signal", "mean"), realized_up=(f"fwd{h}", lambda x: (x > 0).mean()))
    out["calibration"] = cal
    y = (B[f"fwd{h}"] > 0).astype(float)
    out["brier"] = float(((B["signal"] - y) ** 2).mean())
    out["brier_baseline"] = float(((y.mean() - y) ** 2).mean())
    out["signal_dist"] = B["signal"].describe(percentiles=[.05, .25, .5, .75, .95]).to_dict()
    out["regime_occupancy"] = {REGIME_NAMES[i]: float((A["regime"] == i).mean()) for i in range(3)}
    out["trend_up_frac"] = float(A["trend_up"].mean())
    out["atr_pct_median"] = float(A["atr"].median())
    return out


def circular_shift_signals(md: MarketData, s1: Dict[str, Stage1], rng: np.random.Generator,
                           min_shift_bars: int = 96 * 3) -> Dict[str, np.ndarray]:
    """Null model: roll each symbol's valid signal series by a random offset.
    Keeps the signal's marginal distribution and autocorrelation; destroys any
    link to price.  Trend/regime/exit mechanics stay aligned to price."""
    out = {}
    for s in md.syms:
        sig = s1[s].signal.copy()
        v = np.flatnonzero(s1[s].valid)
        if len(v) > 2 * min_shift_bars:
            seg = sig[v]
            sig[v] = np.roll(seg, int(rng.integers(min_shift_bars, len(seg) - min_shift_bars)))
        out[s] = sig
    return out


def availability_by_hour(bars: Dict[str, pd.DataFrame], s1: Dict[str, Stage1], block: int = 3) -> pd.DataFrame:
    """Share of decision cycles with fresh / frozen / no data, by UTC hour block."""
    rows = []
    for s, df in bars.items():
        st = s1[s]
        h = ((df.index + pd.Timedelta(minutes=15)).hour // block) * block
        rows.append(pd.DataFrame({"blk": h, "fresh": st.fresh_mask(), "frozen": st.valid & ~st.fresh_mask(), "none": ~st.valid}))
    A = pd.concat(rows)
    t = A.groupby("blk")[["fresh", "frozen", "none"]].mean() * 100
    t.index = [f"{b:02d}:00-{b + block:02d}:00" for b in t.index]
    t.index.name = "UTC hours"
    return t.rename(columns={"fresh": "fresh %", "frozen": "stale/frozen %", "none": "no data %"})


# ── reporting helpers ─────────────────────────────────────────────────────────
def _f(x, fmt="{:.2f}", na="n/a") -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return na
    return fmt.format(x)


def md_table(df: pd.DataFrame, fmt: str = "{:.3f}", index: bool = True) -> str:
    if df is None or len(df) == 0:
        return "_(empty)_\n"
    d = df.reset_index() if index else df
    head = "| " + " | ".join(str(c) for c in d.columns) + " |\n|" + "|".join("---" for _ in d.columns) + "|\n"
    rows = []
    for tup in d.itertuples(index=False, name=None):      # itertuples keeps per-column dtypes (iterrows upcasts ints)
        cells = []
        for v in tup:
            if isinstance(v, (bool, np.bool_)):
                cells.append(str(v))
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{int(v):,}")
            elif isinstance(v, (float, np.floating)):
                cells.append(_f(float(v), fmt))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(rows) + "\n"


def _usd(x, fmt="{:,.2f}") -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return ("-$" if x < 0 else "$") + fmt.format(abs(x))


def summary_row(label: str, s: dict) -> dict:
    return {
        "scenario": label, "trades": s.get("n_trades", 0),
        "total_ret_%": s.get("total_return", np.nan) * 100, "sharpe": s.get("sharpe", np.nan),
        "max_dd_%": s.get("max_dd", np.nan) * 100, "mean_net_bps/trade": s.get("mean_net_bps", np.nan),
        "win_%": s.get("win_rate", np.nan) * 100, "PF": s.get("profit_factor", np.nan),
    }


def make_plots(res: SimResult, bench: Dict[str, pd.Series], out: Path, oos_start) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    eq = res.equity["equity"]
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    ax[0].plot(eq.index, eq / eq.iloc[0], label="Strategy (net of fees/slippage)", lw=1.4)
    for name, r in bench.items():
        cum = (1 + r).cumprod()
        ax[0].plot(cum.index, cum / cum.iloc[0], lw=0.9, alpha=0.8, label=f"{name} buy&hold")
    if oos_start is not None:
        ax[0].axvline(oos_start, color="k", ls="--", lw=0.8)
        ax[0].text(oos_start, ax[0].get_ylim()[1], " out-of-sample", va="top", fontsize=8)
    ax[0].set_ylabel("Growth of 1.0"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    peak = eq.cummax()
    ax[1].fill_between(eq.index, (eq / peak - 1) * 100, 0, color="tab:red", alpha=0.5)
    ax[1].set_ylabel("Drawdown %"); ax[1].grid(alpha=0.3)
    ax[2].plot(res.equity.index, res.equity["mv"] / res.equity["equity"] * 100, lw=0.7)
    ax[2].set_ylabel("Exposure % equity"); ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fp = out / "equity.png"
    fig.savefig(fp, dpi=110)
    plt.close(fig)
    return fp


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Orchestration / report
# ══════════════════════════════════════════════════════════════════════════════
def run_scenarios(md: MarketData, base: Params, grid: List[Tuple[str, dict]]) -> pd.DataFrame:
    rows = []
    for label, over in grid:
        p = copy.copy(base)
        for k, v in over.items():
            setattr(p, k, v)
        res = Simulator(md, p).run()
        rows.append(summary_row(label, summarize(res)))
    return pd.DataFrame(rows).set_index("scenario")


def run_null(md: MarketData, s1: Dict[str, Stage1], params: Params, trials: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(trials):
        md_n = md.with_signals(circular_shift_signals(md, s1, rng))
        res = Simulator(md_n, params).run()
        s = summarize(res)
        rows.append({"trial": i, "total_return": s.get("total_return", np.nan), "mean_net_bps": s.get("mean_net_bps", np.nan),
                     "sharpe": s.get("sharpe", np.nan), "trades": s.get("n_trades", 0)})
    return pd.DataFrame(rows).set_index("trial")


def build_report(ctx: dict) -> str:
    a = ctx["args"]; res: SimResult = ctx["res"]; S = ctx["summary"]; S_is = ctx.get("summary_is"); S_oos = ctx.get("summary_oos")
    L: List[str] = []
    w = L.append
    w(f"# Apex backtest report\n")
    if a.source == "synthetic":
        w("> **SYNTHETIC DATA.** This run only exercises the machinery. The generator contains no exploitable pattern, "
          "so none of the numbers below say anything about the strategy.\n")
    g = ctx["git"]
    w("## 0. Run configuration\n")
    w(md_table(pd.DataFrame([
        ["data source", a.source], ["window (UTC)", f"{ctx['t0']:%Y-%m-%d %H:%M} -> {ctx['t1']:%Y-%m-%d %H:%M} ({S.get('days', 0):.0f} days)"],
        ["symbols with data", ", ".join(ctx["symbols"])], ["start equity", f"${res.params.equity0:,.0f}"],
        ["fee / slippage (per fill)", f"{res.params.fee_bps:g} bps / {res.params.slippage_bps:g} bps"],
        ["fill model", res.params.fill_model], ["kill-switch mode", res.params.killswitch],
        ["universe mode", res.params.universe_mode], ["data window", ctx.get("primary", "latest64")], ["BUY / SELL base signal", f"{res.params.buy_signal} / {res.params.sell_signal}"],
        ["repo commit", f"{g.get('commit', '')[:10]}{' (dirty)' if g.get('dirty') else ''}"],
        ["model sha256", ctx["model_sig"].get("model_sha256", "")[:16]],
        ["model last committed", g.get("model_last_commit_date", "") or "unknown"],
        ["out-of-sample from", str(ctx["oos_start"]) if ctx["oos_start"] is not None else "none"],
        ["seed", a.seed],
    ], columns=["", ""]), index=False))
    w("Stage-1 parity vs repo functions: **" + ("PASS" if not ctx["parity_fails"] else "FAIL") + "**"
      + (f" ({len(ctx['parity_fails'])} mismatches, see log)" if ctx["parity_fails"] else "") + "\n")

    w("## 1. Headline (net of fees and slippage)\n")
    rows = [("Total return", _f(S.get("total_return", np.nan) * 100, "{:.2f}%")),
            ("CAGR (only meaningful for >= 90 days)", _f(S.get("cagr", np.nan) * 100, "{:.1f}%")),
            ("Sharpe (daily, 365d) [95% block-bootstrap CI]", f"{_f(S.get('sharpe'))} [{_f(S.get('sharpe_ci_lo'))}, {_f(S.get('sharpe_ci_hi'))}]"),
            ("Max drawdown / longest underwater", f"{_f(S.get('max_dd', np.nan) * 100, '{:.2f}%')} / {_f(S.get('max_dd_days'), '{:.1f}')} d"),
            ("Average / max exposure (% of equity)", f"{_f(S.get('avg_exposure', np.nan) * 100, '{:.1f}')} / {_f(S.get('max_exposure', np.nan) * 100, '{:.1f}')}"),
            ("Trades", str(S.get("n_trades", 0))),
            ("Win rate", _f(S.get("win_rate", np.nan) * 100, "{:.1f}%")),
            ("Mean NET return per trade [95% CI], bps", f"{_f(S.get('mean_net_bps'), '{:.1f}')} [{_f(S.get('mean_net_bps_ci_lo'), '{:.1f}')}, {_f(S.get('mean_net_bps_ci_hi'), '{:.1f}')}]"),
            ("Mean GROSS return per trade, bps", _f(S.get("mean_gross_bps"), "{:.1f}")),
            ("t-statistic of mean net trade return", _f(S.get("t_stat"))),
            ("Profit factor", _f(S.get("profit_factor"))),
            ("Avg win / avg loss, bps", f"{_f(S.get('avg_win_bps'), '{:.0f}')} / {_f(S.get('avg_loss_bps'), '{:.0f}')}"),
            ("Avg hold (h) / avg trade size", f"{_f(S.get('avg_hold_h'))} / ${_f(S.get('avg_notional_usd'), '{:,.0f}')}"),
            ("Total P&L / total fees", f"{_usd(S.get('total_pnl_usd'))} / {_usd(S.get('total_fees_usd'))}"),
            ("Avg worst intrabar excursion (MAE) vs close-based exit", f"{_f(S.get('avg_mae_bps'), '{:.0f}')} bps")]
    w(md_table(pd.DataFrame(rows, columns=["metric", "value"]), index=False))
    if res.halted_at is not None:
        w(f"\n**The bot halted at {res.halted_at:%Y-%m-%d %H:%M} UTC -- {res.halt_reason}.** Under the `{res.params.killswitch}` "
          f"kill-switch mode it never trades again, so everything after that is idle equity. See section 8.\n")
    if S.get("days", 0) < 90:
        w("\n> Window shorter than 90 days: annualised figures and Sharpe are unreliable.\n")

    if S_is is not None or S_oos is not None:
        w("\n### In-sample vs out-of-sample\n")
        w("The model's training window is not in this repo (train_transformer.py is absent), so the model's last-commit date is "
          "used as an upper bound on when training could have ended. Anything BEFORE that date may be in-sample and must not "
          "be read as evidence of edge.\n")
        r = []
        if S_is: r.append(summary_row("before cutoff (possibly in-sample)", S_is))
        if S_oos: r.append(summary_row("after cutoff (out-of-sample)", S_oos))
        w(md_table(pd.DataFrame(r).set_index("scenario"), "{:.2f}"))

    sd = ctx.get("sigdiag") or {}
    w("\n## 2. Does the signal predict anything? (rule-independent)\n")
    if "ic_by_symbol" in sd:
        h = sd["horizon"]
        w(f"Spearman rank IC between the raw model output and forward log-return, per symbol. The model was trained on a 6-bar "
          f"(90 min) target. t-stats use n/h effective observations to respect overlapping returns. |IC| of 0.01-0.03 is typical of a "
          f"real but weak 15-minute signal; ~0 means no information.\n")
        w(md_table(sd["ic_by_symbol"], "{:.3f}"))
        w("\nMean IC across symbols by horizon (bars): " + ", ".join(f"{k}: {v:+.4f}" for k, v in sd["ic_mean"].items()) + "\n")
        if "ic_monthly_summary" in sd:
            m = sd["ic_monthly_summary"]
            w(f"\nMonthly IC at {h} bars: mean {m['mean']:+.4f}, std {_f(m['std'], '{:.4f}')}, positive in "
              f"{m['pct_positive']*100:.0f}% of {m['n_months']} months.\n")
        w(f"\n**Forward {h}-bar return by signal decile** (monotonicity = Spearman of decile vs return: {sd['decile_monotonicity']:+.2f}; "
          f"unconditional mean {sd['uncond_fwd_bps']:+.1f} bps, base rate up {sd['base_rate_up']*100:.1f}%)\n")
        w(md_table(sd["deciles"], "{:.3f}"))
        w(f"\n**Return when the entry condition's signal threshold is met (trend up):**\n")
        w(md_table(sd["thresholds"], "{:.3f}"))
        w(f"\nProbability calibration (the bot treats the output as a win probability in Kelly sizing). Brier score "
          f"{sd['brier']:.4f} vs {sd['brier_baseline']:.4f} for always predicting the base rate "
          f"({'better' if sd['brier'] < sd['brier_baseline'] else 'WORSE'} than the naive baseline).\n")
        w(md_table(sd["calibration"], "{:.3f}"))
        w(f"\nRegime occupancy (ATR% on 15m bars): " + ", ".join(f"{k} {v*100:.1f}%" for k, v in sd["regime_occupancy"].items())
          + f"; median ATR% {sd['atr_pct_median']:.2f}; trend-up {sd['trend_up_frac']*100:.0f}% of bars.\n")
    else:
        w("_Not enough data for signal diagnostics._\n")

    tr = res.trades
    w("\n## 3. Trade breakdown\n")
    if len(tr):
        by_reason = tr.groupby("reason").agg(n=("ret_net", "size"), mean_net_bps=("ret_net", lambda x: x.mean() * 1e4),
                                             win_pct=("ret_net", lambda x: (x > 0).mean() * 100), pnl_usd=("pnl_usd", "sum"),
                                             avg_hold_h=("hold_h", "mean"))
        w("**By exit reason**\n"); w(md_table(by_reason, "{:.2f}"))
        by_sym = tr.groupby("symbol").agg(n=("ret_net", "size"), mean_net_bps=("ret_net", lambda x: x.mean() * 1e4),
                                          win_pct=("ret_net", lambda x: (x > 0).mean() * 100), pnl_usd=("pnl_usd", "sum"))
        w("\n**By symbol**\n"); w(md_table(by_sym, "{:.2f}"))
        by_reg = tr.groupby("regime0").agg(n=("ret_net", "size"), mean_net_bps=("ret_net", lambda x: x.mean() * 1e4), pnl_usd=("pnl_usd", "sum"))
        w("\n**By regime at entry**\n"); w(md_table(by_reg, "{:.2f}"))
        mo = tr.groupby(tr["entry_ts"].dt.to_period("M")).agg(n=("ret_net", "size"), mean_net_bps=("ret_net", lambda x: x.mean() * 1e4), pnl_usd=("pnl_usd", "sum"))
        mo.index = mo.index.astype(str)
        w("\n**By month of entry**\n"); w(md_table(mo, "{:.2f}"))
        sb = tr.assign(bucket=pd.cut(tr["signal0"], [0, .5, .55, .6, .7, 1.0])).groupby("bucket", observed=True).agg(
            n=("ret_net", "size"), mean_net_bps=("ret_net", lambda x: x.mean() * 1e4), win_pct=("ret_net", lambda x: (x > 0).mean() * 100))
        sb.index = sb.index.astype(str)
        w("\n**By signal at entry**\n"); w(md_table(sb, "{:.2f}"))
    else:
        w("_No trades._\n")

    w("\n## 4. Benchmarks\n")
    if ctx.get("bench_table") is not None:
        w(md_table(ctx["bench_table"], "{:.2f}"))
    if ctx.get("alpha"):
        al = ctx["alpha"]
        w(f"\nRegression of strategy daily returns on the equal-weight basket (Newey-West t-stats): beta {al['beta']:.3f} "
          f"(t {al['t_beta']:.1f}), annualised alpha {al['alpha']*365*100:.2f}% (t {al['t_alpha']:.2f}), R2 {al['r2']:.2f}.\n")

    w("\n## 5. Null test: is the result better than the same machinery with a meaningless signal?\n")
    if ctx.get("null") is not None and len(ctx["null"]):
        nn = ctx["null"]
        act = S
        def pval(col, val):
            return (1 + (nn[col] >= val).sum()) / (1 + len(nn)) if np.isfinite(val) else float("nan")
        w(f"Each trial circularly shifts every symbol's signal by a random >=3-day offset (same distribution and autocorrelation, "
          f"no link to price) and re-runs everything: universe, trend filter, sizing, exits, costs. {len(nn)} trials.\n")
        t = pd.DataFrame([
            ["total return %", act.get("total_return", np.nan) * 100, nn["total_return"].mean() * 100, nn["total_return"].quantile(.95) * 100, pval("total_return", act.get("total_return", np.nan))],
            ["mean net bps/trade", act.get("mean_net_bps", np.nan), nn["mean_net_bps"].mean(), nn["mean_net_bps"].quantile(.95), pval("mean_net_bps", act.get("mean_net_bps", np.nan))],
            ["sharpe", act.get("sharpe", np.nan), nn["sharpe"].mean(), nn["sharpe"].quantile(.95), pval("sharpe", act.get("sharpe", np.nan))],
        ], columns=["metric", "actual", "null mean", "null 95th pct", "one-sided p"]).set_index("metric")
        w(md_table(t, "{:.3f}"))
        w(f"\nWith {len(nn)} trials the smallest attainable p-value is {1/(1+len(nn)):.3f}.\n")
    else:
        w("_Skipped (use --null-trials N, recommended >= 30)._\n")

    w("\n## 6. Cost and fill sensitivity\n")
    w("Same signals, same everything, different friction. Alpaca tier-1 crypto fees are 0.15% maker / 0.25% taker; "
      "the bot's marketable limits are takers.\n")
    w(md_table(ctx["cost_table"], "{:.2f}"))
    if ctx.get("sweep_table") is not None:
        w("\n### Parameter sweep (research only)\n")
        w(f"**{len(ctx['sweep_table'])} configurations were evaluated. Picking the best of them and trading it would be "
          f"data-mining: expect the winner to disappoint out of sample.**\n")
        w(md_table(ctx["sweep_table"], "{:.2f}"))

    w("\n## 7. Where capital actually goes\n")
    w(md_table(pd.DataFrame(sorted(res.counters.items()), columns=["event", "count"]), index=False))

    q = ctx.get("quirk")
    if q:
        w("\n## 7b. Live data-window check (hypothesis -- verify before acting on it)\n")
        w("data_feeds.py requests `limit=64` bars with no `start`. Alpaca documents the default `start` as the beginning of the "
          "current UTC day and the default order as ascending; alpaca-py 0.33.0 adds neither. If that is how the live API "
          "responds, the bot receives the FIRST 64 bars of today, not the latest 64. The share of cycles in which it would have "
          "usable data, by time of day:\n")
        w(md_table(q["avail"], "{:.0f}"))
        if q.get("summary"):
            w("\nSame strategy, same costs, same everything else -- only the data window differs:\n")
            w(md_table(pd.DataFrame([q["base"], q["summary"]]).set_index("scenario"), "{:.2f}"))
            c = q.get("counters", {})
            w(f"\nUnder the day-anchored window: {c.get('sell_unfilled', 0)} sell attempts could not fill (limit set from a stale price) "
              f"and {c.get('buy_unfilled', 0)} buy attempts could not fill.\n")
        w("\n**How to check in 30 seconds** (crypto data needs no key):\n\n```python\n" + VERIFY_SNIPPET + "\n```\n")

    w("\n## 8. Observations about live behaviour (computed from this run, not assumed)\n")
    for line in ctx["observations"]:
        w(f"- {line}")

    w("\n\n## 9. Assumptions and limitations\n")
    for line in ASSUMPTIONS:
        w(f"- {line}")
    return "\n".join(L) + "\n"



VERIFY_SNIPPET = """\
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
bars = CryptoHistoricalDataClient().get_crypto_bars(CryptoBarsRequest(
    symbol_or_symbols="BTC/USD", timeframe=TimeFrame(15, TimeFrameUnit.Minute), limit=64)).data["BTC/USD"]
print(len(bars), bars[0].timestamp, bars[-1].timestamp)
# Run it at e.g. 20:00 UTC. If bars[0] is today's 00:00 UTC and bars[-1] is ~15:45 UTC, the bot is trading on
# frozen data in the evening. If bars[-1] is within ~15 minutes of now, the intended behaviour holds."""

ASSUMPTIONS = [
    "**Orderbook whale filter is not modelled.** It vetoes buys when bid/ask depth < 0.65, and Alpaca does not provide historical L2. "
    "It can only remove trades, so the real bot trades somewhat less than this simulation.",
    "**Fills are modelled, not observed.** Decision at bar close, fill at the next bar's open +/- slippage, bounded by the bot's real "
    "limit prices (close*1.001 / close*0.999). `strict` mode cancels anything not marketable at the open; `touch` (default) also fills at "
    "the limit if the bar's range reaches it, approximating the bot's repeated re-submission of the same limit price within a bar. "
    "Real fills may be better or worse; see the cost table for the range.",
    "**Fees:** a flat taker rate on every fill (Alpaca tier 1: 0.25%), charged on the credited asset as Alpaca does. Account volume at "
    "this position size stays in tier 1. The bot's own DB does not record fees (alpaca-py's Order has no commission field), so the live "
    "P&L it reports may overstate net returns.",
    "**Decisions run once per 15-min bar close.** Live loops every ~40s but sees identical bars/prices within a bar; only the exact second "
    "at which a time-based threshold (1h/2h/4h) is crossed can differ.",
    "**Live window = last 64 completed bars** (the INTENDED behaviour of data_feeds.py's `limit=64`; see section 7b for evidence it may "
    "instead return the first 64 bars of the UTC day). Bars with zero trades are absent from Alpaca's feed, so illiquid alts can span more "
    "wall-clock time; the simulator reproduces this by indexing bars, not time.",
    "**Universe rotation** replicates scan_stable_assets (top 5 of the 10 trained symbols by the current UTC day's partial volume x last "
    "close, refreshed hourly). The 10-symbol pool itself is survivorship-biased, as config.py acknowledges.",
    "**Account:** cash-only, long-only, no margin. Buying power = cash. Equity is marked at the last completed bar's close.",
    "**Single run, single path.** Even a clean result is one realisation of a noisy process; confidence intervals here are bootstrap "
    "estimates under an i.i.d./block assumption and understate regime risk.",
    "**Not modelled:** API failures/rate limits, circuit breaker, DB state recovery, crash restarts, partial fills, exchange downtime, "
    "withdrawal/transfer effects, taxes.",
]


def observations(res: SimResult, S: dict, sd: dict, args, quirk=None) -> List[str]:
    out = []
    p = res.params
    cnt = res.counters
    if res.halted_at is not None:
        out.append(f"Kill-switch (`{p.killswitch}`) tripped on {res.halted_at:%Y-%m-%d}: {res.halt_reason}. In main_bot.py `start_equity` is set once "
                   f"and never reset, so DAILY_LOSS_LIMIT ({cfg.DAILY_LOSS_LIMIT}%) is really a permanent cumulative -3% stop-out from the start "
                   f"of the process, after which the bot flattens and exits until manually restarted. Re-run with `--killswitch daily` to see the "
                   f"behaviour a per-day limit would give, or `--killswitch off` for the unconstrained strategy.")
    out.append(f"Risk taper (-5%/-7% from start) and MAX_DRAWDOWN_STOP ({cfg.MAX_DRAWDOWN_STOP}%) are only reachable if the -3% kill-switch is "
               f"off or per-day; in `live` mode the -3% stop always fires first.")
    max_pv = cfg.BASE_RISK_PERCENT * cfg.MAX_OPEN_POSITIONS
    out.append(f"max_portfolio_value = equity x BASE_RISK_PERCENT x MAX_OPEN_POSITIONS = {max_pv*100:.0f}% of equity, so although "
               f"MAX_OPEN_POSITIONS is {cfg.MAX_OPEN_POSITIONS}, total exposure is capped at {max_pv*100:.0f}%. Observed exposure: average "
               f"{_f(S.get('avg_exposure', np.nan)*100, '{:.1f}')}%, maximum {_f(S.get('max_exposure', np.nan)*100, '{:.1f}')}%. "
               f"Cap hit {cnt.get('exposure_cap_hit', 0)} cycles ({cnt.get('forced_sells', 0)} forced sells); headroom blocks "
               f"{cnt.get('blocked_headroom', 0)}; swaps {cnt.get('swaps', 0)}.")
    if sd.get("regime_occupancy"):
        ro = sd["regime_occupancy"]
        out.append(f"Regime labels are computed from ATR% of 15-minute bars (median {sd['atr_pct_median']:.2f}%) but thresholded at 2% / 4% "
                   f"(quiet/normal/wild): here {ro['quiet']*100:.1f}% of bars are 'quiet', {ro['normal']*100:.1f}% 'normal', {ro['wild']*100:.1f}% 'wild'. "
                   f"If 'quiet' dominates, the regime logic is effectively a constant: buy threshold = BUY_SIGNAL-0.04, sell = SELL_SIGNAL+0.02, "
                   f"take-profit arm 1.5%, trailing stop pinned at its 1% floor (atr%/100 x 0.5 is far below it), and the ATR>6% / wild vetoes never fire.")
    thr_buy = res.params.buy_signal - 0.04
    thr_sell = res.params.sell_signal + 0.02
    out.append(f"In the 'quiet' regime the entry threshold ({thr_buy:.2f}) and weak-signal exit threshold ({thr_sell:.2f}) are "
               f"{'identical' if abs(thr_buy-thr_sell) < 1e-9 else 'only %.2f apart' % abs(thr_buy-thr_sell)}, so the signal exit has almost no hysteresis.")
    out.append(f"Stops are evaluated on bar closes only (no resting stop orders). Average worst intrabar excursion was "
               f"{_f(S.get('avg_mae_bps'), '{:.0f}')} bps versus the {cfg.STOP_LOSS_PCT*100:.0f}% nominal stop; gaps through the stop are real losses here.")
    out.append("data_feeds.get_clean_ohlcv_dataframe sends `limit=64` with no `start`. Per Alpaca's docs the default start is 00:00 UTC "
               "today with ascending order, which would return the first 64 bars of the day instead of the latest 64 (no data before "
               "08:00 UTC, frozen data after 16:00 UTC). This run assumed the intended latest-64 behaviour"
               + (" and compared against the day-anchored behaviour in section 7b." if quirk and quirk.get("summary") else
                  "; run with `--data-window both` to quantify the difference.")
               + " Unverified from the environment this was written in -- see the check in section 7b.")
    if cnt.get("buy_unfilled", 0) or cnt.get("sell_unfilled", 0):
        out.append(f"Unfilled limit orders under the `{p.fill_model}` fill model: {cnt.get('buy_unfilled', 0)} buys and {cnt.get('sell_unfilled', 0)} sells.")
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Live-parity backtest for the Grok Alpaca Apex bot.")
    ap.add_argument("--source", choices=["alpaca", "csv", "synthetic"], default="alpaca")
    ap.add_argument("--csv-dir", default="historical_data")
    ap.add_argument("--symbols", default=",".join(cfg.DYNAMIC_UNIVERSE_CANDIDATES))
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (UTC). Default: end - 120 days")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (UTC). Default: today")
    ap.add_argument("--cache-dir", default="bt_cache")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--model", default=cfg.MODEL_PATH)
    ap.add_argument("--equity", type=float, default=float(cfg.ACCOUNT_BASE))
    ap.add_argument("--fee-bps", type=float, default=25.0, help="per-fill fee; Alpaca tier-1 taker = 25, maker = 15")
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--fill-model", choices=["touch", "strict"], default="touch")
    ap.add_argument("--killswitch", choices=["live", "daily", "off"], default="live")
    ap.add_argument("--universe-mode", choices=["live", "trailing24h", "all"], default="live")
    ap.add_argument("--data-window", choices=["latest64", "day-anchored", "both"], default="latest64",
                    help="latest64 = what data_feeds.py intends (most recent 64 bars). day-anchored = what Alpaca's documented "
                         "default start (00:00 UTC today, ascending) would return for limit=64 with no start. 'both' runs the "
                         "primary analysis on latest64 and adds the day-anchored scenario for comparison.")
    ap.add_argument("--buy-signal", type=float, default=float(cfg.BUY_SIGNAL))
    ap.add_argument("--sell-signal", type=float, default=float(cfg.SELL_SIGNAL))
    ap.add_argument("--oos-start", default="auto", help="date, 'auto' (model's last git commit date) or 'none'")
    ap.add_argument("--null-trials", type=int, default=0)
    ap.add_argument("--sweep", action="store_true", help="also run the full cost grid and a BUY_SIGNAL sweep")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    t_wall = time.time()
    symbols = [s.strip() for s in a.symbols.split(",") if s.strip()]
    end = pd.Timestamp(a.end) if a.end else pd.Timestamp(datetime.now(timezone.utc).date())
    start = pd.Timestamp(a.start) if a.start else end - pd.Timedelta(days=120)
    warm = start - pd.Timedelta(days=3)       # bars before `start` only warm up the 64-bar windows
    out_dir = Path(a.out_dir or f"bt_results/{datetime.now():%Y%m%d_%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache_dir)

    print(f"== Loading data ({a.source}) {start:%Y-%m-%d} -> {end:%Y-%m-%d}")
    if a.source == "alpaca":
        bars = fetch_alpaca(symbols, warm, end, cache)
    elif a.source == "csv":
        bars = load_csv_dir(Path(a.csv_dir), symbols)
        bars = {s: d[(d.index >= warm) & (d.index < end)] for s, d in bars.items()}
    else:
        bars = make_synthetic(symbols, warm, end, seed=a.seed, gap_prob=[0.0, 0.0, 0.002, 0.004, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03][:len(symbols)] + [0.0] * max(0, len(symbols) - 10))
    bars = {s: d for s, d in bars.items() if len(d) >= WIN + 10}
    if not bars:
        print("No usable data.")
        return 2
    spacing = {s: float(np.median(np.diff(d.index.values).astype("timedelta64[s]").astype(float))) for s, d in bars.items()}
    bad = {s: v for s, v in spacing.items() if abs(v - BAR_SECONDS) > 1}
    if bad:
        print(f"ERROR: expected 15-minute bars, median spacing seen: {bad}")
        return 2
    for s, d in bars.items():
        full = pd.date_range(d.index[0], d.index[-1], freq="15min")
        print(f"  {s:9s} {len(d):>7,} bars  {d.index[0]:%Y-%m-%d} -> {d.index[-1]:%Y-%m-%d}  missing {(1 - len(d) / len(full)) * 100:5.2f}%")

    print("== Loading model / computing stage-1 signals (cached)")
    infer, model_sig = make_infer(a.model)
    modes = {"latest64": ["latest64"], "day-anchored": ["day-anchored"], "both": ["latest64", "day-anchored"]}[a.data_window]
    primary = modes[0]
    s1_modes: Dict[str, Dict[str, Stage1]] = {}
    fails: List[str] = []
    for mode in modes:
        s1_modes[mode] = {}
        for s_, d in bars.items():
            t0 = time.time()
            s1_modes[mode][s_] = stage1_cached(s_, d, infer, model_sig, cache, mode)
            print(f"  [{mode}] {s_:9s} valid bars {int(s1_modes[mode][s_].valid.sum()):>7,}  ({time.time() - t0:.1f}s)")
        if not a.no_verify:
            f_mode = verify_stage1_parity(bars, s1_modes[mode], infer, mode=mode)
            print(f"== Stage-1 parity vs repo's per-window functions [{mode}]:", "PASS" if not f_mode else f"FAIL ({len(f_mode)})")
            for x in f_mode[:10]:
                print("   ", x)
            fails += f_mode
    if fails:
        print("   Continuing, but treat results with suspicion.")
    s1 = s1_modes[primary]

    g = git_info(REPO_DIR)
    oos = None
    if a.oos_start == "auto" and g.get("model_last_commit_date"):
        oos = pd.Timestamp(g["model_last_commit_date"]).tz_convert("UTC").tz_localize(None).normalize()
    elif a.oos_start not in ("auto", "none"):
        oos = pd.Timestamp(a.oos_start)
    if oos is not None and not (start < oos < end):
        oos = None if oos >= end or oos <= start else oos

    params = Params(equity0=a.equity, fee_bps=a.fee_bps, slippage_bps=a.slippage_bps, fill_model=a.fill_model,
                    killswitch=a.killswitch, universe_mode=a.universe_mode, buy_signal=a.buy_signal,
                    sell_signal=a.sell_signal, start=start, end=end)
    md = MarketData(bars, s1, start, end, a.universe_mode, list(cfg.DYNAMIC_UNIVERSE_CANDIDATES))
    print(f"== Simulating {len(md.grid):,} cycles over {len(md.syms)} symbols")
    t0 = time.time()
    res = Simulator(md, params).run()
    print(f"   done in {time.time() - t0:.1f}s: {len(res.trades)} trades")

    S = summarize(res, a.seed)
    S_is = S_oos = None
    if oos is not None and len(res.equity):
        S_is = summarize(SimResult(res.trades[res.trades["entry_ts"] < oos] if len(res.trades) else res.trades,
                                   res.equity[res.equity.index < oos], res.counters, None, "", params), a.seed)
        S_oos = summarize(res, a.seed, t_from=oos)

    t0_, t1_ = res.equity.index[0], res.equity.index[-1]
    sigdiag = signal_diagnostics(bars, s1, t_from=t0_, t_to=t1_)

    bench = benchmark_daily(bars, list(md.cands), t0_, t1_)
    bench_rows, strat_dr = [], daily_returns(res.equity["equity"])
    bench_rows.append({"series": "Strategy", "total_ret_%": S.get("total_return", np.nan) * 100, "sharpe": S.get("sharpe", np.nan),
                       "max_dd_%": S.get("max_dd", np.nan) * 100})
    for name, r in bench.items():
        cum = (1 + r).cumprod()
        bench_rows.append({"series": f"{name} buy&hold", "total_ret_%": (cum.iloc[-1] - 1) * 100, "sharpe": sharpe(r.to_numpy()),
                           "max_dd_%": max_drawdown(cum.to_numpy())[0] * 100})
        if name == "EW basket":
            scaled = r * S.get("avg_exposure", 0.0)
            cs = (1 + scaled).cumprod()
            bench_rows.append({"series": f"EW basket x strategy avg exposure ({S.get('avg_exposure', 0)*100:.0f}%)",
                               "total_ret_%": (cs.iloc[-1] - 1) * 100, "sharpe": sharpe(scaled.to_numpy()),
                               "max_dd_%": max_drawdown(cs.to_numpy())[0] * 100})
    alpha = None
    if "EW basket" in bench:
        j = pd.concat([strat_dr, bench["EW basket"]], axis=1, join="inner").dropna()
        if len(j) > 30:
            alpha = ols_hac(j.iloc[:, 0].to_numpy(), j.iloc[:, 1].to_numpy())

    print("== Cost sensitivity")
    base_p = params
    cost_grid = [("frictionless (0 bps fee, 0 slip)", dict(fee_bps=0.0, slippage_bps=0.0)),
                 ("maker fee 15 bps, 0 slip", dict(fee_bps=15.0, slippage_bps=0.0)),
                 (f"BASE ({a.fee_bps:g} bps fee, {a.slippage_bps:g} bps slip, {a.fill_model})", {}),
                 ("strict fills (cancel non-marketable)", dict(fill_model="strict")),
                 ("stressed (40 bps fee, 8 bps slip)", dict(fee_bps=40.0, slippage_bps=8.0)),
                 ("wide spread (15 bps slip > the bot's 10 bps limit offset)", dict(slippage_bps=15.0)),
                 ("kill-switch off", dict(killswitch="off")),
                 ("kill-switch per-day", dict(killswitch="daily"))]
    cost_table = run_scenarios(md, base_p, cost_grid)
    sweep_table = None
    if a.sweep:
        print("== Parameter sweep")
        grid = [(f"fee {f:g} / slip {sl:g} / {fm}", dict(fee_bps=f, slippage_bps=sl, fill_model=fm))
                for f in (0, 15, 25, 40) for sl in (0, 5, 15) for fm in ("touch", "strict")]
        grid += [(f"BUY_SIGNAL {b:.2f}", dict(buy_signal=b)) for b in (0.51, 0.55, 0.58, 0.62, 0.66, 0.70)]
        sweep_table = run_scenarios(md, base_p, grid)

    quirk = None
    if "day-anchored" in s1_modes and primary != "day-anchored":
        print("== Day-anchored data-window scenario")
        md_day = MarketData(bars, s1_modes["day-anchored"], start, end, a.universe_mode, list(cfg.DYNAMIC_UNIVERSE_CANDIDATES))
        res_day = Simulator(md_day, params).run()
        S_day = summarize(res_day, a.seed)
        quirk = {"summary": summary_row("day-anchored window (documented API default)", S_day),
                 "base": summary_row("latest-64 window (intended)", S),
                 "avail": availability_by_hour(bars, s1_modes["day-anchored"]),
                 "counters": res_day.counters, "trades": res_day.trades, "S": S_day}
    elif primary == "day-anchored":
        quirk = {"avail": availability_by_hour(bars, s1)}

    null = None
    if a.null_trials > 0:
        print(f"== Null test ({a.null_trials} circular-shift trials)")
        null = run_null(md, s1, params, a.null_trials, a.seed)

    ctx = dict(args=a, res=res, summary=S, summary_is=S_is, summary_oos=S_oos, sigdiag=sigdiag, git=g, model_sig=model_sig,
               oos_start=oos, parity_fails=fails, symbols=list(bars), t0=t0_, t1=t1_, cost_table=cost_table,
               sweep_table=sweep_table, null=null, quirk=quirk, primary=primary, bench_table=pd.DataFrame(bench_rows).set_index("series"), alpha=alpha,
               observations=observations(res, S, sigdiag, a, quirk))
    report = build_report(ctx)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    res.trades.to_csv(out_dir / "trades.csv", index=False)
    res.equity.to_csv(out_dir / "equity.csv")
    cost_table.to_csv(out_dir / "cost_sensitivity.csv")
    if null is not None:
        null.to_csv(out_dir / "null_trials.csv")
    clean = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in S.items()}
    (out_dir / "summary.json").write_text(json.dumps({"summary": clean, "params": {k: str(v) for k, v in vars(params).items()},
                                                      "counters": res.counters, "git": g, "model": model_sig,
                                                      "halted_at": str(res.halted_at), "args": vars(a)}, indent=2, default=str))
    if not a.no_plots:
        make_plots(res, bench, out_dir, oos)
    print(f"\n== Report written to {out_dir}/report.md   ({time.time() - t_wall:.0f}s total)")
    print(f"   Net return {_f(S.get('total_return', np.nan) * 100)}%  |  trades {S.get('n_trades', 0)}  |  "
          f"mean net/trade {_f(S.get('mean_net_bps'))} bps  |  Sharpe {_f(S.get('sharpe'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
