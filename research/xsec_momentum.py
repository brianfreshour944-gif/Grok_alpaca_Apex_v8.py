"""A different signal SOURCE: cross-sectional relative strength (market-neutral).

Everything tested so far is a TIME-SERIES rule on one price series (Donchian
breakout, ML direction, funding timing). This tests a different mechanism: rank
the whole universe each day and hold the RELATIVE winners against the relative
losers. The book is dollar-neutral, so the market's direction cancels and the
only thing traded is cross-sectional dispersion — a beta-free bet that a
time-series breakout cannot express.

Honest by construction:
  * No look-ahead: ranks at bar t use closes up to t; the book earns t+1.
  * Weights drift between rebalances and costs are charged on the traded
    notional at every rebalance (turnover is the whole game here).
  * A random-ranking NULL gives the Sharpe distribution you'd get from noise,
    so the reported t-stat is against a real null, not zero.
  * Beta of the L/S return vs the equal-weight market is reported; a real
    market-neutral edge should be ~0.

Data: okx_daily/<SYM>_1D.csv (15 symbols, 2020-06 -> 2026-10).

Usage:
  python research/xsec_momentum.py --lookback 30 --k 3 --hold 7
  python research/xsec_momentum.py --scan        # lookback/hold grid, net
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_close(cache: str) -> pd.DataFrame:
    """Wide close matrix (dates x symbols) from the daily cache."""
    cols = {}
    for fp in sorted(glob.glob(os.path.join(cache, "*_1D.csv"))):
        sym = os.path.basename(fp).replace("_1D.csv", "")
        df = pd.read_csv(fp)
        ts = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
        s = pd.Series(df["close"].astype(float).to_numpy(), index=ts).sort_index()
        cols[sym] = s[~s.index.duplicated()]
    close = pd.DataFrame(cols).sort_index()
    return close


def xsec_targets(mom_row: pd.Series, k: int, gross: float = 1.0) -> pd.Series:
    """Dollar-neutral weights: +gross/2 over the top-k, -gross/2 over the
    bottom-k. Fewer than 2k live symbols -> all flat (cannot form the spread)."""
    valid = mom_row.dropna()
    if len(valid) < 2 * k:
        return pd.Series(0.0, index=mom_row.index)
    ranked = valid.sort_values()
    w = pd.Series(0.0, index=mom_row.index)
    w[ranked.index[-k:]] = gross / 2.0 / k      # relative winners
    w[ranked.index[:k]] = -gross / 2.0 / k      # relative losers
    return w


def backtest_xsec(close: pd.DataFrame, lookback: int, k: int, hold: int,
                  fee_bps: float = 25.0, reverse: bool = False,
                  shuffle_seed: int | None = None) -> dict:
    """Dollar-neutral cross-sectional book with weight drift + turnover costs.

    `reverse=True` flips to cross-sectional REVERSAL (short the winners).
    `shuffle_seed` replaces the ranking with a random one (the null)."""
    ret = close.pct_change()
    mom = close / close.shift(lookback) - 1.0
    if reverse:
        mom = -mom
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None

    syms = close.columns
    dates = close.index
    n = len(dates)
    w = pd.Series(0.0, index=syms)
    equity = 1.0
    eq = np.ones(n)
    daily = np.zeros(n)
    contrib = pd.Series(0.0, index=syms)      # per-symbol PnL attribution
    traded_total = 0.0
    n_reb = 0
    last_reb = None

    for i in range(lookback, n - 1):
        due = last_reb is None or (i - last_reb) >= hold
        if due:
            if rng is not None:
                vals = mom.iloc[i].dropna()
                shuffled = pd.Series(rng.permutation(vals.to_numpy()), index=vals.index)
                target = xsec_targets(shuffled, k)
            else:
                target = xsec_targets(mom.iloc[i], k)
            traded = float((target - w).abs().sum())
            equity *= (1.0 - traded * fee_bps / 1e4)
            traded_total += traded
            n_reb += 1
            w = target
            last_reb = i
        r = ret.iloc[i + 1].fillna(0.0)
        leg = w * r
        contrib += leg
        port = float(leg.sum())
        equity *= (1.0 + port)
        daily[i + 1] = port
        eq[i + 1] = equity
        w = w * (1.0 + r) / (1.0 + port) if (1.0 + port) != 0 else w

    eq = eq[:n]
    d = daily[lookback + 1:]
    return {
        "equity": pd.Series(eq, index=dates),
        "daily": pd.Series(daily, index=dates),
        "contrib": contrib,
        "final_equity": float(eq[-1]),
        "total_return": float(eq[-1] - 1.0),
        "sharpe": _sharpe(d),
        "tstat": _tstat(d),
        "max_drawdown": _maxdd(eq),
        "avg_turnover_per_day": traded_total / max(n - lookback - 1, 1),
        "n_rebalances": n_reb,
        "exposure": 1.0,
    }


def _sharpe(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(365))


def _tstat(r: np.ndarray) -> float:
    r = r[np.isfinite(r)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / (r.std(ddof=1) / np.sqrt(r.size)))


def _maxdd(eq: np.ndarray) -> float:
    peak = np.maximum.accumulate(eq)
    return float(np.min(eq / peak - 1.0))


def beta_vs_market(close: pd.DataFrame, daily: pd.Series) -> float:
    mkt = close.pct_change().mean(axis=1).reindex(daily.index).fillna(0.0)
    x = np.vstack([np.ones(len(mkt)), mkt.to_numpy()])
    coef, *_ = np.linalg.lstsq(x.T, daily.to_numpy(), rcond=None)
    return float(coef[1])


def year_table(close, lookback, k, hold, fee_bps, reverse=False):
    print(f"\nPER-YEAR L/S total % (lookback={lookback} k={k} hold={hold} "
          f"fee={fee_bps}bps reverse={reverse})")
    print("  {:<6} {:>8} {:>8} {:>7}".format("year", "gross%", "net%", "sharpe"))
    for y in range(2021, 2027):
        lo, hi = pd.Timestamp(f"{y}-01-01", tz="UTC"), pd.Timestamp(f"{y+1}-01-01", tz="UTC")
        sub = close[(close.index >= lo) & (close.index < hi)]
        if len(sub) < lookback + hold + 5:
            continue
        g = backtest_xsec(sub, lookback, k, hold, fee_bps=0.0, reverse=reverse)
        n = backtest_xsec(sub, lookback, k, hold, fee_bps=fee_bps, reverse=reverse)
        print("  {:<6} {:>+8.1f} {:>+8.1f} {:>7.2f}".format(
            y, g["total_return"] * 100, n["total_return"] * 100, n["sharpe"]))


def null_test(close, lookback, k, hold, fee_bps, reverse=False, trials=200):
    real = backtest_xsec(close, lookback, k, hold, fee_bps, reverse=reverse)
    nulls = [backtest_xsec(close, lookback, k, hold, fee_bps, reverse=reverse,
                           shuffle_seed=s)["sharpe"] for s in range(trials)]
    nulls = np.array(nulls)
    p = float((nulls >= real["sharpe"]).mean())
    print(f"\nNULL (random ranking, {trials} trials): mean Sharpe {nulls.mean():+.2f} "
          f"sd {nulls.std():.2f}; real {real['sharpe']:+.2f} -> p={p:.3f}")
    return real, nulls


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="okx_daily")
    ap.add_argument("--lookback", type=int, default=30)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--hold", type=int, default=7)
    ap.add_argument("--fee-bps", type=float, default=25.0)
    ap.add_argument("--reverse", action="store_true", help="cross-sectional reversal")
    ap.add_argument("--scan", action="store_true")
    a = ap.parse_args(argv)

    close = load_close(a.cache)
    if close.empty:
        print(f"no data in {a.cache}/ — run research/fetch_okx_daily.py first")
        return 1
    print(f"loaded {close.shape[1]} symbols x {close.shape[0]} days, "
          f"{close.index.min().date()} -> {close.index.max().date()}")

    for fee in (a.fee_bps, 0.0):
        r = backtest_xsec(close, a.lookback, a.k, a.hold, fee, reverse=a.reverse)
        beta = beta_vs_market(close, r["daily"])
        print(f"\n{'GROSS' if fee == 0 else 'NET'} (lookback={a.lookback} k={a.k} "
              f"hold={a.hold} fee={fee}bps reverse={a.reverse})")
        print(f"  total {r['total_return']*100:+.1f}%  sharpe {r['sharpe']:+.2f}  "
              f"t {r['tstat']:+.2f}  maxDD {r['max_drawdown']*100:+.1f}%  "
              f"turn/day {r['avg_turnover_per_day']:.3f}  beta {beta:+.2f}")

    year_table(close, a.lookback, a.k, a.hold, a.fee_bps, reverse=a.reverse)
    null_test(close, a.lookback, a.k, a.hold, a.fee_bps, reverse=a.reverse)

    if a.scan:
        print(f"\nSCAN net total % (fee {a.fee_bps}bps): lookback x hold, k={a.k}")
        holds = (1, 3, 7, 14, 30)
        print("  look\\hold " + " ".join(f"{h:>7d}" for h in holds))
        for lb in (10, 20, 30, 60, 90):
            row = []
            for h in holds:
                r = backtest_xsec(close, lb, a.k, h, a.fee_bps, reverse=a.reverse)
                row.append(f"{r['total_return']*100:>+7.1f}")
            print(f"  {lb:>9d} " + " ".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
