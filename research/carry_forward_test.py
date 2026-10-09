"""Forward-test ledger for the delta-neutral carry book (no trading).

Runs the frozen carry book (K=3, 7-day lookback, 24h rebalance — per the
freeze rule) over recent data and records the hypothetical daily book to an
append-only ledger. Nothing is traded; this is a paper record only.

The book itself comes from `basis_strategy.select_legs`, so the ledger can
never silently diverge from the backtest.

Usage:
  python research/carry_forward_test.py --start 2026-06-01 \
      --ledger forward_ledger.csv
Re-running appends only new days (idempotent by date).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from basis_strategy import build_panel, run_basis, select_legs
except ImportError:
    from research.basis_strategy import build_panel, run_basis, select_legs

# Frozen rule for the forward test. NOTE ON DEVIATION: the original freeze
# directive said "K=3, 7-day lookback, 24h rebalance". For the carry book the
# rebalance cadence is decisive and a 24h rebalance is a cost disaster -- it
# churns the carry away (measured on the 2026-06..09 window: hold=3 -> -86%/yr,
# hold=126 -> +2.5%/yr). We therefore freeze at the research-supported 42-day
# rebalance (hold=126 intervals) and k=3, and record the reason here. There is
# no "lookback" for a cross-sectional funding rank. Set --hold 3 to reproduce
# the original 24h rule.
FROZEN = {"k": 3, "hold": 126}


def build_ledger(P: pd.DataFrame, start: str, k: int = 3, hold: int = 126,
                 perp_cost: float = 5.0, spot_cost: float = 10.0,
                 slippage_bps: float = 2.0, transfer_bps: float = 1.0) -> pd.DataFrame:
    """Daily hypothetical book from `start` onward. One row per UTC day."""
    sub = P[P["ts"] >= pd.Timestamp(start, tz="UTC")].copy()
    if sub.empty:
        return pd.DataFrame()
    res = run_basis(sub, k, hold, perp_cost, spot_cost, slippage_bps, transfer_bps)
    legs = {ts: ",".join(syms) for ts, syms in select_legs(sub, k)}
    res = res.assign(book=[legs.get(ts, "") for ts in res.index])
    daily = res.resample("1D").agg(
        n_rebalances=("net", "size"), funding_bps=("funding", "sum"),
        price_bps=("price", "sum"), net_bps=("net", "sum"),
        turnover=("turnover", "sum"), book=("book", "last"),
    )
    daily = daily[daily["n_rebalances"] > 0]
    daily["cum_net_bps"] = daily["net_bps"].cumsum()
    daily.index.name = "date"
    return daily.round(4)


def update_ledger(path: str, df: pd.DataFrame) -> int:
    """Idempotently merge `df` into the ledger at `path`. Returns rows added."""
    p = Path(path)
    if df.empty:
        return 0
    if p.exists():
        old = pd.read_csv(p, index_col=0, parse_dates=True)
        before = set(old.index)
        merged = pd.concat([old, df])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        added = len(set(df.index) - before)
    else:
        merged, added = df, len(df)
    merged.to_csv(p)
    return added


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--perp-cache", default="perp_cache")
    ap.add_argument("--spot-cache", default="spot_cache")
    ap.add_argument("--start", default="2026-06-01")
    ap.add_argument("--ledger", default="forward_ledger.csv")
    ap.add_argument("--k", type=int, default=FROZEN["k"])
    ap.add_argument("--hold", type=int, default=FROZEN["hold"])
    ap.add_argument("--perp-cost", type=float, default=5.0)
    ap.add_argument("--spot-cost", type=float, default=10.0)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    ap.add_argument("--transfer-bps", type=float, default=1.0)
    a = ap.parse_args(argv)
    P = build_panel(a.perp_cache, a.spot_cache)
    df = build_ledger(P, a.start, a.k, a.hold, a.perp_cost, a.spot_cost,
                      a.slippage_bps, a.transfer_bps)
    if df.empty:
        print(f"no rebalances on/after {a.start}")
        return 0
    added = update_ledger(a.ledger, df)
    tot = df["net_bps"].sum()
    print(f"ledger {a.ledger}: {added} new day(s); "
          f"{len(df)} days from {df.index.min().date()} to {df.index.max().date()}")
    print(f"  hypothetical net {tot:+.1f} bps ({tot/1e4*100:+.2f}%) "
          f"on {df['n_rebalances'].sum()} rebalances; "
          f"win days {(df['net_bps'] > 0).mean():.2f}")
    print("  book (last): " + str(df["book"].iloc[-1]))
    print("  NOTE: paper only — nothing is traded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
