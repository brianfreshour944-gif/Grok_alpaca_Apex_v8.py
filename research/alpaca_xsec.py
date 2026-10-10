"""Is the cross-sectional signal expressible on ALPACA (spot crypto only)?

Alpaca crypto cannot be shorted (its asset objects report shortable=false,
marginable=false), so the dollar-neutral book in xsec_momentum.py CANNOT be
placed there. The only crypto-expressible form is LONG-ONLY: hold the top-k
relative winners, cash otherwise. That drops beta-neutrality, so the honest
question becomes: does the momentum *ranking* beat simply holding the market?

This runs, restricted to Alpaca's actual crypto universe (ALPACA_TRADEABLE):
  * long-only top-k momentum vs equal-weight buy & hold,
  * sub-periods (the same windows as the rest of the research),
  * and the dollar-neutral reference (for context only -- not placeable).

Alpaca crypto facts this respects: spot only; market/limit/stop-limit; tif
gtc/ioc; fractional qty; min order notional ~$1. No margin, no borrow.

Data: okx_daily/<SYM>_1D.csv (research proxies for the USD pairs).

Usage:
  python research/alpaca_xsec.py --lookback 30 --k 3 --hold 7
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from donchian_bot import ALPACA_TRADEABLE
from research.xsec_momentum import (
    backtest_xsec,
    backtest_xsec_longonly,
    load_close,
)


def sub(close, lo, hi):
    return close[(close.index >= pd.Timestamp(lo, tz="UTC"))
                 & (close.index < pd.Timestamp(hi, tz="UTC"))]


def row(name, r, bh=None):
    b = f"  B&H {bh['total_return']*100:>+7.1f}%" if bh is not None else ""
    print(f"  {name:<22} tot {r['total_return']*100:>+8.1f}%  sharpe {r['sharpe']:>+5.2f}  "
          f"maxDD {r['max_drawdown']*100:>+6.1f}%  turn/day {r['avg_turnover_per_day']:.3f}{b}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="okx_daily")
    ap.add_argument("--lookback", type=int, default=30)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--hold", type=int, default=7)
    ap.add_argument("--fee-bps", type=float, default=25.0)
    a = ap.parse_args(argv)

    close = load_close(a.cache)
    tradeable = [s for s in close.columns if s in ALPACA_TRADEABLE]
    print(f"Alpaca-tradeable names in the panel ({len(tradeable)}): "
          f"{', '.join(tradeable)}")
    print("  mapped pairs: " + ", ".join(f"{s}->{ALPACA_TRADEABLE[s]}" for s in tradeable))

    # common-history subset of the tradeable names (fair long/short comparison)
    sub_close = close[tradeable].dropna(axis=1)
    print(f"  common-history subset ({len(sub_close.columns)}): "
          f"{', '.join(sub_close.columns)}\n")

    for label, data in [("TRADEABLE (common history)", sub_close)]:
        print(f"{label}: {data.index.min().date()} -> {data.index.max().date()}")
        eqw = data.pct_change().mean(axis=1).fillna(0.0)
        bh = {"total_return": float((1 + eqw).prod() - 1),
              "sharpe": float(eqw.mean() / eqw.std(ddof=1) * np.sqrt(365))}
        print(f"  EQUAL-WEIGHT BUY&HOLD    tot {bh['total_return']*100:>+8.1f}%  "
              f"sharpe {bh['sharpe']:>+5.2f}")
        for h in (3, 7, 14):
            r = backtest_xsec_longonly(data, a.lookback, a.k, h, a.fee_bps)
            row(f"longonly top{a.k} hold{h}", r, bh)
        # dollar-neutral reference -- NOT placeable on Alpaca
        n = backtest_xsec(data, a.lookback, a.k, a.hold, a.fee_bps)
        print("  (reference) dollar-neutral (NOT placeable on Alpaca):")
        row("neutral hold7", n)

        print("\n  sub-periods (longonly hold=7 vs buy&hold):")
        for lo, hi, lab in [("2020-06-01", "2023-01-01", "2020-22"),
                            ("2023-01-01", "2025-01-01", "2023-24"),
                            ("2025-01-01", "2026-11-01", "2025-26")]:
            s = sub(data, lo, hi)
            if len(s) < a.lookback + 20:
                continue
            r = backtest_xsec_longonly(s, a.lookback, a.k, 7, a.fee_bps)
            e = s.pct_change().mean(axis=1).fillna(0.0)
            bhr = float((1 + e).prod() - 1)
            print(f"    {lab}: momentum tot {r['total_return']*100:>+8.1f}%  "
                  f"sharpe {r['sharpe']:>+5.2f}  |  B&H {bhr*100:>+8.1f}%")

    print("\nALPACA CONSTRAINTS "
          "(https://docs.alpaca.markets/us/docs/crypto-trading):")
    print("  * crypto is SPOT ONLY: shortable=false, marginable=false -> no short leg")
    print("  * so only the long-only tilt is placeable; it keeps full market beta")
    print("  * market/limit/stop-limit, tif gtc|ioc, fractional qty, min notional ~$1")
    print("  * 24/7 trading; no PDT/margin rules for crypto")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
