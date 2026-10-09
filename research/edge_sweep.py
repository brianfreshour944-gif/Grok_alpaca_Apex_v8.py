"""Where (if anywhere) the 11-feature signal clears cost.

The incumbent shape is a 2h hold (8 x 15m bars) on Alpaca (50 bps round trip).
This loads the bars ONCE and sweeps the label horizon, reporting the purged
top-decile GROSS edge and the round-trip cost it would need to break even. If
the breakeven cost at the incumbent horizon is ~5 bps, no retrain on these
features can make the current shape profitable -- the horizon/venue must change.

Usage:
  python research/edge_sweep.py --csv-dir bt_cache --horizons 8,24,96,288,672
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oos_eval


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default="bt_cache")
    ap.add_argument("--horizons", default="8,24,96,288,672",
                    help="label horizons in 15m bars (8=2h,96=24h,672=7d)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--symbols", default=None)
    a = ap.parse_args(argv)
    syms = [s.strip() for s in a.symbols.split(",")] if a.symbols else None
    bars = oos_eval.load_bars(a.csv_dir, syms)
    print(f"Loaded {len(bars)} symbols: {', '.join(sorted(bars))}\n")
    print(f"{'horizon':>8} {'label':>6} {'n':>9} {'AUC':>6} {'IC':>8} "
          f"{'top_hit':>8} {'base':>6} {'GROSS bps':>10} {'breakeven rt':>12}")
    for h in [int(x) for x in a.horizons.split(",")]:
        D = oos_eval.build_dataset(bars, h)
        wf = oos_eval.walk_forward(D, a.folds, h, cost=0.0, purge=True)
        gross = wf["top_gross_bps"].mean()
        label = f"{h*15//60}h" if h * 15 >= 60 else f"{h*15}m"
        print(f"{h:>8} {label:>6} {len(D):>9,} {wf['auc'].mean():>6.4f} "
              f"{wf['ic'].mean():>+8.4f} {wf['top_hit'].mean():>8.3f} "
              f"{wf['base_hit'].mean():>6.3f} {gross:>+10.1f} {gross:>+10.1f} bps")
    print("\nbreakeven rt = the round-trip fee at which the top decile turns "
          "flat (equal to GROSS). Alpaca = 50 bps; Binance perp taker = 10, "
          "maker ~4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
