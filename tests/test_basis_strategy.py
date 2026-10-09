# tests/test_basis_strategy.py — delta-neutral cash-and-carry invariants.
import numpy as np
import pandas as pd

import research.basis_strategy as bs


def _panel(rows):
    P = pd.DataFrame(rows)
    P["ts"] = pd.to_datetime(P["ts"], utc=True)
    g = P.groupby("ts")["funding"]
    P["funding_xs"] = (P["funding"] - g.transform("mean")) / g.transform("std")
    return P.sort_values(["ts", "sym"]).reset_index(drop=True)


def _row(ts, sym, funding, fwd_perp, fwd_spot, funding_fwd=None):
    return {"ts": ts, "sym": sym, "funding": funding,
            "funding_fwd": funding if funding_fwd is None else funding_fwd,
            "fwd_perp": fwd_perp, "fwd_spot": fwd_spot}


def test_hedge_cancels_price_when_legs_match():
    # spot and perp move identically -> long spot + short perp price term = 0
    rows = [_row("2024-01-01", s, f, 0.05, 0.05)
            for s, f in zip("ABCD", [0.003, 0.001, 0.0005, 0.0001])]
    res = bs.run_basis(_panel(rows), k=1, perp_cost=0.0, spot_cost=0.0)
    assert abs(res["price"].iloc[0]) < 1e-9


def test_short_perp_receives_positive_funding():
    rows = [_row("2024-01-01", s, f, 0.0, 0.0)
            for s, f in zip("ABCD", [0.003, 0.001, 0.0005, 0.0001])]
    res = bs.run_basis(_panel(rows), k=1, perp_cost=0.0, spot_cost=0.0)
    # A has highest funding -> shorted -> receives ~+30 bps, D long pays ~1 bps
    assert res["funding"].iloc[0] > 0


def test_price_term_tracks_basis_change():
    # perp rises 1% more than spot -> short perp / long spot loses 1% = -100 bps
    rows = [_row("2024-01-01", s, 0.0, 0.01, 0.0) for s in "ABCD"]
    res = bs.run_basis(_panel(rows), k=1, perp_cost=0.0, spot_cost=0.0)
    assert res["price"].iloc[0] < 0


def test_both_legs_cost_charged():
    rows = []
    for t in range(40):
        ts = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=8 * t)
        for s in "ABCD":
            f = np.random.default_rng(t).normal(0, 0.0005)
            rows.append(_row(ts, s, f, 0.01, 0.01))
    P = _panel(rows)
    free = bs.run_basis(P, k=1, hold=3, perp_cost=0.0, spot_cost=0.0)["net"].mean()
    dear = bs.run_basis(P, k=1, hold=3, perp_cost=5.0, spot_cost=10.0)["net"].mean()
    assert dear < free
