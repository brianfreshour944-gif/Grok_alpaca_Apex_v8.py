# tests/test_carry_strategy.py — funding-carry invariants (pure numpy/pandas).
import numpy as np
import pandas as pd

import research.carry_strategy as cs


def _panel(rows):
    """rows: list of dicts with ts, sym, funding, funding_fwd, fwd."""
    P = pd.DataFrame(rows)
    P["ts"] = pd.to_datetime(P["ts"], utc=True)
    g = P.groupby("ts")["funding"]
    P["funding_xs"] = (P["funding"] - g.transform("mean")) / g.transform("std")
    return P.sort_values(["ts", "sym"]).reset_index(drop=True)


def test_short_receives_positive_funding():
    # A has the highest funding -> shorted -> book should earn positive carry.
    rows = [{"ts": "2024-01-01", "sym": s, "funding": f, "funding_fwd": f, "fwd": 0.0}
            for s, f in zip("ABCD", [0.003, 0.001, 0.0005, 0.0001])]
    P = _panel(rows)
    res = cs.run_carry(P, k=1, cost_bps=0.0)
    assert res["carry"].iloc[0] > 0
    # A short (funding +30 bps received), D long (funding +1 bps paid) -> ~+29 bps
    assert 25 < res["carry"].iloc[0] < 35


def test_earned_carry_comes_from_forward_funding():
    # Selection uses the (observable) same-bar funding, but the cash earned must
    # come from funding_fwd. If every symbol's forward funding is identical, the
    # differential a neutral book can collect is zero -> carry must be ~0, even
    # though the book still shorts the highest same-bar funding name.
    base = [{"ts": "2024-01-01", "sym": s, "funding": f, "fwd": 0.0}
            for s, f in zip("ABCD", [0.003, 0.001, 0.0005, 0.0001])]
    same = _panel([{**r, "funding_fwd": r["funding"]} for r in base])
    flat = _panel([{**r, "funding_fwd": 0.001} for r in base])
    assert cs.run_carry(same, k=1, cost_bps=0.0)["carry"].iloc[0] > 20
    assert abs(cs.run_carry(flat, k=1, cost_bps=0.0)["carry"].iloc[0]) < 1e-9


def test_longer_hold_reduces_turnover():
    rng = np.random.default_rng(0)
    rows = []
    for t in range(60):
        ts = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=t)
        for s in "ABCD":
            f = rng.normal(0, 0.0005)
            rows.append({"ts": ts, "sym": s,
                         "funding": f, "funding_fwd": f, "fwd": rng.normal(0, 0.01)})
    P = _panel(rows)
    t1 = cs.run_carry(P, k=1, cost_bps=5.0, hold=1)["turnover"].sum()
    t10 = cs.run_carry(P, k=1, cost_bps=5.0, hold=10)["turnover"].sum()
    assert t10 < t1


def test_cost_is_monotone():
    rng = np.random.default_rng(1)
    rows = []
    for t in range(40):
        ts = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=t)
        for s in "ABCD":
            f = rng.normal(0, 0.0005)
            rows.append({"ts": ts, "sym": s,
                         "funding": f, "funding_fwd": f, "fwd": rng.normal(0, 0.01)})
    P = _panel(rows)
    free = cs.run_carry(P, k=1, cost_bps=0.0, hold=3)["net"].mean()
    dear = cs.run_carry(P, k=1, cost_bps=20.0, hold=3)["net"].mean()
    assert dear < free
