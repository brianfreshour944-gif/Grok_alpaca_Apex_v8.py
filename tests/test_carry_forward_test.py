# tests/test_carry_forward_test.py — forward-test ledger invariants.
import numpy as np
import pandas as pd

import research.basis_strategy as bs
import research.carry_forward_test as ft


def _panel(n_days=10):
    rng = np.random.default_rng(0)
    rows = []
    for d in range(n_days):
        for h in (0, 8, 16):
            ts = pd.Timestamp("2026-06-01", tz="UTC") + pd.Timedelta(days=d, hours=h)
            f = rng.normal(0, 0.0003, 6)
            for i, s in enumerate("ABCDEF"):
                rows.append({"ts": ts, "sym": s, "funding": f[i], "funding_fwd": f[i],
                             "fwd_perp": rng.normal(0, 0.01), "fwd_spot": rng.normal(0, 0.01)})
    P = pd.DataFrame(rows)
    g = P.groupby("ts")["funding"]
    P["funding_xs"] = (P["funding"] - g.transform("mean")) / g.transform("std")
    return P.sort_values(["ts", "sym"]).reset_index(drop=True)


def test_ledger_net_matches_backtest():
    # The ledger must never diverge from the backtest it records.
    P = _panel()
    sub = P[P["ts"] >= pd.Timestamp("2026-06-01", tz="UTC")]
    res = bs.run_basis(sub, 2, 3, 5.0, 10.0, 2.0, 1.0)
    df = ft.build_ledger(P, "2026-06-01", k=2, hold=3)
    assert np.isclose(df["net_bps"].sum(), res["net"].sum(), atol=1e-6)


def test_ledger_cumulative_is_cumsum():
    df = ft.build_ledger(_panel(), "2026-06-01", k=2, hold=3)
    assert len(df) > 0
    assert np.allclose(df["cum_net_bps"].to_numpy(), df["net_bps"].cumsum().to_numpy())


def test_ledger_books_come_from_shared_selection():
    P = _panel()
    sub = P[P["ts"] >= pd.Timestamp("2026-06-01", tz="UTC")]
    expected = {",".join(s) for _, s in bs.select_legs(sub, 2)}
    df = ft.build_ledger(P, "2026-06-01", k=2, hold=3)
    assert set(df["book"]) <= expected
    assert all(len(b.split(",")) == 2 for b in df["book"])


def test_update_ledger_is_idempotent(tmp_path):
    df = ft.build_ledger(_panel(), "2026-06-01", k=2, hold=3)
    path = str(tmp_path / "ledger.csv")
    assert ft.update_ledger(path, df) == len(df)
    assert ft.update_ledger(path, df) == 0            # re-run adds no days
    again = pd.read_csv(path, index_col=0)
    assert len(again) == len(df)
