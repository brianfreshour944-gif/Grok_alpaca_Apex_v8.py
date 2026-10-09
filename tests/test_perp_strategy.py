# tests/test_perp_strategy.py — book construction invariants for the perp study.
# Pure numpy; no network, no data files.
import numpy as np
import pandas as pd

import research.perp_strategy as ps


def _panel(fwd_by_ts):
    """fwd_by_ts: list of dicts {sym: fwd} -> panel with trivial scores."""
    rows = []
    for i, d in enumerate(fwd_by_ts):
        for sym, f in d.items():
            rows.append({"ts": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=i),
                         "sym": sym, "fwd": f, "score": 0.0})
    return pd.DataFrame(rows)


def test_dollar_neutral_book_is_immune_to_common_shift():
    # add a constant to every forward return; a zero-sum weight book must not care
    base = [{"A": 0.01, "B": 0.02, "C": -0.01, "D": 0.005},
            {"A": -0.02, "B": 0.01, "C": 0.03, "D": 0.0}]
    shifted = [{k: v + 0.5 for k, v in d.items()} for d in base]
    out = []
    for data in (base, shifted):
        P = _panel(data)
        P["score"] = [1.0, 0.5, -0.5, -1.0] * len(data)   # A long, D short
        b, i, f, n = ps._group_index(P)
        out.append(ps._run_book(P["score"].to_numpy(float), b, i, f, n, 1, 1, 0.0))
    np.testing.assert_allclose(out[0], out[1], atol=1e-9)


def test_book_goes_long_top_and_short_bottom():
    P = _panel([{"A": 0.10, "B": 0.02, "C": -0.02, "D": -0.10}])
    P["score"] = [3.0, 2.0, 1.0, 0.0]      # A highest -> long, D lowest -> short
    b, i, f, n = ps._group_index(P)
    net = ps._run_book(P["score"].to_numpy(float), b, i, f, n, 1, 1, 0.0)
    # long A (+10%) short D (-10%) -> +20% = 2000 bps
    assert net[0] == 2000.0


def test_cost_reduces_net_and_zero_cost_equals_gross():
    P = _panel([{"A": 0.05, "B": 0.0, "C": 0.0, "D": -0.05}])
    P["score"] = [3.0, 2.0, 1.0, 0.0]
    b, i, f, n = ps._group_index(P)
    gross = ps._run_book(P["score"].to_numpy(float), b, i, f, n, 1, 1, 0.0)
    net = ps._run_book(P["score"].to_numpy(float), b, i, f, n, 1, 1, 5.0)
    assert net[0] < gross[0]
