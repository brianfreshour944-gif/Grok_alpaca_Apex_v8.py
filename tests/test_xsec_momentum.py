"""Unit tests for the cross-sectional momentum research module."""
import numpy as np
import pandas as pd
import pytest

import research.xsec_momentum as xs


def _close(n=400, m=6, seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    cols = {}
    for j in range(m):
        r = rng.normal(drift + j * 0.001, 0.03, n)
        cols[f"S{j}"] = 100 * np.cumprod(1 + r)
    return pd.DataFrame(cols, index=idx)


def test_weights_are_dollar_neutral_and_unit_gross():
    row = pd.Series({"A": 0.1, "B": 0.2, "C": -0.1, "D": 0.05, "E": -0.3, "F": 0.3})
    w = xs.xsec_targets(row, 2)
    assert w.sum() == pytest.approx(0.0, abs=1e-12)
    assert w.abs().sum() == pytest.approx(1.0, abs=1e-12)
    # top-2 long, bottom-2 short
    assert set(w[w > 0].index) == {"B", "F"}
    assert set(w[w < 0].index) == {"C", "E"}


def test_weights_flat_when_universe_too_small():
    w = xs.xsec_targets(pd.Series({"A": 1.0, "B": 2.0, "C": 3.0}), 2)
    assert (w == 0).all()


def test_weights_ignore_nan_symbols():
    row = pd.Series({"A": 0.1, "B": np.nan, "C": 0.2, "D": -0.1, "E": -0.2})
    w = xs.xsec_targets(row, 2)
    assert w["B"] == 0.0
    assert w.sum() == pytest.approx(0.0, abs=1e-12)


def test_no_lookahead_appending_a_bar_cannot_change_prior_equity():
    df = _close(n=200, seed=1)
    r1 = xs.backtest_xsec(df, lookback=20, k=2, hold=5)
    last = df.iloc[[-1]].copy()
    last.index = [df.index[-1] + pd.Timedelta(days=1)]
    appended = pd.concat([df, last * 1.01])
    r2 = xs.backtest_xsec(appended, lookback=20, k=2, hold=5)
    common = r1["equity"].index
    assert np.allclose(r1["equity"].to_numpy(), r2["equity"].reindex(common).to_numpy())


def test_costs_reduce_return_monotonically():
    df = _close(n=400, seed=2)
    gross = xs.backtest_xsec(df, 20, 2, 3, fee_bps=0.0)["total_return"]
    mid = xs.backtest_xsec(df, 20, 2, 3, fee_bps=25.0)["total_return"]
    hi = xs.backtest_xsec(df, 20, 2, 3, fee_bps=50.0)["total_return"]
    assert gross >= mid >= hi


def test_beta_is_near_zero_on_a_synthetic_neutral_book():
    df = _close(n=500, m=8, seed=3, drift=0.001)
    r = xs.backtest_xsec(df, 20, 3, 5)
    beta = xs.beta_vs_market(df, r["daily"])
    assert abs(beta) < 0.35


def test_shuffle_null_is_deterministic_per_seed():
    df = _close(n=300, seed=4)
    a = xs.backtest_xsec(df, 20, 2, 5, shuffle_seed=7)["final_equity"]
    b = xs.backtest_xsec(df, 20, 2, 5, shuffle_seed=7)["final_equity"]
    assert a == b


def test_reverse_flips_the_sign_of_returns():
    df = _close(n=400, seed=5, drift=0.002)
    fwd = xs.backtest_xsec(df, 30, 2, 5)["daily"]
    rev = xs.backtest_xsec(df, 30, 2, 5, reverse=True)["daily"]
    assert fwd.corr(rev) < 0  # opposite bets on the same ranks


def test_contrib_is_per_symbol_and_tracks_daily_pnl():
    df = _close(n=300, seed=6)
    r = xs.backtest_xsec(df, 20, 2, 5, fee_bps=0.0)
    assert list(r["contrib"].index) == list(df.columns)
    # gross book: summed per-symbol contribution ~= summed daily portfolio PnL
    assert r["contrib"].sum() == pytest.approx(r["daily"].sum(), rel=1e-9, abs=1e-9)
