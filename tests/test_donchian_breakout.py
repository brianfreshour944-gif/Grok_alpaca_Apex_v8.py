# tests/test_donchian_breakout.py — regression tests for donchian_breakout.py.
#
# Deliberately dependency-isolated: imports only numpy/pandas/pytest and the
# strategy module, NOT config/alpaca (which build live clients at import time).
# Run with: python -m pytest tests/test_donchian_breakout.py
import numpy as np
import pandas as pd
import pytest

import donchian_breakout as db


def _df(high, low, close):
    return pd.DataFrame({"high": high, "low": low, "close": close},
                        dtype=float)


# ── donchian_channel: the band must exclude the current bar ───────────────────
def test_channel_excludes_current_bar():
    high = [1, 2, 3, 4, 10]
    low = [0, 1, 2, 3, 4]
    upper, lower = db.donchian_channel(high, low, window=3)
    # upper[4] = max(high[1:4]) = 4 — the spike at index 4 must NOT be included
    assert upper[4] == 4
    # upper[3] = max(high[0:3]) = 3
    assert upper[3] == 3
    assert np.isnan(upper[0]) and np.isnan(upper[1]) and np.isnan(upper[2])
    assert lower[4] == min(low[1:4]) == 1


def test_channel_appending_a_bar_cannot_change_prior_values():
    base_hi = np.array([1.0, 2, 3, 4, 5, 6])
    base_lo = base_hi - 1
    up_a, lo_a = db.donchian_channel(base_hi, base_lo, 3)
    up_b, lo_b = db.donchian_channel(np.append(base_hi, 999.0),
                                     np.append(base_lo, 998.0), 3)
    np.testing.assert_allclose(up_a, up_b[:-1], equal_nan=True)
    np.testing.assert_allclose(lo_a, lo_b[:-1], equal_nan=True)


# ── atr ───────────────────────────────────────────────────────────────────────
def test_atr_matches_manual_true_range():
    high = [10.0, 12, 11, 13, 14]
    low = [8.0, 9, 9, 10, 11]
    close = [9.0, 11, 10, 12, 13]
    a = db.atr(high, low, close, window=2)
    # TR[1] = max(12-9, |12-9|, |9-9|) = 3 ; TR[2] = max(11-9,|11-11|,|9-11|)=2
    assert a[2] == pytest.approx((3 + 2) / 2)


# ── breakout_signal ───────────────────────────────────────────────────────────
def test_signal_enters_on_breakout_and_exits_on_reversal():
    # 5 flat bars (band 1..5), breakout up, then breakdown.
    close = [1.0, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    high = [c + 0.5 for c in close]
    low = [c - 0.5 for c in close]
    sig = db.breakout_signal(high, low, close, entry=3, exit=2)
    assert sig[5] == 1                       # close 6 > upper 3-window band
    assert sig[8] == 1                       # still long
    assert sig[9] == 0 or sig[10] == 0       # exits on the way down
    assert sig[-1] == 0


def test_signal_flat_until_band_exists():
    close = [1.0, 2, 3]
    high = [c + 0.5 for c in close]
    low = [c - 0.5 for c in close]
    sig = db.breakout_signal(high, low, close, entry=10, exit=5)
    assert np.all(sig == 0)


def test_signal_short_side():
    close = [9.0, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    high = [c + 0.5 for c in close]
    low = [c - 0.5 for c in close]
    sig = db.breakout_signal(high, low, close, entry=3, exit=2, allow_short=True)
    assert sig[3] == -1
    assert set(np.unique(sig)).issubset({-1, 0, 1})


# ── turtle_units ──────────────────────────────────────────────────────────────
def test_turtle_units_sizes_by_risk_and_caps():
    # risk 1% of equity, 1-ATR move costs risk_pct: fraction = 0.01*price/atr
    assert db.turtle_units(price=100.0, atr_value=2.0, risk_pct=0.01) == pytest.approx(0.5)
    # capped at max_notional_pct
    assert db.turtle_units(100.0, 0.01, 0.01, max_notional_pct=1.0) == 1.0
    # degenerate inputs -> flat
    assert db.turtle_units(100.0, 0.0) == 0.0
    assert db.turtle_units(0.0, 2.0) == 0.0
    assert db.turtle_units(np.nan, 2.0) == 0.0


# ── backtest: accounting + no look-ahead ──────────────────────────────────────
def _trending(n=200, slope=0.01, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + slope + rng.normal(0, noise, n))
    high = close * 1.002
    low = close * 0.998
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"high": high, "low": low, "close": close}, index=idx)


def test_backtest_no_lookahead_signal_is_causal():
    """Appending a future bar must not change any earlier signal."""
    df = _trending(n=120, slope=0.02, noise=0.01, seed=1)
    sig_a = db.breakout_signal(df["high"], df["low"], df["close"], 20, 10)
    df2 = pd.concat([df, _trending(n=1, slope=0.5, seed=9).set_index(
        pd.date_range("2024-05-01", periods=1, freq="D"))])
    sig_b = db.breakout_signal(df2["high"], df2["low"], df2["close"], 20, 10)
    np.testing.assert_array_equal(sig_a, sig_b[:-1])


def test_backtest_pnl_matches_equity_and_fees_monotone():
    df = _trending(n=300, slope=0.01, noise=0.02, seed=3)
    free = db.backtest_donchian(df, entry=20, exit=10, fee_bps=0.0)
    costly = db.backtest_donchian(df, entry=20, exit=10, fee_bps=50.0)
    # more fees cannot help a strategy that trades
    assert free["n_trades"] > 0
    assert costly["final_equity"] < free["final_equity"]
    # sum of trade PnL reconciles with the equity curve
    tr = free["trades"]
    assert tr["pnl"].sum() == pytest.approx(free["final_equity"] - free["initial_equity"], rel=1e-9)


def test_backtest_makes_money_on_a_clean_uptrend():
    df = _trending(n=400, slope=0.02, noise=0.0, seed=0)
    r = db.backtest_donchian(df, entry=20, exit=10, fee_bps=0.0)
    assert r["total_return"] > 0
    assert r["buy_hold_return"] > 0


def test_backtest_stop_reason_present_on_sharp_reversal():
    close = list(np.linspace(100, 130, 60)) + [90.0, 80, 70]
    high = [c * 1.001 for c in close]
    low = [c * 0.999 for c in close]
    df = _df(high, low, close)
    r = db.backtest_donchian(df, entry=20, exit=10, stop_atr=1.0, fee_bps=0.0)
    reasons = set(r["trades"]["reason"]) if not r["trades"].empty else set()
    assert "stop" in reasons


def test_backtest_rejects_too_few_bars():
    with pytest.raises(ValueError):
        db.backtest_donchian(_df([1, 2], [1, 2], [1, 2]), entry=20, exit=10)


def test_backtest_exposure_in_unit_interval():
    df = _trending(n=200, slope=0.005, noise=0.03, seed=5)
    r = db.backtest_donchian(df)
    assert 0.0 <= r["exposure"] <= 1.0
    assert -1.0 <= r["max_drawdown"] <= 0.0
