# tests/test_funding_carry.py — unit tests for the market-neutral funding-carry
# strategy (funding_carry.py). Pure functions only; no network, no venue SDK.
import numpy as np
import pandas as pd
import pytest

import funding_carry as fc


def _funding_panel(symbols, n_hours, rates):
    idx = pd.date_range("2026-01-01", periods=n_hours, freq="1h")
    return pd.DataFrame({s: rates[s] for s in symbols}, index=idx)


def test_cross_sectional_z_is_row_standardised():
    W = pd.DataFrame({"a": [1.0, 10.0], "b": [3.0, 20.0], "c": [5.0, 30.0]})
    Z = fc.cross_sectional_z(W)
    assert np.allclose(Z.mean(axis=1), 0.0, atol=1e-9)
    assert np.allclose(Z.std(axis=1), [1.0, 1.0], atol=1e-9)


def test_cross_sectional_z_constant_row_is_nan():
    W = pd.DataFrame({"a": [2.0], "b": [2.0]})
    assert fc.cross_sectional_z(W).isna().all(axis=1).all()


def test_funding_carry_score_shorts_high_funding():
    # "high" pays a large positive funding; "low" pays negative (receives).
    scores = fc.funding_carry_score(
        _funding_panel(["high", "low"], 200, {"high": 0.001, "low": -0.001}))
    assert scores["low"].iloc[-1] > scores["high"].iloc[-1]
    # higher score == better long, so select_book should be long "low".
    book = fc.select_book(scores.iloc[-1], K=1)
    assert book["low"] > 0 and book["high"] < 0


def test_select_book_is_dollar_neutral_with_unit_sides():
    score = pd.Series({"a": 3.0, "b": 2.0, "c": 1.0, "d": 0.0, "e": -1.0, "f": -2.0})
    book = fc.select_book(score, K=2)
    assert set(book) == {"a", "b", "e", "f"}
    assert pytest.approx(sum(book.values()), abs=1e-12) == 0.0          # dollar-neutral
    long_side = sum(v for v in book.values() if v > 0)
    short_side = sum(v for v in book.values() if v < 0)
    assert pytest.approx(long_side, abs=1e-12) == 1.0
    assert pytest.approx(short_side, abs=1e-12) == -1.0


def test_select_book_long_only():
    score = pd.Series({c: float(i) for i, c in enumerate("abcdef")})
    book = fc.select_book(score, K=2, long_only=True)
    assert set(book) == {"e", "f"}
    assert pytest.approx(sum(book.values()), abs=1e-12) == 1.0


def test_select_book_insufficient_names_returns_empty():
    score = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})
    assert fc.select_book(score, K=3) == {}


def test_plan_rebalance_emits_only_meaningful_deltas():
    score = pd.Series({"a": 3.0, "b": 2.0, "c": 1.0, "d": 0.0, "e": -1.0, "f": -2.0})
    target = fc.select_book(score, K=2)
    # already at target -> no orders
    cur = pd.Series(target)
    assert fc.plan_rebalance(cur, score, K=2) == []
    # flat -> one order per leg, correct sides
    orders = fc.plan_rebalance(pd.Series(dtype=float), score, K=2)
    sides = {o.symbol: o.side for o in orders}
    assert sides == {"a": "buy", "b": "buy", "e": "sell", "f": "sell"}


def test_plan_rebalance_flips_a_name_side():
    score = pd.Series({"a": -5.0, "b": 2.0, "c": 1.0, "d": 0.0, "e": -1.0, "f": -2.0})
    # currently long "a" with weight 0.25, but "a" is now the worst score
    cur = pd.Series({"a": 0.25})
    orders = {o.symbol: o for o in fc.plan_rebalance(cur, score, K=2)}
    assert orders["a"].side == "sell"
    assert orders["a"].target_weight < 0


def test_book_metrics_zero_cost_when_book_held_unchanged():
    idx = pd.date_range("2026-01-01", periods=48, freq="1h")
    weights = pd.DataFrame({"a": 0.5, "b": -0.5}, index=idx)
    # flat prices -> gross 0; entry + final close still cost fees
    prices = pd.DataFrame(100.0, index=idx, columns=["a", "b"])
    m = fc.book_metrics(prices, weights, H=24)
    assert m["gross_bps"] == pytest.approx(0.0, abs=1e-9)
    assert m["net_bps_taker"] < 0
    assert m["net_bps_maker"] > m["net_bps_taker"]   # cheaper tier loses less


def test_book_metrics_cost_scales_with_fee_and_turnover():
    idx = pd.date_range("2026-01-01", periods=72, freq="1h")
    # flip the book every 24-bar rebalance -> every period pays full turnover
    w = pd.DataFrame({"a": 0.5, "b": -0.5}, index=idx)
    w.iloc[24:, :] *= -1
    prices = pd.DataFrame(100.0, index=idx, columns=["a", "b"])
    m = fc.book_metrics(prices, w, H=24)
    # each flip costs |dw| = 2.0 (both legs reverse) + entry + final close
    assert m["turnover"] > 1.0
    assert m["net_bps_taker"] < m["net_bps_maker"] < 0


def test_book_metrics_gross_matches_held_returns():
    idx = pd.date_range("2026-01-01", periods=30, freq="1h")
    weights = pd.DataFrame({"a": 0.5, "b": -0.5}, index=idx)
    prices = pd.DataFrame({"a": 100.0, "b": 100.0}, index=idx)
    prices.loc[idx[24]:, "a"] = 110.0          # +10% over the 24-bar hold
    m = fc.book_metrics(prices, weights, H=24)
    # 0.5 weight on a +10% move -> +5% -> +500 bps gross for the period
    assert m["gross_bps"] == pytest.approx(500.0, rel=1e-6)
