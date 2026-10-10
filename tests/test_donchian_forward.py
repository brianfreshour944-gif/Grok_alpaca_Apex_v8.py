"""Tests for the Donchian paper-ledger forward marking."""
import numpy as np
import pandas as pd
import pytest

import research.donchian_forward as fwd


def _close(intraday=False, n=6):
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    if intraday:
        idx = idx + pd.Timedelta(hours=16)   # venue style (OKX 16:00 UTC)
    a = 100 * np.cumprod(1 + np.full(n, 0.10))   # +10%/day
    b = np.full(n, 100.0)                        # flat
    return pd.DataFrame({"AAA": a, "BBB": b}, index=idx)


def _rows(books):
    return [{"date": f"2024-01-{i+1:02d}", "book": b} for i, b in enumerate(books)]


def test_parse_book_reads_symbol_signal_weight():
    got = fwd.parse_book("BTCUSDT:1:+0.050|ETHUSDT:-1:-0.030|X:0:0.000")
    assert got == {"BTCUSDT": 0.05, "ETHUSDT": -0.03, "X": 0.0}


def test_parse_book_ignores_malformed():
    assert fwd.parse_book("garbage|BTCUSDT:1") == {}
    assert fwd.parse_book("") == {}


def test_mark_ledger_holds_weights_forward():
    # same 0.5 book every day -> held for all 3 intervals, compounding
    rows = _rows(["AAA:1:+0.500"] * 4)
    m = fwd.mark_ledger(rows, _close(), fee_bps=0.0)
    assert m["n_days"] == 3
    assert m["book_total_return"] == pytest.approx((1.05) ** 3 - 1, rel=1e-6)


def test_mark_ledger_benchmark_is_equal_weight():
    rows = _rows(["AAA:0:0.000|BBB:0:0.000"] * 4)   # flat book, 2 symbols in scope
    m = fwd.mark_ledger(rows, _close(), fee_bps=0.0)
    assert m["book_total_return"] == pytest.approx(0.0, abs=1e-12)
    # equal-weight of (+10%, 0%) = +5%/day compounded
    assert m["bench_total_return"] == pytest.approx((1.05) ** 3 - 1, rel=1e-6)


def test_intraday_stamps_align_with_date_only_ledger():
    """The fixed bug: 16:00 UTC bars must line up with date-only ledger rows."""
    rows = _rows(["AAA:1:+0.500", "", ""])
    day = fwd.mark_ledger(rows, _close(intraday=False), fee_bps=0.0)
    intra = fwd.mark_ledger(rows, _close(intraday=True), fee_bps=0.0)
    assert day["book_total_return"] == pytest.approx(intra["book_total_return"], rel=1e-9)


def test_marking_is_directionally_sane():
    """A long book over a rising name must gain, not lose; no sign flip."""
    rows = _rows(["AAA:1:+0.500"] * 3)
    m = fwd.mark_ledger(rows, _close(), fee_bps=0.0)
    assert m["book_total_return"] == pytest.approx((1.05) ** 2 - 1, rel=1e-6)


def test_costs_reduce_the_book_return():
    rows = _rows(["AAA:1:+0.500"] * 2 + ["BBB:1:+0.500"] * 2)
    g = fwd.mark_ledger(rows, _close(), fee_bps=0.0)["book_total_return"]
    n = fwd.mark_ledger(rows, _close(), fee_bps=50.0)["book_total_return"]
    assert g > n


def test_single_row_is_not_enough_to_mark():
    assert fwd.mark_ledger(_rows(["AAA:1:+0.500"]), _close())["n_days"] == 0
