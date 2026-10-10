# tests/test_donchian_bot.py — operational layer tests (no network, no config).
#
# Imports only numpy/pandas/pytest + the two donchian modules, so it runs
# without alpaca/torch and does not touch config's live clients.
import os
import tempfile

import numpy as np
import pandas as pd

import donchian_bot as bot


def _uptrend(n=60, start=100.0, slope=0.01):
    close = start * np.cumprod(np.full(n, 1 + slope))
    high = close * 1.001
    low = close * 0.999
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"high": high, "low": low, "close": close}, index=idx)


def _downtrend(n=60, start=100.0, slope=-0.01):
    close = start * np.cumprod(np.full(n, 1 + slope))
    high = close * 1.001
    low = close * 0.999
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"high": high, "low": low, "close": close}, index=idx)


def test_latest_signal_long_in_uptrend():
    sig, price, a, band = bot.latest_signal(_uptrend())
    assert sig == 1
    assert price > 0 and a > 0 and band > 0


def test_units_for_trend_scales_with_progress_and_caps():
    # at/just above the band -> 1 unit; deep in trend -> more, capped
    assert bot.units_for_trend(price=100.0, band=100.0, atr_value=2.0, max_units=4) == 1
    assert bot.units_for_trend(100.0, 99.0, 2.0, max_units=4) == 2   # 0.5 ATR -> 1+1
    assert bot.units_for_trend(200.0, 100.0, 2.0, max_units=4) == 4  # capped
    assert bot.units_for_trend(100.0, 100.0, 2.0, max_units=1) == 1  # pyramiding off
    assert bot.units_for_trend(100.0, 100.0, float("nan"), max_units=3) == 3


def test_target_book_only_includes_active_signals():
    data = {"BTCUSDT": _uptrend(), "LTCUSDT": _downtrend()}
    book = bot.target_book(data, equity=10_000)
    assert "BTCUSDT" in book and book["BTCUSDT"].signal == 1
    assert "LTCUSDT" not in book          # long-only: downtrend is flat


def test_target_book_caps_position_and_gross():
    data = {f"S{i}": _uptrend(slope=0.02 + 0.001 * i) for i in range(10)}
    book = bot.target_book(data, equity=10_000, risk_pct=0.05, max_position_pct=0.5)
    assert all(abs(t.weight) <= 0.5 + 1e-9 for t in book.values())
    assert sum(abs(t.weight) for t in book.values()) <= bot.MAX_GROSS_PCT + 1e-9


def test_plan_rebalance_enters_exits_and_ignores_dust():
    targets = {"BTCUSDT": bot.Target("BTCUSDT", 1, 100.0, 2.0, 0.30)}
    # no position -> enter
    d = bot.plan_rebalance(targets, {})
    assert [x.side for x in d] == ["buy"]
    # at target already -> hold
    d = bot.plan_rebalance(targets, {"BTCUSDT": 0.30})
    assert d[0].side == "hold"
    # held but no longer targeted -> exit to zero
    d = bot.plan_rebalance({}, {"BTCUSDT": 0.30})
    assert d[0].side == "sell" and d[0].reason == "exit"
    # tiny drift -> suppressed
    d = bot.plan_rebalance(targets, {"BTCUSDT": 0.299})
    assert d[0].side == "hold"


def test_append_ledger_is_idempotent_by_date():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "l.csv")
        book = {"BTCUSDT": bot.Target("BTCUSDT", 1, 100.0, 2.0, 0.3)}
        assert bot.append_ledger(path, "2026-10-10", 10_000, book) is True
        assert bot.append_ledger(path, "2026-10-10", 10_000, book) is False
        assert bot.append_ledger(path, "2026-10-11", 10_100, book) is True
        df = pd.read_csv(path)
        assert len(df) == 2
        assert list(df.columns) == bot.LEDGER_COLS


def test_run_cycle_paper_places_no_orders():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "l.csv")
        data = {"BTCUSDT": _uptrend()}
        broker = bot.PaperBroker(equity=10_000)
        decisions = bot.run_cycle(data, broker, "2026-10-10", ledger=path, live=False)
        assert any(d.side == "buy" for d in decisions)
        assert broker.placed == []                 # paper: nothing submitted
        assert os.path.exists(path)


def test_run_cycle_live_submits_orders_through_broker():
    class FakeBroker(bot.PaperBroker):
        def submit(self, symbol, side, qty):
            self.placed.append({"symbol": symbol, "side": side, "qty": qty})
            return "fake-1"

    data = {"BTCUSDT": _uptrend()}
    broker = FakeBroker(equity=10_000)
    bot.run_cycle(data, broker, "2026-10-10", live=True)
    assert len(broker.placed) == 1
    assert broker.placed[0]["side"] == "buy"
    assert broker.placed[0]["qty"] > 0


def test_symbol_mapping_roundtrip():
    assert bot.to_alpaca_symbol("BTCUSDT") == "BTC/USD"
    assert bot.from_alpaca_symbol("BTC/USD") == "BTCUSDT"
    assert bot.to_alpaca_symbol("TRXUSDT") is None      # research-only
    assert bot.from_alpaca_symbol("SHIB/USD") is None


def test_run_cycle_live_skips_research_only_symbols():
    class FakeBroker(bot.PaperBroker):
        def submit(self, symbol, side, qty):
            self.placed.append({"symbol": symbol, "side": side, "qty": qty})
            return "fake-1"

    # TRXUSDT has a long signal but is not on Alpaca -> must NOT be ordered.
    data = {"TRXUSDT": _uptrend()}
    broker = FakeBroker(equity=10_000)
    decisions = bot.run_cycle(data, broker, "2026-10-10", live=True)
    assert any(d.side == "buy" for d in decisions)      # it IS in the plan
    assert broker.placed == []                          # but no live order


def test_ledger_records_mode():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "l.csv")
        book = {"BTCUSDT": bot.Target("BTCUSDT", 1, 100.0, 2.0, 0.3)}
        bot.append_ledger(path, "2026-10-10", 10_000, book, mode="enhanced")
        df = pd.read_csv(path)
        assert df.loc[0, "mode"] == "enhanced"
        assert "mode" in bot.LEDGER_COLS


def test_enhanced_config_defaults_are_the_validated_combo():
    assert bot.ENHANCED == {"trend_filter": 100, "exit_mode": "midpoint",
                            "pyramid_units": 2}
    assert bot.BASE == {}


def test_target_book_accepts_strategy_config():
    data = {"BTCUSDT": _uptrend(n=160)}
    base = bot.target_book(data, 10_000, strategy=bot.BASE)
    enh = bot.target_book(data, 10_000, strategy=bot.ENHANCED)
    # both should still size a long position; the strategy kw is threaded through
    assert base["BTCUSDT"].signal == enh["BTCUSDT"].signal == 1
    assert enh["BTCUSDT"].weight > 0


def test_trend_filter_needs_enough_history():
    # a 100-bar trend gate cannot fire on a 60-bar series -> stays flat
    short = bot.target_book({"BTCUSDT": _uptrend(n=60)}, 10_000, strategy=bot.ENHANCED)
    assert short == {}


def test_alpaca_broker_sizes_off_account_equity(monkeypatch):
    """The live broker's equity() must use account equity, not buying power.

    Kept dependency-isolated by injecting fakes for the modules make_alpaca_broker
    imports lazily (alpaca.trading.enums, orders, portfolio), so this test never
    touches config's real clients.
    """
    import sys
    import types

    calls = {"equity": 0, "buying_power": 0}

    fake_portfolio = types.ModuleType("portfolio")

    def _equity():
        calls["equity"] += 1
        return 4242.0

    def _bp():
        calls["buying_power"] += 1
        return 0.0

    fake_portfolio.get_account_equity = _equity
    fake_portfolio.get_buying_power = _bp
    fake_portfolio.get_all_positions = dict

    fake_orders = types.ModuleType("orders")
    fake_orders.place_order = lambda *a, **k: True

    fake_alpaca = types.ModuleType("alpaca")
    fake_trading = types.ModuleType("alpaca.trading")
    fake_enums = types.ModuleType("alpaca.trading.enums")
    fake_enums.OrderSide = types.SimpleNamespace(BUY="buy", SELL="sell")
    fake_trading.enums = fake_enums
    fake_alpaca.trading = fake_trading

    for name, mod in [("portfolio", fake_portfolio), ("orders", fake_orders),
                      ("alpaca", fake_alpaca), ("alpaca.trading", fake_trading),
                      ("alpaca.trading.enums", fake_enums)]:
        monkeypatch.setitem(sys.modules, name, mod)

    broker = bot.make_alpaca_broker()
    assert broker.equity() == 4242.0
    assert calls["equity"] == 1
    assert calls["buying_power"] == 0        # never consulted for sizing
    assert broker.positions() == {}
