# tests/test_orders.py — order submission (mocked Alpaca client, no network,
# no real DB since DATABASE_URL is unset in tests).

from unittest.mock import AsyncMock, MagicMock

import pytest
from alpaca.trading.enums import OrderSide

from orders import _sanitize_price, place_order
from conftest import run_async


# ── _sanitize_price ──

@pytest.mark.parametrize("price,expected_places", [
    (238.723456, 2),
    (0.05123456, 4),
    (0.00012345, 6),
    (0.0000001234, 8),
])
def test_sanitize_price_rounds_down_to_expected_precision(price, expected_places):
    result = _sanitize_price(price)
    s = f"{result:.10f}".rstrip("0")
    decimals = len(s.split(".")[1]) if "." in s else 0
    assert decimals <= expected_places


def test_sanitize_price_rounds_down_not_to_nearest():
    # 238.729 at 2dp should floor to 238.72, not round to 238.73.
    assert _sanitize_price(238.729) == 238.72


# ── place_order ──

def test_place_order_buy_success(mock_trading_client):
    fake_order = MagicMock(id="order-1")
    fake_filled = MagicMock(id="order-1", filled_qty="1.0", filled_avg_price="100.05", commission=None)
    mock_trading_client.submit_order.return_value = fake_order
    mock_trading_client.get_order_by_id.return_value = fake_filled

    result = run_async(place_order("BTC/USD", OrderSide.BUY, qty=0.01, price=100.0))

    assert result is True
    order_data = mock_trading_client.submit_order.call_args.kwargs["order_data"]
    assert order_data.symbol == "BTC/USD"
    assert order_data.side == OrderSide.BUY
    # BUY limit is 0.1% above the reference price
    assert order_data.limit_price == pytest.approx(100.1, abs=0.01)


def test_place_order_sell_prices_below_market(mock_trading_client):
    fake_order = MagicMock(id="order-2")
    mock_trading_client.submit_order.return_value = fake_order
    mock_trading_client.get_order_by_id.return_value = MagicMock(
        id="order-2", filled_qty="0.01", filled_avg_price="99.9", commission=None,
    )

    result = run_async(place_order("BTC/USD", OrderSide.SELL, qty=0.01, price=100.0))

    assert result is True
    order_data = mock_trading_client.submit_order.call_args.kwargs["order_data"]
    assert order_data.side == OrderSide.SELL
    # SELL limit is 0.1% below the reference price
    assert order_data.limit_price == pytest.approx(99.9, abs=0.01)


def test_place_order_returns_false_on_submit_failure(mock_trading_client):
    mock_trading_client.submit_order.side_effect = RuntimeError("network error")

    result = run_async(place_order("BTC/USD", OrderSide.BUY, qty=0.01, price=100.0))

    assert result is False


def test_place_order_still_succeeds_if_fill_lookup_fails(mock_trading_client):
    """Order submission succeeding is what matters; a failure to fetch fill
    details afterward (e.g. transient API hiccup) must not be treated as a
    failed trade -- it already executed on the exchange."""
    fake_order = MagicMock(id="order-3")
    mock_trading_client.submit_order.return_value = fake_order
    mock_trading_client.get_order_by_id.side_effect = RuntimeError("transient")

    result = run_async(place_order("BTC/USD", OrderSide.BUY, qty=0.01, price=100.0))

    assert result is True


def test_place_order_returns_false_when_confirmed_unfilled(mock_trading_client, monkeypatch):
    """Regression test: a GTC limit order that Alpaca CONFIRMS never filled
    within the poll window (filled_qty stays 0, no exception -- unlike the
    transient-lookup-failure case above) must return False, not True.

    Before this fix, place_order() returned True unconditionally after a
    successful submission regardless of fill status. main_bot.py treats a
    True return as "trade happened": it sets the entry cooldown, seeds
    entry_time/highest_prices as if a position existed, and logs a phantom
    entry to live_experiences.jsonl for capital that was never deployed.
    Production logs showed this exact pattern: the same symbol's BUY limit
    submitted and canceled every ~15 minutes for hours -- precisely
    COOLDOWN_SECONDS_BUY, confirming the cooldown fired on every attempt
    whether or not anything actually filled.
    """
    import orders
    recorded = {}
    monkeypatch.setattr(orders, "record_trade", lambda *a, **kw: recorded.update(kw))
    # A genuinely-unfilled order polls for the full 60s timeout (30 x 2s) --
    # collapse that to instant so this test doesn't really sleep a minute.
    monkeypatch.setattr(orders.asyncio, "sleep", AsyncMock(return_value=None))

    mock_trading_client.submit_order.return_value = MagicMock(id="order-unfilled")
    # A real Alpaca response confirming zero fill (not an exception/timeout).
    mock_trading_client.get_order_by_id.return_value = MagicMock(
        id="order-unfilled", filled_qty="0", filled_avg_price=None, commission=None,
    )

    result = run_async(place_order("BTC/USD", OrderSide.BUY, qty=0.01, price=100.0))

    assert result is False
    # The attempt is still recorded for audit purposes, with no fill price.
    assert recorded["fill_price"] is None


# ── Realized PnL pass-through ──

def test_sell_with_avg_entry_and_fill_records_realized_pnl(mock_trading_client, monkeypatch):
    import orders
    recorded = {}
    monkeypatch.setattr(orders, "record_trade", lambda *a, **kw: recorded.update(kw))

    fake_order = MagicMock(id="order-4")
    mock_trading_client.submit_order.return_value = fake_order
    mock_trading_client.get_order_by_id.return_value = MagicMock(
        id="order-4", filled_qty="2.0", filled_avg_price="110.0", commission=None,
    )

    result = run_async(place_order("BTC/USD", OrderSide.SELL, qty=2.0, price=109.9, avg_entry=100.0))

    assert result is True
    # alpaca-py's Order model has no 'commission' field, so an ESTIMATED taker
    # fee is applied (default 25 bps/side) and realized PnL is now NET of it.
    # Round-trip notional = (exit 110 + entry 100) * 2 = 420; fee = 420*0.0025 = 1.05.
    assert recorded["fee"] == pytest.approx(1.05)
    assert recorded["commission_estimated"] is True
    assert recorded["realized_pnl"] == pytest.approx(20.0 - 1.05)   # (110-100)*2 - 1.05
    # pct is now NET of the estimated round-trip fee too: 0.10 - 25bps = 0.0975
    assert recorded["realized_pnl_pct"] == pytest.approx(0.10 - 0.0025)


def test_buy_never_records_realized_pnl(mock_trading_client, monkeypatch):
    import orders
    recorded = {}
    monkeypatch.setattr(orders, "record_trade", lambda *a, **kw: recorded.update(kw))

    mock_trading_client.submit_order.return_value = MagicMock(id="order-5")
    mock_trading_client.get_order_by_id.return_value = MagicMock(
        id="order-5", filled_qty="1.0", filled_avg_price="100.1", commission=None,
    )

    run_async(place_order("BTC/USD", OrderSide.BUY, qty=1.0, price=100.0, avg_entry=95.0))

    assert recorded["realized_pnl"] is None
    assert recorded["realized_pnl_pct"] is None


def test_sell_without_avg_entry_does_not_record_realized_pnl(mock_trading_client, monkeypatch):
    """swap_weakest_position/sell_largest_position don't go through place_order
    at all today (a separate known gap), but any SELL that omits avg_entry --
    e.g. because the caller has no position data -- must not fabricate a
    realized PnL number rather than silently guessing at one."""
    import orders
    recorded = {}
    monkeypatch.setattr(orders, "record_trade", lambda *a, **kw: recorded.update(kw))

    mock_trading_client.submit_order.return_value = MagicMock(id="order-6")
    mock_trading_client.get_order_by_id.return_value = MagicMock(
        id="order-6", filled_qty="2.0", filled_avg_price="110.0", commission=None,
    )

    run_async(place_order("BTC/USD", OrderSide.SELL, qty=2.0, price=109.9))  # no avg_entry

    assert recorded["realized_pnl"] is None
    assert recorded["realized_pnl_pct"] is None


def test_sell_without_a_fill_price_does_not_record_realized_pnl(mock_trading_client, monkeypatch):
    """Must not compute realized PnL against the limit price when the actual
    fill price is unknown -- that would conflate slippage with PnL."""
    import orders
    recorded = {}
    monkeypatch.setattr(orders, "record_trade", lambda *a, **kw: recorded.update(kw))

    mock_trading_client.submit_order.return_value = MagicMock(id="order-7")
    mock_trading_client.get_order_by_id.side_effect = RuntimeError("transient")

    run_async(place_order("BTC/USD", OrderSide.SELL, qty=2.0, price=109.9, avg_entry=100.0))

    assert recorded["realized_pnl"] is None
    assert recorded["realized_pnl_pct"] is None


# ── Protective exits: market sells ──
# Stop loss / slow-bleed / max hold / kill switch submit market=True so the
# position is actually out, rather than resting at price*0.999 where a fast
# drop can leave the exit unfilled (and the loop's stale-order cancel then
# re-chases it lower).

def test_market_sell_submits_a_market_order_with_no_limit_price(mock_trading_client):
    from alpaca.trading.requests import MarketOrderRequest

    mock_trading_client.submit_order.return_value = MagicMock(id="mkt-1")
    mock_trading_client.get_order_by_id.return_value = MagicMock(
        id="mkt-1", filled_qty="0.01", filled_avg_price="99.0", commission=None,
    )

    run_async(place_order("BTC/USD", OrderSide.SELL, qty=0.01, price=100.0, market=True))

    order_data = mock_trading_client.submit_order.call_args.kwargs["order_data"]
    assert isinstance(order_data, MarketOrderRequest)
    assert order_data.side == OrderSide.SELL
    assert not hasattr(order_data, "limit_price") or order_data.limit_price is None


def test_market_sell_uses_the_quantity_actually_held_not_quantity_bought(mock_trading_client):
    """The protective exit path passes the qty read from the live position.
    A partial/leftover position (e.g. 0.007 of a 0.01 buy) must be sold at
    exactly that qty -- never the original buy size."""
    mock_trading_client.submit_order.return_value = MagicMock(id="mkt-2")
    mock_trading_client.get_order_by_id.return_value = MagicMock(
        id="mkt-2", filled_qty="0.007", filled_avg_price="99.0", commission=None,
    )

    run_async(place_order("BTC/USD", OrderSide.SELL, qty=0.007, price=100.0, market=True))

    order_data = mock_trading_client.submit_order.call_args.kwargs["order_data"]
    assert order_data.qty == pytest.approx(0.007)


def test_crypto_market_sell_time_in_force_is_gtc(mock_trading_client):
    """Alpaca crypto accepts ONLY gtc and ioc -- day/fok/opg/cls are rejected.
    The protective market sell must use gtc (it already does); this pins it so a
    future edit can't switch the market order to 'day' and get it rejected."""
    from alpaca.trading.enums import TimeInForce

    mock_trading_client.submit_order.return_value = MagicMock(id="tif-1")
    mock_trading_client.get_order_by_id.return_value = MagicMock(
        id="tif-1", filled_qty="0.01", filled_avg_price="99.0", commission=None,
    )

    run_async(place_order("BTC/USD", OrderSide.SELL, qty=0.01, price=100.0, market=True))

    order_data = mock_trading_client.submit_order.call_args.kwargs["order_data"]
    assert order_data.time_in_force == TimeInForce.GTC


def test_crypto_market_sell_tif_is_one_alpaca_accepts_for_crypto(mock_trading_client):
    from alpaca.trading.enums import TimeInForce

    # Per Alpaca docs: "For Crypto Trading, Alpaca only supports gtc, and ioc.
    # OPG, fok, day, and CLS are not supported."
    ALPACA_CRYPTO_TIFS = {TimeInForce.GTC, TimeInForce.IOC}

    mock_trading_client.submit_order.return_value = MagicMock(id="tif-2")
    mock_trading_client.get_order_by_id.return_value = MagicMock(
        id="tif-2", filled_qty="0.01", filled_avg_price="99.0", commission=None,
    )

    run_async(place_order("BTC/USD", OrderSide.SELL, qty=0.01, price=100.0, market=True))

    order_data = mock_trading_client.submit_order.call_args.kwargs["order_data"]
    assert order_data.time_in_force in ALPACA_CRYPTO_TIFS
    assert TimeInForce.DAY not in ALPACA_CRYPTO_TIFS


def test_limit_sell_is_still_the_default(mock_trading_client):
    from alpaca.trading.requests import LimitOrderRequest

    mock_trading_client.submit_order.return_value = MagicMock(id="lim-1")
    mock_trading_client.get_order_by_id.return_value = MagicMock(
        id="lim-1", filled_qty="0.01", filled_avg_price="99.9", commission=None,
    )

    run_async(place_order("BTC/USD", OrderSide.SELL, qty=0.01, price=100.0))

    order_data = mock_trading_client.submit_order.call_args.kwargs["order_data"]
    assert isinstance(order_data, LimitOrderRequest)
    assert order_data.limit_price == pytest.approx(99.9, abs=0.01)


def test_buy_ignores_market_flag_and_stays_a_limit(mock_trading_client):
    """Entries must remain limits even if market=True is passed -- the market
    flag is for protective sells only."""
    from alpaca.trading.requests import LimitOrderRequest

    mock_trading_client.submit_order.return_value = MagicMock(id="buy-1")
    mock_trading_client.get_order_by_id.return_value = MagicMock(
        id="buy-1", filled_qty="0.01", filled_avg_price="100.1", commission=None,
    )

    run_async(place_order("BTC/USD", OrderSide.BUY, qty=0.01, price=100.0, market=True))

    order_data = mock_trading_client.submit_order.call_args.kwargs["order_data"]
    assert isinstance(order_data, LimitOrderRequest)
    assert order_data.limit_price == pytest.approx(100.1, abs=0.01)


# ── Estimated fee fallback ──

def test_exchange_reported_commission_is_used_and_not_flagged_estimated(mock_trading_client, monkeypatch):
    import orders
    recorded = {}
    monkeypatch.setattr(orders, "record_trade", lambda *a, **kw: recorded.update(kw))

    mock_trading_client.submit_order.return_value = MagicMock(id="fee-1")
    mock_trading_client.get_order_by_id.return_value = MagicMock(
        id="fee-1", filled_qty="2.0", filled_avg_price="110.0", commission="1.23",
    )

    run_async(place_order("BTC/USD", OrderSide.SELL, qty=2.0, price=109.9, avg_entry=100.0))

    assert recorded["fee"] == pytest.approx(1.23)
    assert recorded["commission_estimated"] is False
    assert recorded["realized_pnl"] == pytest.approx(20.0 - 1.23)


def test_buy_never_records_an_estimated_fee(mock_trading_client, monkeypatch):
    """Fee estimate applies to SELLs only, so SUM(fee) over the trades table is
    the true round-trip cost and not a double count of the entry leg."""
    import orders
    recorded = {}
    monkeypatch.setattr(orders, "record_trade", lambda *a, **kw: recorded.update(kw))

    mock_trading_client.submit_order.return_value = MagicMock(id="fee-2")
    mock_trading_client.get_order_by_id.return_value = MagicMock(
        id="fee-2", filled_qty="1.0", filled_avg_price="100.1", commission=None,
    )

    run_async(place_order("BTC/USD", OrderSide.BUY, qty=1.0, price=100.0))

    assert recorded["fee"] == 0.0
    assert recorded["commission_estimated"] is False


def test_partial_fill_fee_and_pnl_use_the_filled_qty(mock_trading_client, monkeypatch):
    """A market sell can fill only part of the requested qty. Fee and realized
    PnL must be computed on the FILLED qty (0.4), not the requested qty (1.0) --
    otherwise the exit leg is over-charged and PnL is understated. The exchange's
    filled qty is also handed to record_trade so the stored row reflects it."""
    import orders
    recorded = {}
    monkeypatch.setattr(orders, "record_trade",
                        lambda *a, **kw: recorded.update(kw, _args=a, _kw=kw))

    mock_trading_client.submit_order.return_value = MagicMock(id="part-1")
    mock_trading_client.get_order_by_id.return_value = MagicMock(
        id="part-1", filled_qty="0.4", filled_avg_price="110.0", commission=None,
    )

    run_async(place_order("BTC/USD", OrderSide.SELL, qty=1.0, price=109.9, avg_entry=100.0))

    # Round-trip notional on 0.4: (110 + 100) * 0.4 = 84; fee = 84 * 25bps = 0.21
    assert recorded["filled_qty"] == pytest.approx(0.4)  # what record_trade stores
    assert recorded["fee"] == pytest.approx(0.21)
    assert recorded["commission_estimated"] is True
    # Realized PnL on 0.4: (110 - 100) * 0.4 - 0.21
    assert recorded["realized_pnl"] == pytest.approx(10 * 0.4 - 0.21)


def test_full_fill_passes_the_filled_qty_to_record_trade(mock_trading_client, monkeypatch):
    import orders
    recorded = {}
    monkeypatch.setattr(orders, "record_trade",
                        lambda *a, **kw: recorded.update(kw, _args=a, _kw=kw))

    mock_trading_client.submit_order.return_value = MagicMock(id="full-1")
    mock_trading_client.get_order_by_id.return_value = MagicMock(
        id="full-1", filled_qty="2.0", filled_avg_price="110.0", commission=None,
    )
    run_async(place_order("BTC/USD", OrderSide.SELL, qty=2.0, price=109.9, avg_entry=100.0))
    assert recorded["filled_qty"] == pytest.approx(2.0)


def test_realized_pnl_pct_is_net_of_estimated_fee(mock_trading_client, monkeypatch):
    """The percentage must agree with the fee-netted dollar PnL: gross move back
    out the estimated round-trip fee (25 bps here), not the raw price move."""
    import config
    import orders
    recorded = {}
    monkeypatch.setattr(orders, "record_trade", lambda *a, **kw: recorded.update(kw))

    mock_trading_client.submit_order.return_value = MagicMock(id="pct-1")
    mock_trading_client.get_order_by_id.return_value = MagicMock(
        id="pct-1", filled_qty="1.0", filled_avg_price="110.0", commission=None,
    )
    run_async(place_order("BTC/USD", OrderSide.SELL, qty=1.0, price=109.9, avg_entry=100.0))

    gross = (110.0 - 100.0) / 100.0                       # 0.10
    expected = gross - config.ESTIMATED_TAKER_FEE_BPS / 10000  # - 0.0025
    assert recorded["realized_pnl_pct"] == pytest.approx(expected)
    # and it is strictly less than the gross percentage
    assert recorded["realized_pnl_pct"] < gross


def test_sell_without_avg_entry_estimates_only_the_exit_leg(mock_trading_client, monkeypatch):
    import orders
    recorded = {}
    monkeypatch.setattr(orders, "record_trade", lambda *a, **kw: recorded.update(kw))

    mock_trading_client.submit_order.return_value = MagicMock(id="fee-3")
    mock_trading_client.get_order_by_id.return_value = MagicMock(
        id="fee-3", filled_qty="2.0", filled_avg_price="110.0", commission=None,
    )

    run_async(place_order("BTC/USD", OrderSide.SELL, qty=2.0, price=109.9))  # no avg_entry

    # Exit-leg notional only: 110 * 2 * 25 bps = 0.55
    assert recorded["fee"] == pytest.approx(0.55)
    assert recorded["commission_estimated"] is True
    assert recorded["realized_pnl"] is None
