# orders.py — Order submission with DB logging and sell-qty precision fix.

import asyncio
import math
from decimal import Decimal, ROUND_DOWN

from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from config import logger, trading_client, BOT_NAME, ESTIMATED_TAKER_FEE_BPS
from database import record_trade
from api_utils import call_with_rate_limit_handling_async
from money import (
    realized_pnl as calc_realized_pnl,
    net_pnl_pct as calc_net_pnl_pct,
    estimated_fee as calc_estimated_fee,
)


def _sanitize_price(price: float) -> float:
    """
    Round a price to the correct number of decimal places based on magnitude,
    preventing Alpaca's "limit price exceeds maximum precision" rejection.

    Alpaca's rule: max 9 decimal places. We apply tighter, magnitude-based
    rounding to eliminate any float64 arithmetic noise before submission.
    """
    d = Decimal(str(price))
    if price >= 1.0:
        return float(d.quantize(Decimal('0.01'), rounding=ROUND_DOWN))
    elif price >= 0.01:
        return float(d.quantize(Decimal('0.0001'), rounding=ROUND_DOWN))
    elif price >= 0.0001:
        return float(d.quantize(Decimal('0.000001'), rounding=ROUND_DOWN))
    else:
        return float(d.quantize(Decimal('0.00000001'), rounding=ROUND_DOWN))


async def place_order(symbol: str, side: OrderSide, qty: float, price: float = None,
                       avg_entry: float = None, order_id_out: dict = None,
                       market: bool = False) -> bool:
    """
    Submits an order to Alpaca and logs it to the database.
    BUYs are Limit Orders (0.1% above market to ensure fill).
    SELLs are Limit Orders at the current price, unless market=True.

    market: submit a MARKET order instead of a resting limit. Used for
    PROTECTIVE exits (stop loss, slow-bleed, max hold, kill switch) where the
    goal is to be OUT, not to get a price -- a resting price*0.999 limit can
    sit unfilled through exactly the fast drop the exit exists for, and the
    loop's stale-order cancel then re-chases it lower. Entries stay limits.

    Sell qty is floored to 8 decimal places before submission to prevent
    Alpaca's 'insufficient balance' rejection caused by float64 precision drift.
    Prices are sanitized via Decimal to prevent 'exceeds max precision' errors.

    After submission, fetches the filled order details from the exchange to
    record actual fill_price and fees, enabling accurate PnL tracking. The
    exchange reports no fee (alpaca-py's Order has no 'commission' field), so
    an ESTIMATED taker fee (config.ESTIMATED_TAKER_FEE_BPS, per side) is
    applied and flagged via record_trade(commission_estimated=True).

    avg_entry: the position's average entry price at the time of a SELL,
    used to compute and persist realized PnL (net of fees) for that closed
    position. Ignored for BUY orders; if omitted on a SELL, realized PnL is
    simply not recorded for that trade (rather than guessed).

    order_id_out: optional dict; on successful submission the exchange order
    id is written to order_id_out["order_id"] so callers can associate
    downstream logging (e.g. experience capture) with the real order.
    """
    try:
        # Crash-recovery: record pending order immediately so restart can reconcile
        try:
            import database
            database.log_pending_order(order_id_out.get("order_id") if order_id_out else None, symbol, side, qty, price)
        except Exception:
            pass
        if side == OrderSide.SELL:
            qty = math.floor(qty * 1e8) / 1e8
            if market:
                # No limit price: crosses the spread to fill immediately.
                limit_price = None
                order_data  = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.GTC,
                )
            else:
                raw_limit   = price * 0.999 if price else None
                limit_price = _sanitize_price(raw_limit) if raw_limit else None
                order_data  = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.GTC,
                    limit_price=limit_price
                )
        else:
            qty = math.floor(qty * 1e8) / 1e8  # Floor BUY qty to 8 decimals too
            raw_limit   = price * 1.001 if price else None
            limit_price = _sanitize_price(raw_limit) if raw_limit else None
            order_data  = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC,
                limit_price=limit_price
            )

        order = await call_with_rate_limit_handling_async(
            trading_client.submit_order, order_data=order_data,
            max_retries=5, base_delay=1.0
        )
        if order_id_out is not None:
            order_id_out["order_id"] = str(getattr(order, "id", None))
            order_id_out["requested_qty"] = qty
        
        # Fetch actual fill details from exchange for accurate fee/slippage tracking.
        # CRITICAL FIX: Limit orders may not fill immediately. We must wait for
        # the fill to actually happen before returning, otherwise the next
        # trading cycle reads stale positions from the exchange (which still
        # shows the position as open), causing position sizing based on stale
        # data and "order not found" errors when trying to look up the order.
        actual_fill_price = None
        actual_fee = 0.0
        actual_filled_qty = 0.0
        commission_estimated = False
        try:
            # Poll for fill status up to FILL_WAIT_TIMEOUT seconds
            FILL_WAIT_TIMEOUT = 60  # seconds
            FILL_POLL_INTERVAL = 2  # seconds between polls
            elapsed = 0.0
            filled_order = None

            while elapsed < FILL_WAIT_TIMEOUT:
                filled_order = await call_with_rate_limit_handling_async(
                    trading_client.get_order_by_id, order.id,
                    max_retries=3, base_delay=1.0
                )
                if filled_order is None:
                    await asyncio.sleep(FILL_POLL_INTERVAL)
                    elapsed += FILL_POLL_INTERVAL
                    continue

                # Check if the order has filled (filled_qty == qty means filled)
                filled_qty = getattr(filled_order, 'filled_qty', None)
                if filled_qty and float(filled_qty) > 0:
                    actual_filled_qty = float(filled_qty)
                    if hasattr(filled_order, 'filled_avg_price') and filled_order.filled_avg_price:
                        actual_fill_price = float(filled_order.filled_avg_price)
                    # alpaca-py's Order model has no 'commission' field (verified
                    # against 0.33.0 and 0.44.0), so this is normally absent and
                    # the ESTIMATED fee below is used instead.
                    fee = getattr(filled_order, 'commission', None)
                    if fee is not None:
                        actual_fee = float(fee)
                    break

                await asyncio.sleep(FILL_POLL_INTERVAL)
                elapsed += FILL_POLL_INTERVAL

            if filled_order is not None:
                if actual_fill_price is None and hasattr(filled_order, 'filled_avg_price') and filled_order.filled_avg_price:
                    actual_fill_price = float(filled_order.filled_avg_price)
                # NOTE: alpaca-py 0.33.0 Order model has no 'commission' field.

        except Exception as fill_err:
            logger.warning(f"Could not fetch fill details for order {order.id}: {fill_err}")

        if order_id_out is not None:
            # Actual filled qty (0.0 when the exchange reported none). Callers
            # use this to tell a full fill from a partial one instead of
            # trusting a lagging position re-read.
            order_id_out["filled_qty"] = actual_filled_qty

        # Distinguish "we confirmed the order did NOT fill" from "we never
        # managed to confirm anything." filled_order is only non-None when
        # get_order_by_id actually returned a status (possibly showing zero
        # fill after the full 60s timeout) -- that's a real, honest non-fill.
        # If filled_order is still None here, every poll attempt raised
        # (e.g. persistent API errors), so we genuinely don't know whether
        # the order filled; the submission itself already succeeded, so
        # that uncertainty is treated as success rather than guessed as a
        # failure, same as before this fix.
        order_confirmed_unfilled = filled_order is not None and actual_fill_price is None
        
        # Log slippage if fill price differs from expected
        if actual_fill_price and price:
            slippage = actual_fill_price - price
            slippage_pct = (slippage / price) * 100 if price > 0 else 0
            if abs(slippage_pct) > 0.1:  # Only log significant slippage
                logger.warning(
                    f"Slippage: {side.value} {symbol} | "
                    f"Expected: ${price:.4f} | Actual fill: ${actual_fill_price:.4f} | "
                    f"Diff: ${slippage:.4f} ({slippage_pct:+.2f}%)"
                )
        
        # A market order can fill only PART of the requested qty. Account for the
        # real filled qty (falling back to the requested qty when the exchange
        # reported none) so fee/PnL reflect what actually traded.
        filled_qty_used = actual_filled_qty if actual_filled_qty > 0 else qty

        # The exchange reported no commission (alpaca-py's Order has no
        # 'commission' field), which left the recorded fee at 0.0 and made
        # realized PnL silently GROSS of costs. Fall back to an ESTIMATED
        # taker fee so recorded PnL is net. The estimate covers the whole
        # round trip (entry + exit notional) on a SELL closing a known
        # position, because the entry-leg fee is only realized once the
        # position closes; a SELL with no known entry estimates just its own
        # leg. BUY rows keep fee=0.0 so SUM(fee) over the trades table is the
        # true total round-trip cost, not a double count. commission_estimated
        # flags the value as an estimate, never an exchange-confirmed fee.
        if actual_fee == 0.0 and side == OrderSide.SELL and actual_fill_price is not None:
            fee_notional = (
                (actual_fill_price + avg_entry) * filled_qty_used if avg_entry is not None
                else actual_fill_price * filled_qty_used
            )
            estimated = calc_estimated_fee(fee_notional, ESTIMATED_TAKER_FEE_BPS)
            if estimated > 0:
                actual_fee = estimated
                commission_estimated = True

        # Realized PnL is only meaningful for a SELL closing a known position,
        # and only once we have an actual fill price to measure it against --
        # measuring it against the limit `price` would conflate slippage with PnL.
        realized_pnl_dollar = None
        realized_pnl_pct = None
        if side == OrderSide.SELL and avg_entry is not None and actual_fill_price is not None:
            realized_pnl_dollar = calc_realized_pnl(avg_entry, actual_fill_price, filled_qty_used, fee=actual_fee)
            # Same round-trip basis as the dollar PnL above, so
            # realized_pnl_pct * (avg_entry * filled_qty_used) == realized_pnl_dollar.
            realized_pnl_pct = calc_net_pnl_pct(
                avg_entry, actual_fill_price, ESTIMATED_TAKER_FEE_BPS, filled_qty_used
            )

        # Pass the qty that ACTUALLY filled so record_trade stores it (and values
        # it) instead of the qty requested: on a partial fill the requested qty
        # would overstate the position and the fee/PnL computed against it.
        await asyncio.to_thread(
            record_trade,
            BOT_NAME, symbol, side.value, qty, price,
            order_id=order.id, fee=actual_fee, fill_price=actual_fill_price,
            filled_qty=actual_filled_qty or None,
            realized_pnl=realized_pnl_dollar, realized_pnl_pct=realized_pnl_pct,
            commission_estimated=commission_estimated,
        )
        fee_note = " (est)" if commission_estimated else ""
        pnl_log = f" | Realized PnL: ${realized_pnl_dollar:+.2f} ({realized_pnl_pct*100:+.2f}%)" if realized_pnl_dollar is not None else ""
        logger.info(f"Order submitted: {side.value} {symbol} {qty:.6f} limit={limit_price} fill={actual_fill_price or 'pending'} fee=${actual_fee:.4f}{fee_note}{pnl_log}")

        # CRITICAL: this used to `return True` unconditionally here, meaning
        # a GTC limit order CONFIRMED never to have filled within the 60s
        # poll window (verified in production logs: DOTUSD submitted-and-
        # canceled every ~15 minutes for hours, exactly matching
        # COOLDOWN_SECONDS_BUY) was still treated by main_bot.py as a
        # successful trade -- setting the entry cooldown, seeding
        # entry_time/highest_prices as if a position existed, and logging a
        # phantom entry to live_experiences.jsonl for capital that was never
        # actually deployed. The trades-table row above (fill_price=None)
        # still records the attempt for audit purposes; only the
        # caller-facing success signal changes.
        return not order_confirmed_unfilled

    except Exception as e:
        logger.error(f"Order failed ({side.value} {symbol} qty={qty:.6f}): {e}")
        return False
