#!/usr/bin/env python3
import asyncio
import logging
import os
import sys
import time
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

from ml_predictor import MLPredictor

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# --- GLOBAL CONFIGURATION ---
BOT_NAME = os.getenv("BOT_NAME", "Grok_Alpaca_Apex_v8")
SYMBOLS = ["BTC/USD", "ETH/USD", "LTC/USD", "DOGE/USD"]
ORDER_AMOUNT = 50.0   # USD per trade
MODEL_PATH = "/app/data/grok_gqa_v9_best.pth" if os.path.exists("/app/data") else "grok_gqa_v9_best.pth"
SEQUENCE_LEN = 32      # must match training

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
PAPER = os.getenv("APCA_API_PAPER", "true").lower() == "true"

trading_client = TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=PAPER)
data_client = CryptoHistoricalDataClient()

# --- POSITION TRACKING (in‑memory, avoid double buys) ---
positions = {symbol: False for symbol in SYMBOLS}
cooldown_until = 0.0

# --- DATABASE HELPERS ---
def log_error_to_db(bot_name, error_msg):
    db_url = os.getenv('DATABASE_URL')
    if not db_url: return
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO bot_errors (bot_name, error_message) VALUES (%s, %s)", (bot_name, str(error_msg)))
                conn.commit()
    except Exception as e:
        logger.error(f"Critical failure logging error to DB: {e}")

def check_status(bot_name):
    db_url = os.getenv('DATABASE_URL')
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO bot_status (bot_name, last_update, status) 
                    VALUES (%s, NOW(), 'RUNNING') 
                    ON CONFLICT (bot_name) DO UPDATE SET last_update = NOW(), status = EXCLUDED.status;
                """, (bot_name,))
                cur.execute("SELECT status FROM bot_status WHERE bot_name = %s", (bot_name,))
                result = cur.fetchone()
                conn.commit()
                if result and result[0] == 'STOP':
                    sys.exit(0)
    except Exception as e:
        logger.error(f"Heartbeat failed: {e}")

def sync_trade_to_db(bot_name, side, price, qty, symbol, order_id, fee=0.0):
    try:
        price, qty = float(price or 0.0), float(qty or 0.0)
        db_url = os.getenv('DATABASE_URL')
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trades 
                    (bot_name, exchange, symbol, side, price, quantity, value, fee, order_id, timestamp) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW());
                """, (bot_name, 'Alpaca', symbol, side, price, qty, price * qty, fee, str(order_id)))
                conn.commit()
    except Exception as e:
        log_error_to_db(bot_name, f"Database sync failed: {e}")

def register_order_in_db(bot_name, order_id, symbol, side, price):
    db_url = os.getenv('DATABASE_URL')
    if not db_url: return
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO bot_orders (order_id, bot_name, symbol, side, price, status)
                    VALUES (%s, %s, %s, %s, %s, 'OPEN')
                """, (str(order_id), bot_name, symbol, side, float(price)))
                conn.commit()
    except Exception as e:
        logger.error(f"Failed to register order in DB: {e}")

def execute_trade(bot_name, symbol, side, qty):
    """Place market order and log it."""
    try:
        order = trading_client.submit_order(
            order_data=MarketOrderRequest(
                symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.GTC
            )
        )
        register_order_in_db(bot_name, order.id, symbol, side.value, 0.0)
        logger.info(f"✅ Placed {side.value} order for {symbol} | Qty: {qty:.6f} | Order ID: {order.id}")
        return order
    except Exception as e:
        log_error_to_db(bot_name, f"Trade execution failed for {symbol}: {e}")
        return None

async def sync_filled_orders(bot_name):
    """Check open orders and mark filled ones as CLOSED, log to trades."""
    db_url = os.getenv('DATABASE_URL')
    if not db_url: return
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT order_id, symbol FROM bot_orders WHERE bot_name = %s AND status = 'OPEN'", (bot_name,))
            for oid, symbol in cur.fetchall():
                try:
                    alpaca_order = trading_client.get_order_by_id(oid)
                    if alpaca_order.status == 'filled':
                        cur.execute("UPDATE bot_orders SET status = 'CLOSED' WHERE order_id = %s", (oid,))
                        conn.commit()
                        # Use 0.0 fee (Alpaca doesn't provide per‑order fee in standard call)
                        sync_trade_to_db(
                            bot_name, alpaca_order.side.value,
                            alpaca_order.filled_avg_price, alpaca_order.filled_qty,
                            symbol, oid, fee=0.0
                        )
                        # Update in‑memory position flag (only if sell order)
                        if alpaca_order.side == OrderSide.SELL:
                            positions[symbol] = False
                except Exception as e:
                    logger.error(f"Error syncing order {oid}: {e}")

async def get_features_for_symbol(symbol, predictor, seq_len=SEQUENCE_LEN):
    """
    Fetch the last `seq_len` 5‑minute bars, convert to features expected by MLPredictor.
    Assumes `predictor` has a `prepare_features` method or similar.
    Adjust based on your actual `MLPredictor` interface.
    """
    # Use 5‑minute bars for fast signals
    request_symbol = symbol.replace("/", "")  # Alpaca data needs e.g. "BTCUSD"
    end = datetime.now()
    start = end - timedelta(hours=6)   # enough for 32 * 5min = 160min
    request = CryptoBarsRequest(
        symbol_or_symbols=request_symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        limit=200
    )
    bars = data_client.get_crypto_bars(request).data.get(request_symbol, [])
    if len(bars) < seq_len:
        logger.warning(f"Insufficient bars for {symbol}: {len(bars)} < {seq_len}")
        return None

    # Convert to DataFrame and resample to 5 minutes
    df = pd.DataFrame([{'timestamp': b.timestamp, 'close': float(b.close),
                        'high': float(b.high), 'low': float(b.low), 'volume': float(b.volume)} for b in bars])
    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    ohlc_5 = df.resample('5min').agg({'close': 'last', 'high': 'max', 'low': 'min', 'volume': 'sum'}).dropna()
    if len(ohlc_5) < seq_len:
        logger.warning(f"Not enough 5‑min bars for {symbol}: {len(ohlc_5)}")
        return None

    # Use last `seq_len` rows
    features = ohlc_5[['close', 'high', 'low', 'volume']].values[-seq_len:]
    # The MLPredictor likely expects a numpy array of shape (1, seq_len, num_features)
    # or (seq_len, num_features). Adjust according to your predictor.
    # I assume it takes a 2D array (seq_len, features) and returns a binary signal.
    return features

async def run_trading_mode(bot_name):
    global cooldown_until, positions
    predictor = MLPredictor(model_path=MODEL_PATH, seq_len=SEQUENCE_LEN)
    logger.info("MLPredictor loaded. Starting trading loop...")

    while True:
        try:
            check_status(bot_name)
            await sync_filled_orders(bot_name)

            # Global cooldown after any trade (avoid rapid fire)
            if time.time() < cooldown_until:
                logger.info(f"Global cooldown active, sleeping {cooldown_until - time.time():.0f}s")
                await asyncio.sleep(60)
                continue

            for symbol in SYMBOLS:
                # Skip if already in position (to avoid multiple buys)
                if positions.get(symbol, False):
                    logger.debug(f"Already in position for {symbol}, skipping")
                    continue

                # Get features for this symbol
                features = await get_features_for_symbol(symbol, predictor)
                if features is None:
                    continue

                # Get prediction from model
                # I assume MLPredictor has a method .predict(features) returning 0 or 1
                # If it returns a probability > threshold, adjust accordingly.
                try:
                    signal = predictor.predict(features)  # adjust method name if needed
                except AttributeError:
                    # Fallback: try to call .forward or .__call__
                    signal = predictor(features)

                # Signal interpretation (1 = buy, 0 = hold)
                if signal == 1:
                    current_price = features[-1][0]  # last closing price
                    qty = ORDER_AMOUNT / current_price
                    order = execute_trade(bot_name, symbol, OrderSide.BUY, qty)
                    if order:
                        positions[symbol] = True
                        cooldown_until = time.time() + 300  # 5 min cooldown after any trade
                        logger.info(f"🎯 BUY signal for {symbol} at {current_price:.2f}")
                    else:
                        logger.error(f"Order placement failed for {symbol}")
                else:
                    logger.debug(f"No buy signal for {symbol}")

                # Small delay between symbols to avoid rate limits
                await asyncio.sleep(2)

            # Also need to check for sell signals – but your model may only predict buys.
            # If the model can also predict sells, you'd add a similar loop to check positions and sell.
            # For now, only buy signals are used; sells will be handled manually or by a stop loss.
            # Optionally, you can add a trailing stop or time‑based exit.

            # Sleep 5 minutes before next full scan (adjust as needed)
            await asyncio.sleep(300)

        except Exception as e:
            error_msg = f"Main loop error: {e}"
            logger.error(error_msg)
            log_error_to_db(bot_name, error_msg)
            await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        asyncio.run(run_trading_mode(BOT_NAME))
    except Exception as e:
        log_error_to_db(BOT_NAME, f"FATAL CRASH: {e}")
        sys.exit(1)
