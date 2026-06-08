#!/usr/bin/env python3
import asyncio
import logging
import os
import sys
import time
import psycopg2
import pandas as pd
import numpy as np
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
ORDER_AMOUNT = 50.0
MODEL_PATH = "/app/data/grok_gqa_v9_best.pth" if os.path.exists("/app/data") else "grok_gqa_v9_best.pth"
SEQUENCE_LEN = 32

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
PAPER = os.getenv("APCA_API_PAPER", "true").lower() == "true"

trading_client = TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=PAPER)
data_client = CryptoHistoricalDataClient()

positions = {symbol: False for symbol in SYMBOLS}
cooldown_until = 0.0

# ---------- DATABASE HELPERS (unchanged) ----------
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
                        sync_trade_to_db(
                            bot_name, alpaca_order.side.value,
                            alpaca_order.filled_avg_price, alpaca_order.filled_qty,
                            symbol, oid, fee=0.0
                        )
                        if alpaca_order.side == OrderSide.SELL:
                            positions[symbol] = False
                except Exception as e:
                    logger.error(f"Error syncing order {oid}: {e}")

async def get_clean_ohlcv_dataframe(symbol):
    """
    Fetch minute bars, resample to 5min, and return a DataFrame with columns
    ['open','high','low','close','volume'] that contains no None or NaN.
    """
    end = datetime.now()
    start = end - timedelta(hours=6)
    request = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        limit=500
    )
    bars = data_client.get_crypto_bars(request).data.get(symbol, [])
    if len(bars) < SEQUENCE_LEN:
        logger.warning(f"Insufficient minute bars for {symbol}: {len(bars)} < {SEQUENCE_LEN}")
        return None

    # Build DataFrame and convert None to nan
    data = []
    for b in bars:
        # Ensure each field is a float; if None, use nan
        data.append({
            'timestamp': b.timestamp,
            'open': float(b.open) if b.open is not None else np.nan,
            'high': float(b.high) if b.high is not None else np.nan,
            'low': float(b.low) if b.low is not None else np.nan,
            'close': float(b.close) if b.close is not None else np.nan,
            'volume': float(b.volume) if b.volume is not None else np.nan,
        })
    df = pd.DataFrame(data)
    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Resample to 5 minutes
    ohlc_5 = df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    # Fill missing values: forward fill prices, volume fill with 0
    ohlc_5 = ohlc_5.fillna(method='ffill')
    ohlc_5['volume'] = ohlc_5['volume'].fillna(0.0)

    # Drop any rows that still have NaN (should not happen after ffill)
    ohlc_5.dropna(inplace=True)

    if len(ohlc_5) < SEQUENCE_LEN:
        logger.warning(f"Not enough clean 5‑min bars for {symbol}: {len(ohlc_5)}")
        return None

    # Keep only the last SEQUENCE_LEN rows
    ohlc_5 = ohlc_5.iloc[-SEQUENCE_LEN:]

    # Double‑check no None remains
    for col in ['open','high','low','close','volume']:
        if ohlc_5[col].isnull().any():
            logger.warning(f"Still NaN in {col} for {symbol}, filling with 0")
            ohlc_5[col] = ohlc_5[col].fillna(0.0)

    return ohlc_5

async def run_trading_mode(bot_name):
    global cooldown_until, positions
    predictor = MLPredictor(model_path=MODEL_PATH, seq_len=SEQUENCE_LEN)
    logger.info("MLPredictor loaded. Starting trading loop...")

    while True:
        try:
            check_status(bot_name)
            await sync_filled_orders(bot_name)

            if time.time() < cooldown_until:
                logger.info(f"Global cooldown active, sleeping {cooldown_until - time.time():.0f}s")
                await asyncio.sleep(60)
                continue

            for symbol in SYMBOLS:
                if positions.get(symbol, False):
                    logger.debug(f"Already in position for {symbol}, skipping")
                    continue

                df = await get_clean_ohlcv_dataframe(symbol)
                if df is None:
                    continue

                # Predict using the cleaned DataFrame
                try:
                    signal = predictor.predict(df)   # returns float 0-1
                except Exception as e:
                    logger.error(f"Prediction error for {symbol}: {e}")
                    continue

                # Use thresholds from MLPredictor docstring
                if signal > 0.51:
                    current_price = df['close'].iloc[-1]
                    qty = ORDER_AMOUNT / current_price
                    order = execute_trade(bot_name, symbol, OrderSide.BUY, qty)
                    if order:
                        positions[symbol] = True
                        cooldown_until = time.time() + 300
                        logger.info(f"🎯 BUY signal for {symbol} at {current_price:.2f} (prob={signal:.3f})")
                elif signal < 0.49 and positions.get(symbol, False):
                    # Optional auto‑sell on bearish signal
                    try:
                        position = trading_client.get_position(symbol)
                        qty = float(position.qty)
                        if qty > 0:
                            order = execute_trade(bot_name, symbol, OrderSide.SELL, qty)
                            if order:
                                positions[symbol] = False
                                cooldown_until = time.time() + 300
                                logger.info(f"🔻 SELL signal for {symbol} at {df['close'].iloc[-1]:.2f} (prob={signal:.3f})")
                    except Exception:
                        positions[symbol] = False
                else:
                    logger.debug(f"No trade for {symbol} (signal={signal:.3f})")

                await asyncio.sleep(2)   # small delay between symbols

            await asyncio.sleep(300)   # scan every 5 minutes

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
