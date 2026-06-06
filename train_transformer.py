#!/usr/bin/env python3
import asyncio
import logging
import os
import time
import json
import psycopg2 
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv 

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from ml_predictor import MLPredictor

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# --- NEW: ERROR LOGGING HELPER ---
def log_error_to_db(bot_name, error_msg):
    """Logs errors to the bot_errors table."""
    db_url = os.getenv('DATABASE_URL')
    if not db_url: return
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO bot_errors (bot_name, error_message) VALUES (%s, %s)",
                    (bot_name, str(error_msg))
                )
                conn.commit()
    except Exception as e:
        logger.error(f"Critical failure logging error to DB: {e}")

# --- Telemetry & Control Logic ---
def check_status(bot_name):
    """Updates heartbeat and checks for a 'STOP' command."""
    db_url = os.getenv('DATABASE_URL')
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO bot_status (bot_name, last_update, status)
                    VALUES (%s, NOW(), 'RUNNING')
                    ON CONFLICT (bot_name) 
                    DO UPDATE SET last_update = NOW(), status = EXCLUDED.status;
                """, (bot_name,))
                cur.execute("SELECT status FROM bot_status WHERE bot_name = %s", (bot_name,))
                result = cur.fetchone()
                conn.commit()
                if result and result[0] == 'STOP':
                    logger.warning(f"🛑 Kill switch activated for {bot_name}. Shutting down.")
                    exit(0)
    except Exception as e:
        logger.error(f"Database heartbeat failed: {e}")
        log_error_to_db(bot_name, f"Heartbeat failed: {e}")

# --- Existing Helpers ---
def sync_trade_to_db(bot_name, side, raw_price, raw_qty, symbol, order_id):
    try:
        price = float(raw_price) if raw_price is not None else 0.0
        qty = float(raw_qty) if raw_qty is not None else 0.0
        db_url = os.getenv('DATABASE_URL')
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO trades (bot_name, exchange, symbol, side, price, quantity, value, order_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);", 
                            (bot_name, 'Alpaca', symbol, side, price, qty, price * qty, str(order_id)))
                conn.commit()
    except Exception as e:
        error_msg = f"Database sync failed: {e}"
        logger.error(error_msg)
        log_error_to_db(bot_name, error_msg)

# ... (load_trade_state, save_trade_state, get_current_qty, execute_trade_signal remain similar)

async def run_trading_mode(bot_name):
    predictor = MLPredictor(model_path=MODEL_PATH, seq_len=SEQ_LEN)
    while True:
        try:
            check_status(bot_name)
            for symbol in SYMBOLS:
                try:
                    df = data_client.get_crypto_bars(CryptoBarsRequest(
                        symbol_or_symbols=[symbol], timeframe=TIMEFRAME, 
                        start=datetime.now(timezone.utc) - timedelta(days=3)
                    )).df.xs(symbol)
                    prob = predictor.predict(df)
                    execute_trade_signal(symbol, prob, get_current_qty(symbol))
                except Exception as e: 
                    error_msg = f"Loop cycle error for {symbol}: {e}"
                    logger.error(error_msg)
                    log_error_to_db(bot_name, error_msg)
        except Exception as e:
            logger.critical(f"Critical loop crash: {e}")
            log_error_to_db(bot_name, f"Critical loop crash: {e}")
        await asyncio.sleep(900)

if __name__ == "__main__":
    # Define/Fetch variables here so they exist if the try-block crashes
    bot_name = os.getenv('BOT_NAME', 'Bot_Alpha')
    
    try:
        # We define these here so they are in scope if run_trading_mode fails
        asyncio.run(run_trading_mode(bot_name))
    except Exception as e:
        error_msg = f"FATAL SYSTEM CRASH: {e}"
        logger.critical(error_msg)
        # Use a fallback name if the bot name wasn't set
        log_error_to_db(bot_name, error_msg)
        sys.exit(1)
