#!/usr/bin/env python3
import asyncio
import logging
import os
import psycopg2
import sys
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from ml_predictor import MLPredictor

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# --- GLOBAL CONFIGURATION ---
BOT_NAME = os.getenv("BOT_NAME", "Grok_Alpaca_Apex_v8")
SYMBOLS = ["BTC/USD", "ETH/USD", "LTC/USD", "DOGE/USD"]
ORDER_AMOUNT = 50.0
MODEL_PATH = "/app/data/grok_gqa_v9_best.pth" if os.path.exists("/app/data") else "grok_gqa_v9_best.pth"

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
PAPER = os.getenv("APCA_API_PAPER", "true").lower() == "true"

trading_client = TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=PAPER)

# --- DATABASE HELPERS ---
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
                cur.execute("INSERT INTO bot_status (bot_name, last_update, status) VALUES (%s, NOW(), 'RUNNING') ON CONFLICT (bot_name) DO UPDATE SET last_update = NOW(), status = EXCLUDED.status;", (bot_name,))
                cur.execute("SELECT status FROM bot_status WHERE bot_name = %s", (bot_name,))
                result = cur.fetchone()
                conn.commit()
                if result and result[0] == 'STOP':
                    sys.exit(0)
    except Exception as e:
        logger.error(f"Heartbeat failed: {e}")

def sync_trade_to_db(bot_name, side, raw_price, raw_qty, symbol, order_id, fee=0.0):
    try:
        price, qty = float(raw_price or 0.0), float(raw_qty or 0.0)
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

# --- UPDATED TRADE LOGIC ---
def execute_trade_signal(bot_name, symbol, side, qty):
    db_url = os.getenv('DATABASE_URL')
    try:
        order = trading_client.submit_order(
            order_data=MarketOrderRequest(
                symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.GTC
            )
        )
        if db_url:
            with psycopg2.connect(db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO bot_orders (order_id, bot_name, symbol, side, price, status)
                        VALUES (%s, %s, %s, %s, %s, 'OPEN')
                    """, (str(order.id), bot_name, symbol, side.value, 0.0))
                    conn.commit()
        logger.info(f"Placed {side} order for {symbol}. Order ID: {order.id}")
        return order
    except Exception as e:
        log_error_to_db(bot_name, f"Trade execution failed: {e}")
        return None

# --- MAIN LOOP ---
async def run_trading_mode(bot_name):
    predictor = MLPredictor(model_path=MODEL_PATH, seq_len=32)
    while True:
        try:
            check_status(bot_name)
            
            # SYNC FILLED ORDERS
            db_url = os.getenv('DATABASE_URL')
            if db_url:
                with psycopg2.connect(db_url) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT order_id FROM bot_orders WHERE bot_name = %s AND status = 'OPEN'", (bot_name,))
                        open_orders = cur.fetchall()
                        for (oid,) in open_orders:
                            alpaca_order = trading_client.get_order_by_id(oid)
                            if alpaca_order.status == 'filled':
                                cur.execute("UPDATE bot_orders SET status = 'CLOSED' WHERE order_id = %s", (oid,))
                                conn.commit()
                                # ALPACA FEE: If your account has commission info, fetch it here. Otherwise use 0.0.
                                fee = 0.0 
                                sync_trade_to_db(bot_name, alpaca_order.side.value, alpaca_order.filled_avg_price, alpaca_order.filled_qty, alpaca_order.symbol, oid, fee)
            
        except Exception as e:
            log_error_to_db(bot_name, f"Loop error: {e}")
        await asyncio.sleep(900)

if __name__ == "__main__":
    try:
        asyncio.run(run_trading_mode(BOT_NAME))
    except Exception as e:
        log_error_to_db(BOT_NAME, f"FATAL CRASH: {e}")
        sys.exit(1)
