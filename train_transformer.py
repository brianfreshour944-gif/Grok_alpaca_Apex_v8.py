
#!/usr/bin/env python3
import asyncio
import logging
import os
import time
import json
import psycopg2
import sys
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

# --- GLOBAL CONFIGURATION ---
BOT_NAME = os.getenv("BOT_NAME", "Bot_Alpha")
SYMBOLS = ["BTC/USD", "ETH/USD", "LTC/USD", "DOGE/USD"]
COOLDOWN_SECONDS = 3600
ORDER_AMOUNT = 50.0
TIMEFRAME = TimeFrame.Hour
SEQ_LEN = 32
MODEL_PATH = "/app/data/grok_gqa_v9_best.pth" if os.path.exists("/app/data") else "grok_gqa_v9_best.pth"
STATE_FILE = "/app/data/trade_state.json"

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
PAPER = os.getenv("APCA_API_PAPER", "true").lower() == "true"

data_client = CryptoHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)
trading_client = TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=PAPER)

# --- HELPER FUNCTIONS (Defined before use) ---
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
        log_error_to_db(bot_name, f"Heartbeat failed: {e}")

def sync_trade_to_db(bot_name, side, raw_price, raw_qty, symbol, order_id):
    try:
        price, qty = float(raw_price or 0.0), float(raw_qty or 0.0)
        db_url = os.getenv('DATABASE_URL')
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO trades (bot_name, exchange, symbol, side, price, quantity, value, order_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);", 
                            (bot_name, 'Alpaca', symbol, side, price, qty, price * qty, str(order_id)))
                conn.commit()
    except Exception as e:
        log_error_to_db(bot_name, f"Database sync failed: {e}")

def get_current_qty(symbol):
    try: return float(trading_client.get_open_position(symbol).qty)
    except: return 0.0

def execute_trade_signal(symbol, prediction_prob, current_qty):
    # Implementation of your trade logic...
    pass

# --- MAIN LOOP (Now knows about the helpers above) ---
async def run_trading_mode(bot_name):
    predictor = MLPredictor(model_path=MODEL_PATH, seq_len=SEQ_LEN)
    while True:
        try:
            check_status(bot_name)
            for symbol in SYMBOLS:
                # ... (rest of your logic using execute_trade_signal)
                pass
        except Exception as e:
            log_error_to_db(bot_name, f"Loop error: {e}")
        await asyncio.sleep(900)

if __name__ == "__main__":
    try:
        asyncio.run(run_trading_mode(BOT_NAME))
    except Exception as e:
        log_error_to_db(BOT_NAME, f"FATAL CRASH: {e}")
        sys.exit(1)
