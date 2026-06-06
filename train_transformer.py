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

# --- Config ---
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

# --- NEW: Telemetry & Control Logic ---
def check_status():
    """Updates heartbeat and checks for a 'STOP' command in the database."""
    db_url = os.getenv('DATABASE_URL')
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        
        # 1. Update Heartbeat (Upsert)
        cur.execute("""
            INSERT INTO bot_status (bot_name, last_update, status)
            VALUES (%s, NOW(), 'RUNNING')
            ON CONFLICT (bot_name) 
            DO UPDATE SET last_update = NOW(), status = EXCLUDED.status;
        """, (BOT_NAME,))
        
        # 2. Check for Kill Switch
        cur.execute("SELECT status FROM bot_status WHERE bot_name = %s", (BOT_NAME,))
        result = cur.fetchone()
        
        conn.commit()
        cur.close()
        conn.close()
        
        if result and result[0] == 'STOP':
            logger.warning(f"🛑 Kill switch activated for {BOT_NAME}. Shutting down.")
            exit(0) # Terminate the bot process
            
    except Exception as e:
        logger.error(f"Database heartbeat failed: {e}")

# --- Existing Helpers ---
def sync_trade_to_db(side, raw_price, raw_qty, symbol, order_id):
    try:
        price = float(raw_price) if raw_price is not None else 0.0
        qty = float(raw_qty) if raw_qty is not None else 0.0
        order_id_str = str(order_id)
        
        db_url = os.getenv('DATABASE_URL')
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        query = "INSERT INTO trades (bot_name, exchange, symbol, side, price, quantity, value, order_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);"
        cursor.execute(query, (BOT_NAME, 'Alpaca', symbol, side, price, qty, price * qty, order_id_str))
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"✅ Synced {symbol} trade to database.")
    except Exception as e:
        logger.error(f"Database sync failed: {e}")

def load_trade_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f: return json.load(f)
        except: pass
    return {symbol: 0 for symbol in SYMBOLS}

def save_trade_state(state):
    with open(STATE_FILE, 'w') as f: json.dump(state, f)

last_trade_times = load_trade_state()

def get_current_qty(symbol):
    try:
        pos = trading_client.get_open_position(symbol)
        return float(pos.qty)
    except Exception: return 0.0

def execute_trade_signal(symbol, prediction_prob, current_qty):
    BUY_THRESHOLD = 0.54
    SELL_THRESHOLD = 0.46
    
    if time.time() - last_trade_times.get(symbol, 0) < COOLDOWN_SECONDS: return

    if prediction_prob >= BUY_THRESHOLD and current_qty <= 0:
        logger.info(f"🔮 Bullish {symbol}. Attempting BUY.")
        try:
            order = trading_client.submit_order(MarketOrderRequest(
                symbol=symbol, notional=ORDER_AMOUNT, side=OrderSide.BUY, time_in_force=TimeInForce.GTC
            ))
            sync_trade_to_db('BUY', order.filled_avg_price, order.filled_qty, symbol, order.id)
            last_trade_times[symbol] = time.time()
            save_trade_state(last_trade_times)
        except Exception as e: logger.error(f"❌ BUY ERROR {symbol}: {str(e)}")
            
    elif prediction_prob <= SELL_THRESHOLD and current_qty > 0:
        logger.info(f"🔮 Bearish {symbol}. Liquidating.")
        try:
            order = trading_client.submit_order(MarketOrderRequest(
                symbol=symbol, qty=str(current_qty), side=OrderSide.SELL, time_in_force=TimeInForce.GTC
            ))
            sync_trade_to_db('SELL', order.filled_avg_price, order.filled_qty, symbol, order.id)
            last_trade_times[symbol] = time.time()
            save_trade_state(last_trade_times)
        except Exception as e: logger.error(f"❌ SELL ERROR {symbol}: {e}")

async def run_trading_mode():
    predictor = MLPredictor(model_path=MODEL_PATH, seq_len=SEQ_LEN)
    while True:
        # Check heartbeat and kill switch at the start of every cycle
        check_status()
        
        for symbol in SYMBOLS:
            try:
                df = data_client.get_crypto_bars(CryptoBarsRequest(
                    symbol_or_symbols=[symbol], timeframe=TIMEFRAME, 
                    start=datetime.now(timezone.utc) - timedelta(days=3)
                )).df.xs(symbol)
                prob = predictor.predict(df)
                execute_trade_signal(symbol, prob, get_current_qty(symbol))
            except Exception as e: logger.error(f"Error: {e}")
        await asyncio.sleep(900)

if __name__ == "__main__":
    asyncio.run(run_trading_mode())
