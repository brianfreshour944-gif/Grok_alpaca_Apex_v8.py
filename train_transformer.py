
#!/usr/bin/env python3
import asyncio
import logging
import os
import sys
import time
import psycopg2
import pandas as pd
import numpy as np
import torch
import joblib
from datetime import datetime, timedelta
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

from ml_predictor import GrokGQA_Transformer, FEATURE_COLS

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

cooldown_until = {symbol: 0.0 for symbol in SYMBOLS}

# ---------- HELPERS ----------
def check_status(bot_name):
    db_url = os.getenv('DATABASE_URL')
    if not db_url: return
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT status FROM bot_status WHERE bot_name = %s", (bot_name,))
                result = cur.fetchone()
                if result and result[0] == 'STOP': sys.exit(0)
    except Exception as e:
        logger.error(f"Heartbeat failed: {e}")

def execute_trade(bot_name, symbol, side, qty):
    try:
        order = trading_client.submit_order(
            order_data=MarketOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.GTC)
        )
        logger.info(f"✅ Placed {side.value} order for {symbol} | Qty: {qty:.6f} | Order ID: {order.id}")
        return order
    except Exception as e:
        logger.error(f"Trade execution failed for {symbol}: {e}")
        return None

async def sync_filled_orders(bot_name):
    # Logic omitted for brevity, keeping your existing implementation
    pass

async def get_clean_ohlcv_dataframe(symbol):
    end = datetime.now()
    start = end - timedelta(hours=6)
    request = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, start=start, end=end, limit=500)
    bars = data_client.get_crypto_bars(request).data.get(symbol, [])
    if len(bars) < SEQUENCE_LEN: return None
    
    data = [{'timestamp': b.timestamp, 'open': float(b.open), 'high': float(b.high), 'low': float(b.low), 'close': float(b.close), 'volume': float(b.volume)} for b in bars]
    df = pd.DataFrame(data).set_index('timestamp')
    ohlc_5 = df.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    return ohlc_5.iloc[-SEQUENCE_LEN:].astype(float)

# --- TRADING LOOP ---
async def run_trading_mode(bot_name):
    global cooldown_until
    from ml_predictor import SafeMLPredictor # Assuming this class is imported/defined
    predictor = SafeMLPredictor(model_path=MODEL_PATH, seq_len=SEQUENCE_LEN)
    logger.info("Starting trading loop...")

    while True:
        try:
            check_status(bot_name)
            await sync_filled_orders(bot_name)

            for symbol in SYMBOLS:
                if time.time() < cooldown_until.get(symbol, 0.0):
                    continue

                # 1. Properly Indented & Corrected Position Check
                has_position = False
                qty_held = 0.0
                try:
                    position = trading_client.get_position(symbol)
                    if float(position.qty) > 0:
                        has_position = True
                        qty_held = float(position.qty)
                except:
                    has_position = False

                # 2. Strategy Logic
                df = await get_clean_ohlcv_dataframe(symbol)
                if df is None: continue
                signal = predictor.predict(df)
                current_price = df['close'].iloc[-1]

                # 3. Sell Logic
                if has_position and signal < 0.49:
                    logger.info(f"🔻 SELL signal for {symbol} (signal={signal:.3f})")
                    if execute_trade(bot_name, symbol, OrderSide.SELL, qty_held):
                        cooldown_until[symbol] = time.time() + 3600
                    continue

                # 4. Buy Logic
                if not has_position and signal > 0.51:
                    qty = ORDER_AMOUNT / current_price
                    logger.info(f"🎯 BUY signal for {symbol} at {current_price:.2f}")
                    if execute_trade(bot_name, symbol, OrderSide.BUY, qty):
                        cooldown_until[symbol] = time.time() + 600

                await asyncio.sleep(2)
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            await asyncio.sleep(30)

if __name__ == "__main__":
    asyncio.run(run_trading_mode(BOT_NAME))
