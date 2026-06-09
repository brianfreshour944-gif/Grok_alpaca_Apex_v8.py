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

from ml_predictor import GrokGQA_Transformer, FEATURE_COLS, MLPredictor

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# --- GLOBAL CONFIGURATION ---
BOT_NAME = os.getenv("BOT_NAME", "Grok_Alpaca_Apex_v8")
SYMBOLS = ["BTC/USD", "ETH/USD", "LTC/USD", "DOGE/USD"]
ORDER_AMOUNT = 50.0
MAX_PORTFOLIO_VALUE = 200.0  # HARD LIMIT: Never hold > $200 total in crypto
MAX_SINGLE_TRADE_USD = 100.0 # Safety cap per order
MODEL_PATH = "/app/data/grok_gqa_v9_best.pth" if os.path.exists("/app/data") else "grok_gqa_v9_best.pth"
SEQUENCE_LEN = 32

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
PAPER = os.getenv("APCA_API_PAPER", "true").lower() == "true"

trading_client = TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=PAPER)
data_client = CryptoHistoricalDataClient()
cooldown_until = {symbol: 0.0 for symbol in SYMBOLS}

# ---------- SAFETY HELPERS ----------
def get_total_portfolio_value():
    """Returns the total dollar value of all crypto positions."""
    try:
        positions = trading_client.get_all_positions()
        return sum(float(p.market_value) for p in positions)
    except Exception as e:
        logger.error(f"Error fetching portfolio value: {e}")
        return 999999.0 # Fail-safe: return huge value to block new buys

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

# --- MAIN TRADING LOOP ---
async def run_trading_mode(bot_name):
    global cooldown_until
    predictor = MLPredictor(model_path=MODEL_PATH, seq_len=SEQUENCE_LEN)
    logger.info("MLPredictor loaded. Starting trading loop...")

    while True:
        try:
            for symbol in SYMBOLS:
                if time.time() < cooldown_until.get(symbol, 0.0):
                    continue

                # 1. State Check
                has_position = False
                qty_held = 0.0
                try:
                    position = trading_client.get_position(symbol)
                    if float(position.qty) > 0:
                        has_position = True
                        qty_held = float(position.qty)
                except:
                    has_position = False

                # 2. Prediction
                # (Assuming get_clean_ohlcv_dataframe exists)
                df = await get_clean_ohlcv_dataframe(symbol)
                if df is None: continue
                signal = predictor.predict(df)
                current_price = df['close'].iloc[-1]

                # 3. SELL Logic
                if has_position and signal < 0.49:
                    logger.info(f"🔻 SELL signal for {symbol} (signal={signal:.3f})")
                    if execute_trade(bot_name, symbol, OrderSide.SELL, qty_held):
                        cooldown_until[symbol] = time.time() + 3600
                    continue

                # 4. BUY Logic (with Guardian Layer)
                if not has_position and signal > 0.51:
                    # Guardian: Check Total Exposure
                    if get_total_portfolio_value() >= MAX_PORTFOLIO_VALUE:
                        logger.warning(f"⚠️ Portfolio cap reached. Skipping BUY for {symbol}")
                        continue

                    qty = ORDER_AMOUNT / current_price
                    # Guardian: Check Single Order Size
                    if (qty * current_price) > MAX_SINGLE_TRADE_USD:
                        logger.error(f"❌ Trade too large! Aborting BUY for {symbol}")
                        continue

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
