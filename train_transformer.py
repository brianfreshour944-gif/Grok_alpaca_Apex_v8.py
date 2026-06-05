#!/usr/bin/env python3
import asyncio
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
import os
import joblib
import time
from datetime import datetime, timedelta, timezone
from sklearn.preprocessing import StandardScaler

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from feature_engineering import add_features, FEATURE_COLS
from ml_predictor import GrokGQA_Transformer, MLPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# --- Configurations ---
SYMBOLS = ["BTC/USD", "ETH/USD", "LTC/USD", "DOGE/USD"]
MAX_HOLDINGS = {
    "BTC/USD": 0.003,
    "ETH/USD": 0.05,
    "LTC/USD": 5.0,
    "DOGE/USD": 500.0
}
COOLDOWN_SECONDS = 3600
last_trade_times = {symbol: 0 for symbol in SYMBOLS}
entry_prices = {symbol: 0.0 for symbol in SYMBOLS}
ORDER_AMOUNT = 11.0 

TIMEFRAME = TimeFrame.Hour
SEQ_LEN = 32
MODEL_PATH = "/app/data/grok_gqa_v9_best.pth" if os.path.exists("/app/data") else "grok_gqa_v9_best.pth"
SCALER_PATH = os.path.join(os.path.dirname(MODEL_PATH) if os.path.dirname(MODEL_PATH) else "", 'feature_scaler.pkl')

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
PAPER = os.getenv("APCA_API_PAPER", "true").lower() == "true"

data_client = CryptoHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)
trading_client = TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=PAPER)

# --- Logic Functions ---

def execute_trade_signal(symbol, prediction_prob, current_qty, df):
    BUY_THRESHOLD = 0.54
    SELL_THRESHOLD = 0.46
    norm_symbol = symbol.replace("/", "")
    latest_price = float(df['close'].iloc[-1])
    
    if time.time() - last_trade_times[symbol] < COOLDOWN_SECONDS:
        return

    # BUY Logic
    if prediction_prob >= BUY_THRESHOLD and current_qty < MAX_HOLDINGS.get(symbol, 0):
        logger.info(f"🔮 Bullish {symbol} ({prediction_prob:.2%}). Executing BUY.")
        try:
            trading_client.submit_order(MarketOrderRequest(
                symbol=norm_symbol, notional=ORDER_AMOUNT, side=OrderSide.BUY, time_in_force=TimeInForce.GTC
            ))
            entry_prices[symbol] = latest_price
            last_trade_times[symbol] = time.time()
        except Exception as e:
            logger.error(f"Failed to place BUY for {symbol}: {e}")
            
    # SELL Logic
    elif prediction_prob <= SELL_THRESHOLD and current_qty > 0:
        logger.info(f"🔮 Bearish {symbol} ({prediction_prob:.2%}). Liquidating.")
        try:
            gain_loss = (latest_price - entry_prices[symbol]) * current_qty
            logger.info(f"💰 Realized P/L for {symbol}: ${gain_loss:.2f}")
            
            trading_client.submit_order(MarketOrderRequest(
                symbol=norm_symbol, qty=current_qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC
            ))
            last_trade_times[symbol] = time.time()
        except Exception as e:
            logger.error(f"Failed to place SELL for {symbol}: {e}")

async def run_trading_mode():
    logger.info("Entering multi-asset live inference mode...")
    predictor = MLPredictor(model_path=MODEL_PATH, seq_len=SEQ_LEN)
    
    while True:
        try:
            for symbol in SYMBOLS:
                try:
                    df = data_client.get_crypto_bars(CryptoBarsRequest(
                        symbol_or_symbols=[symbol], timeframe=TIMEFRAME, 
                        start=datetime.now(timezone.utc) - timedelta(days=3)
                    )).df.xs(symbol)
                    
                    prob = predictor.predict(df)
                    
                    try:
                        pos = trading_client.get_open_position(symbol.replace("/", ""))
                        current_qty = float(pos.qty)
                    except:
                        current_qty = 0.0
                    
                    # Passing df now to avoid API issues
                    execute_trade_signal(symbol, prob, current_qty, df)
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
        except Exception as loop_e:
            logger.error(f"Critical loop error: {loop_e}")
            
        await asyncio.sleep(900)

async def nightly_refit_task():
    while True:
        try:
            # Sleep 24 hours
            await asyncio.sleep(86400)
            logger.info("Initiating training...")
            await asyncio.to_thread(train_model, epochs=10, is_refit=True)
        except Exception as e:
            logger.error(f"Error in refit: {e}")
            await asyncio.sleep(300)

async def main():
    await asyncio.gather(run_trading_mode(), nightly_refit_task())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.critical(f"Bot crashed with: {e}", exc_info=True)
