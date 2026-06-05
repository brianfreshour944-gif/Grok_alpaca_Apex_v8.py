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

TIMEFRAME = TimeFrame.Hour
SEQ_LEN = 32
MODEL_PATH = "/app/data/grok_gqa_v9_best.pth" if os.path.exists("/app/data") else "grok_gqa_v9_best.pth"
SCALER_PATH = os.path.join(os.path.dirname(MODEL_PATH) if os.path.dirname(MODEL_PATH) else "", 'feature_scaler.pkl')

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
PAPER = os.getenv("APCA_API_PAPER", "true").lower() == "true"

data_client = CryptoHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)
trading_client = TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=PAPER)

# ... [Keep FinancialTimeSeriesDataset and train_model as is] ...

def execute_trade_signal(symbol, prediction_prob, current_qty):
    """Executes market orders based on signal boundary values with safety checks."""
    BUY_THRESHOLD = 0.54
    SELL_THRESHOLD = 0.46
    norm_symbol = symbol.replace("/", "")
    
    # Cooldown Check
    if time.time() - last_trade_times[symbol] < COOLDOWN_SECONDS:
        return

    # Buy logic: Signal is high AND we are below our max limit
    if prediction_prob >= BUY_THRESHOLD and current_qty < MAX_HOLDINGS.get(symbol, 0):
        logger.info(f"🔮 Bullish {symbol} ({prediction_prob:.2%}). Executing BUY.")
        try:
            trading_client.submit_order(MarketOrderRequest(
                symbol=norm_symbol, qty=0.001, side=OrderSide.BUY, time_in_force=TimeInForce.GTC
            ))
            last_trade_times[symbol] = time.time()
        except Exception as e:
            logger.error(f"Failed to place BUY for {symbol}: {e}")
            
    # Sell logic: Signal is low AND we have a position to liquidate
    elif prediction_prob <= SELL_THRESHOLD and current_qty > 0:
        logger.info(f"🔮 Bearish {symbol} ({prediction_prob:.2%}). Liquidating.")
        try:
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
        for symbol in SYMBOLS:
            try:
                # 1. Fetch live data
                df = data_client.get_crypto_bars(CryptoBarsRequest(
                    symbol_or_symbols=[symbol], timeframe=TIMEFRAME, 
                    start=datetime.now(timezone.utc) - timedelta(days=3)
                )).df.xs(symbol)
                
                # 2. Get prediction
                prob = predictor.predict(df)
                
                # 3. Get position for this specific asset
                try:
                    pos = trading_client.get_open_position(symbol.replace("/", ""))
                    current_qty = float(pos.qty)
                except:
                    current_qty = 0.0
                
                # 4. Signal
                execute_trade_signal(symbol, prob, current_qty)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        await asyncio.sleep(900)

async def main():
    # Initial setup ... [Keep existing initialization logic] ...
    await asyncio.gather(run_trading_mode(), nightly_refit_task())

if __name__ == "__main__":
    asyncio.run(main())
