
#!/usr/bin/env python3
import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Assuming these are your custom local modules
from ml_predictor import MLPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# --- Configurations ---
SYMBOLS = ["BTC/USD", "ETH/USD", "LTC/USD", "DOGE/USD"]
MAX_HOLDINGS = {"BTC/USD": 0.003, "ETH/USD": 0.05, "LTC/USD": 5.0, "DOGE/USD": 500.0}
COOLDOWN_SECONDS = 3600 # 1 hour
ORDER_AMOUNT = 11.0 

TIMEFRAME = TimeFrame.Hour
SEQ_LEN = 32
MODEL_PATH = "/app/data/grok_gqa_v9_best.pth" if os.path.exists("/app/data") else "grok_gqa_v9_best.pth"

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
PAPER = os.getenv("APCA_API_PAPER", "true").lower() == "true"

data_client = CryptoHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)
trading_client = TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=PAPER)

last_trade_times = {symbol: 0 for symbol in SYMBOLS}

def get_current_qty(symbol):
    """Safely check if we currently hold the symbol."""
    try:
        # Alpaca-py requires "BTC/USD" format for position queries
        pos = trading_client.get_open_position(symbol)
        return float(pos.qty)
    except Exception:
        # If position doesn't exist, Alpaca raises an error; return 0.0
        return 0.0

def execute_trade_signal(symbol, prediction_prob, current_qty, df):
    BUY_THRESHOLD = 0.54
    SELL_THRESHOLD = 0.46
    
    # 1. Cooldown Check
    if time.time() - last_trade_times[symbol] < COOLDOWN_SECONDS:
        return

    # 2. BUY Logic: Only buy if we hold 0 (or less)
    if prediction_prob >= BUY_THRESHOLD and current_qty <= 0:
        logger.info(f"🔮 Bullish {symbol} ({prediction_prob:.2%}). Executing BUY.")
        try:
            trading_client.submit_order(MarketOrderRequest(
                symbol=symbol, 
                notional=ORDER_AMOUNT, 
                side=OrderSide.BUY, 
                time_in_force=TimeInForce.GTC
            ))
            last_trade_times[symbol] = time.time()
        except Exception as e:
            logger.error(f"Failed to place BUY for {symbol}: {e}")
            
    # 3. SELL Logic: Only sell if we have a positive position
    elif prediction_prob <= SELL_THRESHOLD and current_qty > 0:
        logger.info(f"🔮 Bearish {symbol} ({prediction_prob:.2%}). Liquidating {current_qty} units.")
        try:
            trading_client.submit_order(MarketOrderRequest(
                symbol=symbol, 
                qty=str(current_qty), 
                side=OrderSide.SELL, 
                time_in_force=TimeInForce.GTC
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
                df = data_client.get_crypto_bars(CryptoBarsRequest(
                    symbol_or_symbols=[symbol], timeframe=TIMEFRAME, 
                    start=datetime.now(timezone.utc) - timedelta(days=3)
                )).df.xs(symbol)
                
                prob = predictor.predict(df)
                
                # Verify actual position to prevent duplicate buys
                current_qty = get_current_qty(symbol)
                logger.info(f"DEBUG: {symbol} | Prob: {prob:.2f} | Current Holding: {current_qty}")
                
                execute_trade_signal(symbol, prob, current_qty, df)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        await asyncio.sleep(900) # 15-minute wait

# ... (main and refit tasks remain the same)
