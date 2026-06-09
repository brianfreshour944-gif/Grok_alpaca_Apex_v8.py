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

# Only import the model architecture and constants, NOT the predictor
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

# ---------- SAFE FEATURE ENGINEERING (No None errors) ----------
def safe_add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all features with aggressive None -> 0 conversion."""
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    df = df.copy()

    df['returns'] = df['close'].pct_change().fillna(0.0)
    df['vol_14'] = df['returns'].rolling(window=14).std().fillna(0.0)

    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    df['rsi'] = rsi.fillna(50.0).replace([np.inf, -np.inf], 50.0)

    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['macd'] = macd_line - signal_line
    df['macd'] = df['macd'].fillna(0.0).replace([np.inf, -np.inf], 0.0)

    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().fillna(0.0)

    sma = df['close'].rolling(window=20).mean()
    std = df['close'].rolling(window=20).std()
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    df['bb_width'] = (upper - lower) / sma
    df['bb_width'] = df['bb_width'].fillna(0.0).replace([np.inf, -np.inf], 0.0)

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            df[col] = df[col].replace([np.inf, -np.inf], 0.0)

    return df[FEATURE_COLS]

# ---------- LOCAL SAFE PREDICTOR (no import from ml_predictor) ----------
class SafeMLPredictor:
    def __init__(self, model_path, seq_len=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = seq_len
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.input_dim = len(FEATURE_COLS)
        self.model = GrokGQA_Transformer(
            input_dim=self.input_dim, seq_len=seq_len,
            embed_dim=128, num_layers=8, num_q_heads=16, num_kv_heads=4, dropout=0.1
        ).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        logger.info(f"✅ Model weights loaded from {model_path}")
        scaler_path = os.path.join(os.path.dirname(model_path), 'feature_scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info(f"✅ Scaler loaded from {scaler_path}")
        else:
            self.scaler = None
            logger.warning("No scaler found; predictions will be unnormalized")

    def predict(self, df: pd.DataFrame) -> float:
        try:
            df = df.copy()
            required = ['open', 'high', 'low', 'close', 'volume']
            for col in required:
                if col not in df.columns:
                    df[col] = 0.0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            df = df.map(lambda x: 0.0 if x is None else x)

            df_features = safe_add_features(df)   # uses the safe function above

            data = df_features[FEATURE_COLS].tail(self.seq_len).values.astype(np.float32)
            if len(data) < self.seq_len:
                logger.warning(f"Insufficient rows after feature engineering: {len(data)}")
                return 0.5

            if self.scaler is not None:
                data = self.scaler.transform(data).astype(np.float32)

            x = torch.tensor(data).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = self.model(x).item()
            return float(pred)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.5

# ---------- DATABASE HELPERS (simplified, adjust as needed) ----------
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
    # (keep your existing implementation; omitted for brevity but include it here)
    pass

async def get_clean_ohlcv_dataframe(symbol):
    end = datetime.now()
    start = end - timedelta(hours=6)
    request = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, start=start, end=end, limit=500)
    bars = data_client.get_crypto_bars(request).data.get(symbol, [])
    if len(bars) < SEQUENCE_LEN: return None
    
    data = [{'timestamp': b.timestamp, 'open': float(b.open), 'high': float(b.high), 'low': float(b.low), 
             'close': float(b.close), 'volume': float(b.volume)} for b in bars]
    df = pd.DataFrame(data).set_index('timestamp')
    ohlc_5 = df.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    return ohlc_5.iloc[-SEQUENCE_LEN:].astype(float)

# --- MAIN TRADING LOOP ---
async def run_trading_mode(bot_name):
    global cooldown_until
    predictor = SafeMLPredictor(model_path=MODEL_PATH, seq_len=SEQUENCE_LEN)
    logger.info("Starting trading loop...")

    while True:
        try:
            check_status(bot_name)
            await sync_filled_orders(bot_name)

            for symbol in SYMBOLS:
                if time.time() < cooldown_until.get(symbol, 0.0):
                    continue

                # 1. Check real position (using original symbol with slash)
                has_position = False
                qty_held = 0.0
                try:
                    position = trading_client.get_position(symbol)   # 'BTC/USD' format works
                    if float(position.qty) > 0:
                        has_position = True
                        qty_held = float(position.qty)
                except Exception:
                    has_position = False

                # 2. Get signal
                df = await get_clean_ohlcv_dataframe(symbol)
                if df is None: continue
                signal = predictor.predict(df)
                current_price = df['close'].iloc[-1]

                # 3. Sell logic
                if has_position and signal < 0.49:
                    logger.info(f"🔻 SELL signal for {symbol} (signal={signal:.3f})")
                    if execute_trade(bot_name, symbol, OrderSide.SELL, qty_held):
                        cooldown_until[symbol] = time.time() + 3600   # 1 hour cooldown
                    continue

                # 4. Buy logic
                if not has_position and signal > 0.51:
                    qty = ORDER_AMOUNT / current_price
                    logger.info(f"🎯 BUY signal for {symbol} at {current_price:.2f}")
                    if execute_trade(bot_name, symbol, OrderSide.BUY, qty):
                        cooldown_until[symbol] = time.time() + 600   # 10 min cooldown

                await asyncio.sleep(2)
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            await asyncio.sleep(30)

if __name__ == "__main__":
    asyncio.run(run_trading_mode(BOT_NAME))
