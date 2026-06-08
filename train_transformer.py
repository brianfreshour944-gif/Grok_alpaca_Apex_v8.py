
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

# Per‑symbol cooldowns (seconds) after a buy or failed sell
cooldown_until = {symbol: 0.0 for symbol in SYMBOLS}

# ---------- HELPER FUNCTION FOR SYMBOL FORMAT ----------
def get_position_symbol(symbol: str) -> str:
    """Convert 'BTC/USD' -> 'BTCUSD' for Alpaca's get_position endpoint."""
    return symbol.replace("/", "")

# ---------- SAFE FEATURE ENGINEERING (unchanged) ----------
def safe_add_features(df: pd.DataFrame) -> pd.DataFrame:
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

# ---------- DATABASE HELPERS (same as before) ----------
def log_error_to_db(bot_name, error_msg):
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        return
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
                cur.execute("""
                    INSERT INTO bot_status (bot_name, last_update, status) 
                    VALUES (%s, NOW(), 'RUNNING') 
                    ON CONFLICT (bot_name) DO UPDATE SET last_update = NOW(), status = EXCLUDED.status;
                """, (bot_name,))
                cur.execute("SELECT status FROM bot_status WHERE bot_name = %s", (bot_name,))
                result = cur.fetchone()
                conn.commit()
                if result and result[0] == 'STOP':
                    sys.exit(0)
    except Exception as e:
        logger.error(f"Heartbeat failed: {e}")

def sync_trade_to_db(bot_name, side, price, qty, symbol, order_id, fee=0.0):
    try:
        price, qty = float(price or 0.0), float(qty or 0.0)
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

def register_order_in_db(bot_name, order_id, symbol, side, price):
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        return
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO bot_orders (order_id, bot_name, symbol, side, price, status)
                    VALUES (%s, %s, %s, %s, %s, 'OPEN')
                    ON CONFLICT (order_id) DO NOTHING
                """, (str(order_id), bot_name, symbol, side, float(price)))
                conn.commit()
    except Exception as e:
        logger.error(f"Failed to register order in DB: {e}")

def execute_trade(bot_name, symbol, side, qty):
    try:
        order = trading_client.submit_order(
            order_data=MarketOrderRequest(
                symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.GTC
            )
        )
        # Verify the order exists on Alpaca before storing
        try:
            trading_client.get_order_by_id(order.id)
            register_order_in_db(bot_name, order.id, symbol, side.value, 0.0)
            logger.info(f"✅ Placed {side.value} order for {symbol} | Qty: {qty:.6f} | Order ID: {order.id}")
            return order
        except Exception as verify_err:
            logger.error(f"Order {order.id} not confirmed by Alpaca: {verify_err}")
            return None
    except Exception as e:
        log_error_to_db(bot_name, f"Trade execution failed for {symbol}: {e}")
        return None

async def sync_filled_orders(bot_name):
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        return
    with psycopg2.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT order_id, symbol FROM bot_orders WHERE bot_name = %s AND status = 'OPEN'", (bot_name,))
            for oid, symbol in cur.fetchall():
                try:
                    alpaca_order = trading_client.get_order_by_id(oid)
                    if alpaca_order.status == 'filled':
                        cur.execute("UPDATE bot_orders SET status = 'CLOSED' WHERE order_id = %s", (oid,))
                        conn.commit()
                        sync_trade_to_db(
                            bot_name, alpaca_order.side.value,
                            alpaca_order.filled_avg_price, alpaca_order.filled_qty,
                            symbol, oid, fee=0.0
                        )
                        # No need to update any in‑memory flag – we will query real positions later
                except Exception as e:
                    error_msg = str(e)
                    if "40410000" in error_msg or "order not found" in error_msg.lower():
                        logger.warning(f"Order {oid} not found – marking as ERROR")
                        cur.execute("UPDATE bot_orders SET status = 'ERROR' WHERE order_id = %s", (oid,))
                        conn.commit()
                    else:
                        logger.error(f"Error syncing order {oid}: {e}")

async def get_clean_ohlcv_dataframe(symbol):
    end = datetime.now()
    start = end - timedelta(hours=6)
    request = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        limit=500
    )
    bars = data_client.get_crypto_bars(request).data.get(symbol, [])
    if len(bars) < SEQUENCE_LEN:
        logger.warning(f"Insufficient minute bars for {symbol}: {len(bars)} < {SEQUENCE_LEN}")
        return None

    data = []
    for b in bars:
        data.append({
            'timestamp': b.timestamp,
            'open': float(b.open) if b.open is not None else 0.0,
            'high': float(b.high) if b.high is not None else 0.0,
            'low': float(b.low) if b.low is not None else 0.0,
            'close': float(b.close) if b.close is not None else 0.0,
            'volume': float(b.volume) if b.volume is not None else 0.0,
        })
    df = pd.DataFrame(data)
    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    ohlc_5 = df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    for col in ['open', 'high', 'low', 'close', 'volume']:
        ohlc_5[col] = pd.to_numeric(ohlc_5[col], errors='coerce').fillna(0.0)
        ohlc_5[col] = ohlc_5[col].replace([np.inf, -np.inf], 0.0)
    ohlc_5 = ohlc_5.map(lambda x: 0.0 if x is None else x)

    if len(ohlc_5) < SEQUENCE_LEN:
        logger.warning(f"Not enough clean 5‑min bars for {symbol}: {len(ohlc_5)}")
        return None

    ohlc_5 = ohlc_5.iloc[-SEQUENCE_LEN:]
    return ohlc_5.astype(float)

# --- Safe ML Predictor (unchanged) ---
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

            df_features = safe_add_features(df)

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

async def run_trading_mode(bot_name):
    global cooldown_until
    predictor = SafeMLPredictor(model_path=MODEL_PATH, seq_len=SEQUENCE_LEN)
    logger.info("SafeMLPredictor loaded. Starting trading loop...")

    while True:
        try:
            check_status(bot_name)
            await sync_filled_orders(bot_name)

            for symbol in SYMBOLS:
                # 1. Respect per‑symbol cooldown
                if time.time() < cooldown_until.get(symbol, 0.0):
                    logger.debug(f"{symbol} cooldown active, skipping")
                    continue

                # 2. Check if we already have an open position (real, not memory)
                try:
                    pos_symbol = get_position_symbol(symbol)
                    position = trading_client.get_position(pos_symbol)
                    has_position = True
                    qty_held = float(position.qty)
                except Exception:
                    has_position = False
                    qty_held = 0.0

                # 3. Get fresh data and signal
                df = await get_clean_ohlcv_dataframe(symbol)
                if df is None:
                    continue
                signal = predictor.predict(df)
                current_price = df['close'].iloc[-1]

                # 4. SELL logic (if we have a position and signal < 0.49)
                if has_position and signal < 0.49:
                    logger.info(f"🔻 SELL signal for {symbol} (signal={signal:.3f}), attempting to sell {qty_held} units")
                    order = execute_trade(bot_name, symbol, OrderSide.SELL, qty_held)
                    if order:
                        # Set a long cooldown after a successful sell (1 hour)
                        cooldown_until[symbol] = time.time() + 3600
                        logger.info(f"✅ Sell order placed for {symbol}")
                    else:
                        # If sell order fails, do NOT reset cooldown – just log and continue
                        logger.error(f"❌ Sell order failed for {symbol}")
                    # After attempting to sell, skip further actions for this symbol this cycle
                    continue

                # 5. BUY logic (only if we have NO position AND signal > 0.51)
                if not has_position and signal > 0.51:
                    qty = ORDER_AMOUNT / current_price
                    logger.info(f"🎯 BUY signal for {symbol} at {current_price:.2f} (signal={signal:.3f})")
                    order = execute_trade(bot_name, symbol, OrderSide.BUY, qty)
                    if order:
                        # Set cooldown after buying to prevent immediate re‑buy on next cycle
                        cooldown_until[symbol] = time.time() + 600   # 10 minutes
                        logger.info(f"✅ Buy order placed for {symbol}")
                    else:
                        logger.error(f"❌ Buy order failed for {symbol}")

                # Small delay between symbols to avoid rate limits
                await asyncio.sleep(2)

            # Wait 60 seconds before the next full cycle
            await asyncio.sleep(60)

        except Exception as e:
            error_msg = f"Main loop error: {e}"
            logger.error(error_msg)
            log_error_to_db(bot_name, error_msg)
            await asyncio.sleep(30)

if __name__ == "__main__":
    try:
        asyncio.run(run_trading_mode(BOT_NAME))
    except Exception as e:
        log_error_to_db(BOT_NAME, f"FATAL CRASH: {e}")
        sys.exit(1)
