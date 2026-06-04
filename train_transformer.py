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

# Alpaca Official SDK Imports
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Import centralized logic files
from feature_engineering import add_features, FEATURE_COLS
from ml_predictor import GrokGQA_Transformer, MLPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# --- Configurations ---
SYMBOL = "BTC/USD"
TIMEFRAME = TimeFrame.Hour
SEQ_LEN = 32
MODEL_PATH = "/app/data/grok_gqa_v9_best.pth" if os.path.exists("/app/data") else "grok_gqa_v9_best.pth"
SCALER_PATH = MODEL_PATH.replace(".pth", "_scaler.pkl")

# Fetch keys securely from Coolify UI environment settings
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
PAPER = os.getenv("APCA_API_PAPER", "true").lower() == "true"

# Initialize Alpaca clients safely
if not API_KEY or not API_SECRET:
    logger.error("❌ Critical Error: Alpaca API credentials missing from environment variables!")

data_client = CryptoHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)
trading_client = TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=PAPER)

# --- 1. PyTorch Dataset Pipeline ---
class FinancialTimeSeriesDataset(Dataset):
    """
    Transforms historic 2D feature data into sequence tensors for sequence modeling.
    Target matches next-bar direction (1 for higher close, 0 for lower/equal close).
    """
    def __init__(self, df_features, seq_len=32):
        self.seq_len = seq_len
        
        # Scale the features
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(df_features[FEATURE_COLS].values)
        
        # Set up data arrays
        self.X_data = scaled_data[:-1] 
        # Target: Did the next close price go up compared to current close price?
        close_prices = df_features['close'].values
        self.y_data = (close_prices[1:] > close_prices[:-1]).astype(np.float32)
        
    def __len__(self):
        return len(self.X_data) - self.seq_len + 1

    def __getitem__(self, idx):
        X_seq = self.X_data[idx : idx + self.seq_len]
        y_target = self.y_data[idx + self.seq_len - 1]
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)

# --- 2. PyTorch Training Engine ---
def fetch_historical_training_data(days=365):
    """Fetches high-volume historic data to feed initial model optimizations."""
    logger.info(f"Fetching {days} days of historical training bars for {SYMBOL}...")
    start_time = datetime.now(timezone.utc) - timedelta(days=days)
    
    request_params = CryptoBarsRequest(
        symbol_or_symbols=[SYMBOL],
        timeframe=TIMEFRAME,
        start=start_time
    )
    bars = data_client.get_crypto_bars(request_params)
    df = bars.df.xs(SYMBOL)
    return df

def train_model(epochs=40, batch_size=16, lr=3e-4, seq_len=32, num_layers=8):
    """Handles data pulling, processing, and optimization training for the GQA Transformer."""
    logger.info("Starting model training routine...")
    
    try:
        raw_df = fetch_historical_training_data(days=180) # 6 months of lookback hourly data
        processed_df = add_features(raw_df)
        
        dataset = FinancialTimeSeriesDataset(processed_df, seq_len=seq_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        # Instantiate Model Architecture
        model = GrokGQA_Transformer(
            input_dim=len(FEATURE_COLS),
            seq_len=seq_len,
            embed_dim=128,
            num_layers=num_layers,
            num_q_heads=16,
            num_kv_heads=4,
            dropout=0.1
        )
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                predictions = model(X_batch).squeeze(-1)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss / len(dataloader):.4f}")
        
        # Save training assets for quick recovery container reboots
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True) if os.path.dirname(MODEL_PATH) else None
        torch.save(model.state_dict(), MODEL_PATH)
        joblib.dump(dataset.scaler, SCALER_PATH)
        logger.info(f"💾 Model artifact saved to '{MODEL_PATH}' and Scaler to '{SCALER_PATH}'")
        
        model.eval()
        return model
        
    except Exception as e:
        logger.error(f"Error during training workflow: {e}")
        raise e

# --- 3. Live Trading Pipeline Loop ---
def execute_trade_signal(prediction_prob, position_active):
    """Executes market orders based on standard threshold boundary values."""
    BUY_THRESHOLD = 0.55
    SELL_THRESHOLD = 0.45
    
    # Check buying framework
    if prediction_prob >= BUY_THRESHOLD and not position_active:
        logger.info(f"🔮 Bullish Signal ({prediction_prob:.2%}). Executing BUY order.")
        try:
            order_data = MarketOrderRequest(
                symbol=SYMBOL.replace("/", ""), # Convert BTC/USD to BTCUSD for order execution
                qty=0.001, # Keep order minimum tiny for baseline execution confirmation
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC
            )
            trading_client.submit_order(order_data)
            logger.info("🛒 Buy Order Placed successfully.")
        except Exception as e:
            logger.error(f"Failed to place BUY execution: {e}")
            
    # Check selling framework
    elif prediction_prob <= SELL_THRESHOLD and position_active:
        logger.info(f"🔮 Bearish Signal ({prediction_prob:.2%}). Executing SELL liquidation order.")
        try:
            order_data = MarketOrderRequest(
                symbol=SYMBOL.replace("/", ""),
                qty=0.001,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )
            trading_client.submit_order(order_data)
            logger.info("🛒 Liquidating Sell Position successful.")
        except Exception as e:
            logger.error(f"Failed to place SELL execution: {e}")
    else:
        logger.info(f"Neutral zone ({prediction_prob:.2%}). No trade adjustments made.")

def run_trading_mode(model):
    """
    Live trading/inference loop. Runs indefinitely, protecting from rate limits.
    """
    logger.info("Entering live trading/inference mode...")
    
    # Load scaling tracking object 
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    else:
        logger.warning("No existing scaler artifact detected. Instantiating a default live standardizer.")
        scaler = StandardScaler()
    
    while True:
        try:
            # 1. Fetch live technical inference sequences
            # Get data for past 72 hours to ensure enough rows are available for feature generation lags
            lookback_start = datetime.now(timezone.utc) - timedelta(days=3)
            request_params = CryptoBarsRequest(
                symbol_or_symbols=[SYMBOL],
                timeframe=TIMEFRAME,
                start=lookback_start
            )
            live_bars = data_client.get_crypto_bars(request_params)
            df_live = live_bars.df.xs(SYMBOL)
            
            # 2. Run feature engineering 
            df_features = add_features(df_live)
            
            if len(df_features) < SEQ_LEN:
                logger.warning(f"Insufficient real-time sequence historical bars. Have {len(df_features)}, need {SEQ_LEN}. Waiting next iteration.")
                time.sleep(60)
                continue
                
            # Pull the most current available complete sequence trailing metrics
            recent_sequence = df_features[FEATURE_COLS].tail(SEQ_LEN).values
            
            # Apply identical scaling transformations as training parameters
            try:
                scaled_sequence = scaler.transform(recent_sequence)
            except Exception:
                # Fallback safeguard option if scaler bounds mismatch columns
                scaled_sequence = scaler.fit_transform(recent_sequence)
                
            # Convert matrix elements into expected model tensor shapes [Batch Size=1, Seq Len, Feature Dimensions]
            tensor_input = torch.tensor(scaled_sequence, dtype=torch.float32).unsqueeze(0)
            
            # 3. Predict with model
            with torch.no_grad():
                logits = model(tensor_input).squeeze(-1)
                probability = torch.sigmoid(logits).item() # Map output directly to bounded probability range [0, 1]
            
            # Check positional stance tracking directly via Alpaca's current ledger
            try:
                positions = trading_client.get_all_positions()
                position_active = any(p.symbol == SYMBOL.replace("/", "") for p in positions)
            except Exception as pos_err:
                logger.error(f"Could not reach account positions framework: {pos_err}")
                position_active = False

            # 4. Execute trades if signals exist
            logger.info(f"Bot monitoring market... [Trading Loop Active] | Spot: {SYMBOL} | Prob Score: {probability:.4f}")
            execute_trade_signal(probability, position_active)
            
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            
        # 5. Sleep 15 minutes to align with market bar evaluation pacing and prevent API hammer
        time.sleep(900)

if __name__ == "__main__":
    # Ensure this points to a persistent volume path in Coolify
    # If using Coolify Storage, use: "/app/data/grok_gqa_v9_best.pth"
    
    # 1. Check if model already exists to break the restart loop
    if os.path.exists(MODEL_PATH):
        logger.info(f"Existing model found at {MODEL_PATH}. Loading weights.")
        
        # Re-initialize the model structure
        model = GrokGQA_Transformer(
            input_dim=len(FEATURE_COLS),
            seq_len=SEQ_LEN,
            embed_dim=128,
            num_layers=8,
            num_q_heads=16,
            num_kv_heads=4,
            dropout=0.1
        )
        
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            model.eval()
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}. Falling back to training.")
            model = train_model(epochs=40, batch_size=16, lr=3e-4, seq_len=SEQ_LEN, num_layers=8)
    else:
        logger.info("No model found. Starting initial training...")
        model = train_model(epochs=40, batch_size=16, lr=3e-4, seq_len=SEQ_LEN, num_layers=8)
        logger.info("🎉 Initial training cycle complete.")

    # 2. Transition to Trading Mode
    # This keeps the container alive without triggering re-training
    run_trading_mode(model)
