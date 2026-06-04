#!/usr/bin/env python3
# PRODUCTION READY — ALPACA DEEP LEARNING ENGINE WITH INTEGRATED ASYNC FORWARD LEARNING

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
SCALER_PATH = os.path.join(os.path.dirname(MODEL_PATH) if os.path.dirname(MODEL_PATH) else "", 'feature_scaler.pkl')

# Fetch keys securely from Coolify UI environment settings
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
PAPER = os.getenv("APCA_API_PAPER", "true").lower() == "true"

# Validate Credentials Early
if not API_KEY or not API_SECRET:
    logger.error("❌ Critical Error: Alpaca API credentials missing from environment variables!")

# Initialize Clients
data_client = CryptoHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)
trading_client = TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=PAPER)


# --- 1. PyTorch Dataset Pipeline ---
class FinancialTimeSeriesDataset(Dataset):
    """
    Transforms pre-scaled 2D feature arrays into historical sequence tensors.
    """
    def __init__(self, scaled_features, df_features, seq_len=32):
        self.seq_len = seq_len
        # FIXED: Accept pre-scaled data to prevent future-data leakage
        self.X_data = scaled_features[:-1] 
        
        close_prices = df_features['close'].values
        self.y_data = (close_prices[1:] > close_prices[:-1]).astype(np.float32)
        
    def __len__(self):
        return len(self.X_data) - self.seq_len + 1

    def __getitem__(self, idx):
        X_seq = self.X_data[idx : idx + self.seq_len]
        y_target = self.y_data[idx + self.seq_len - 1]
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)


# --- 2. Core Training Logic ---
def fetch_historical_training_data(days=120):
    """Fetches high-volume historic data for model optimization routines."""
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

def train_model(epochs=15, batch_size=64, lr=3e-4, is_refit=False):
    """Handles dataset generation, feature scaling tracking, and model training loops."""
    logger.info(f"Starting model optimization routine (Is Re-fit: {is_refit})...")
    
    try:
        raw_df = fetch_historical_training_data(days=120) 
        processed_df = add_features(raw_df)
        
        # FIXED: Fit data scaler cleanly outside dataset loop to isolate history and prevent leakage
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(processed_df[FEATURE_COLS].values)
        
        dataset = FinancialTimeSeriesDataset(scaled_features, processed_df, seq_len=SEQ_LEN)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        # OPTIMIZED: Hardened architecture footprint to slash computational delays on cloud nodes
        model = GrokGQA_Transformer(
            input_dim=len(FEATURE_COLS),
            seq_len=SEQ_LEN,
            embed_dim=128,
            num_layers=4,        # Cut from 8 to 4 for lightning-fast execution
            num_q_heads=8,        # Optimized head matrix split sizes
            num_kv_heads=2,       # Preserves perfect 4:1 GQA framework ratio
            dropout=0.1
        )
        
        # If hot-reloading/refitting, warm-start using existing weights to accelerate convergence
        if is_refit and os.path.exists(MODEL_PATH):
            try:
                model.load_state_dict(torch.load(MODEL_PATH))
                logger.info("Loaded previous weights to accelerate convergence fine-tuning.")
                lr = 1e-4  # Drop learning rate down for fine-tuning stability
            except Exception as e:
                logger.warning(f"Could not warm-start weights, training from scratch: {e}")

        criterion = nn.BCELoss()
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
            
            logger.info(f"Epoch {epoch+1:02d}/{epochs} | Avg Loss: {total_loss / len(dataloader):.4f}")
        
        # Atomic Write Optimization: Save to a temp file first, then swap them out instantly
        if os.path.dirname(MODEL_PATH):
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            
        TMP_MODEL_PATH = MODEL_PATH + ".tmp"
        TMP_SCALER_PATH = SCALER_PATH + ".tmp"
        
        torch.save(model.state_dict(), TMP_MODEL_PATH)
        joblib.dump(scaler, TMP_SCALER_PATH)
        
        os.replace(TMP_MODEL_PATH, MODEL_PATH)
        os.replace(TMP_SCALER_PATH, SCALER_PATH)
        logger.info(f"💾 Production assets successfully written and committed to disk storage.")
        
        return model
        
    except Exception as e:
        logger.error(f"Error during training workflow: {e}")
        raise e


# --- 3. Live Execution & Order Placement Framework ---
def execute_trade_signal(prediction_prob, position_active):
    """Executes market orders based on signal boundary values."""
    BUY_THRESHOLD = 0.54
    SELL_THRESHOLD = 0.46
    norm_symbol = SYMBOL.replace("/", "")
    
    if prediction_prob >= BUY_THRESHOLD and not position_active:
        logger.info(f"🔮 Bullish Signal ({prediction_prob:.2%}). Executing BUY order.")
        try:
            order_data = MarketOrderRequest(
                symbol=norm_symbol, 
                qty=0.001, 
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC
            )
            trading_client.submit_order(order_data)
            logger.info("🛒 Buy Order Placed successfully.")
        except Exception as e:
            logger.error(f"Failed to place BUY execution: {e}")
            
    elif prediction_prob <= SELL_THRESHOLD and position_active:
        logger.info(f"🔮 Bearish Signal ({prediction_prob:.2%}). Executing SELL liquidation order.")
        try:
            order_data = MarketOrderRequest(
                symbol=norm_symbol,
                qty=0.001,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )
            trading_client.submit_order(order_data)
            logger.info("🛒 Liquidating Sell Position successful.")
        except Exception as e:
            logger.error(f"Failed to place SELL execution: {e}")
    else:
        logger.info(f"Neutral zone ({prediction_prob:.2%}). Monitoring matrix tracks.")


# --- 4. Asynchronous Live Loop Tasks ---
async def nightly_refit_task():
    """Background task that runs side-by-side with trading loop. Re-fits the network daily at 00:00 UTC."""
    while True:
        try:
            now = datetime.now(timezone.utc)
            # Calculate total seconds until next midnight UTC
            tomorrow = now + timedelta(days=1)
            next_midnight = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 0, 0, 0, tzinfo=timezone.utc)
            seconds_until_midnight = (next_midnight - now).total_seconds()
            
            logger.info(f"📅 Nightly Re-fit engine sleeping for {seconds_until_midnight/3600:.2f} hours until midnight UTC.")
            await asyncio.sleep(seconds_until_midnight)
            
            logger.info("⏰ Midnight UTC reached. Initiating background forward learning re-fit cycle...")
            # Run a 10-epoch fine-tune setup using current model weights as seed baseline
            await asyncio.to_thread(train_model, epochs=10, batch_size=64, is_refit=True)
            logger.info("🎉 Nightly forward learning optimization round finished successfully.")
            
            # Extra safety buffer delay to prevent multi-trigger glitches inside the midnight timestamp second
            await asyncio.sleep(60)
            
        except Exception as e:
            logger.error(f"Critical error inside the asynchronous scheduling thread: {e}")
            await asyncio.sleep(300) # Sleep 5 minutes before trying again if an error blows up

async def run_trading_mode():
    """Live trading loop. Runs indefinitely, hot-reloading model assets if updated on disk by background worker."""
    logger.info("Entering live inference mode with Automated Hot-Reloading enabled...")
    
    predictor = MLPredictor(model_path=MODEL_PATH, seq_len=SEQ_LEN)
    last_known_model_time = os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0
    
    while True:
        try:
            # HOT-RELOAD CHECK: Did the background trainer update the weights on disk?
            if os.path.exists(MODEL_PATH):
                current_model_time = os.path.getmtime(MODEL_PATH)
                if current_model_time > last_known_model_time:
                    logger.info("🔄 Detectable update found for model weights on disk. Hot-reloading model assets...")
                    try:
                        predictor = MLPredictor(model_path=MODEL_PATH, seq_len=SEQ_LEN)
                        last_known_model_time = current_model_time
                        logger.info("🎉 Hot-reload successful! Bot is now executing transactions on fresh intelligence.")
                    except Exception as reload_err:
                        logger.error(f"Failed to hot-reload model, continuing execution on previous weights: {reload_err}")

            # 1. Fetch live technical inference sequences (72-hour window lookback)
            lookback_start = datetime.now(timezone.utc) - timedelta(days=3)
            request_params = CryptoBarsRequest(
                symbol_or_symbols=[SYMBOL],
                timeframe=TIMEFRAME,
                start=lookback_start
            )
            live_bars = data_client.get_crypto_bars(request_params)
            df_live = live_bars.df.xs(SYMBOL)
            
            # 2. Compute prediction via tracking wrapper
            probability = predictor.predict(df_live)
            
            # 3. Check tracking stances directly on Alpaca
            try:
                positions = trading_client.get_all_positions()
                position_active = any(p.symbol == SYMBOL.replace("/", "") for p in positions)
            except Exception as pos_err:
                logger.error(f"Could not reach account positions framework: {pos_err}")
                position_active = False

            logger.info(f"Bot monitoring market... | Spot: {SYMBOL} | Probability Score: {probability:.4f}")
            execute_trade_signal(probability, position_active)
            
        except Exception as e:
            logger.error(f"Error in main trading pipeline loop: {e}")
            
        # FIXED: Precision clock synchronization to stop timeframe drift on server nodes
        now = datetime.now()
        minutes_to_sleep = 15 - (now.minute % 15)
        seconds_to_sleep = (minutes_to_sleep * 60) - now.second
        if seconds_to_sleep <= 0:
            seconds_to_sleep = 900
            
        logger.info(f"Sleeping for {seconds_to_sleep // 60}m {seconds_to_sleep % 60}s to align with next 15-min candle interval close.")
        await asyncio.sleep(seconds_to_sleep)


# --- 5. Async Thread Orchestration Entrypoint ---
async def main():
    # Initial Boot Check
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        logger.info(f"Existing model and scaler files located at {MODEL_PATH}. Skipping initialization training.")
    else:
        logger.info("Missing essential model components. Starting baseline training lifecycle...")
        # Increased batch size to 64 and restricted epochs to 15 to ensure swift container lifecycles
        await asyncio.to_thread(train_model, epochs=15, batch_size=64, is_refit=False)
        logger.info("🎉 Initial baseline training cycle complete.")

    # Execute both loops simultaneously inside the async runtime ecosystem
    await asyncio.gather(
        run_trading_mode(),
        nightly_refit_task()
    )

if __name__ == "__main__":
    asyncio.run(main())
