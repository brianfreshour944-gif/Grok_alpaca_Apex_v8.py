import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
import os
import joblib
from datetime import datetime, timedelta, timezone
from sklearn.preprocessing import StandardScaler

# Alpaca Official SDK Imports
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

# Import centralized logic files
from feature_engineering import add_features, FEATURE_COLS
from ml_predictor import GrokGQA_Transformer, MLPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class FinancialTimeSeriesDataset(Dataset):
    def __init__(self, df, seq_len=32, target_horizon=8, scaler=None):
        self.seq_len = seq_len
        self.target_horizon = target_horizon
        self.scaler = scaler
        self.data, self.labels = self._prepare_data(df)

    def _prepare_data(self, df):
        df = df.copy()
        
        # 1. Calculate target variables on raw dataframe first
        df['future_return'] = df['close'].pct_change(self.target_horizon).shift(-self.target_horizon)
        target_labels = (df['future_return'] > 0).astype(float).fillna(0).values

        # 2. Run your technical analysis feature extractor
        df_features = add_features(df)
        
        # 3. Pull the exact features specified for model inputs
        raw_data = df_features[FEATURE_COLS].values

        # 4. Apply or fit standard scalers
        if self.scaler is None:
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(raw_data)
        else:
            scaled_data = self.scaler.transform(raw_data)

        return scaled_data, target_labels

    def __len__(self):
        return len(self.data) - self.seq_len - self.target_horizon

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.labels[idx + self.seq_len + self.target_horizon - 1] 
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def train_model(
    epochs=100,
    batch_size=16,
    lr=3e-4,
    seq_len=32,
    embed_dim=128,
    num_layers=8,  # Enforced 8 layers baseline
    num_q_heads=16,
    num_kv_heads=4,
    dropout=0.1,
    patience=10,
    target_horizon=8
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Fetching historical crypto training bars via Alpaca API...")
    client = CryptoHistoricalDataClient()
    
    # Fetch past 60 days of hourly data to form a deep training canvas
    start_time = datetime.now(timezone.utc) - timedelta(days=60)
    
    request_params = CryptoBarsRequest(
        symbol_or_symbols=["BTC/USD"],
        timeframe=TimeFrame.Hour,
        start=start_time
    )
    
    bars = client.get_crypto_bars(request_params)
    df = bars.df.loc["BTC/USD"].copy()
    
    df.index = pd.to_datetime(df.index)
    df = df[['open', 'high', 'low', 'close', 'volume']]

    dataset = FinancialTimeSeriesDataset(df, seq_len=seq_len, target_horizon=target_horizon)
    joblib.dump(dataset.scaler, "feature_scaler.pkl")
    logger.info("📀 Saved feature_scaler.pkl to project root folder.")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GrokGQA_Transformer(
        input_dim=len(FEATURE_COLS),
        seq_len=seq_len,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        dropout=dropout
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCELoss() 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience//2, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0

    logger.info("Starting Transformer network training optimization run...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X).squeeze(-1)
            
            if output.shape != y.shape:
                output = output[:y.shape[0]]

            loss = criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                output = model(X).squeeze(-1)
                if output.shape != y.shape:
                    output = output[:y.shape[0]]
                val_loss += criterion(output, y).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)

        logger.info(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train:.5f} | Val Loss: {avg_val:.5f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "grok_gqa_v9_best.pth")
            patience_counter = 0
            logger.info("📀 Updated optimal model weights: grok_gqa_v9_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("⏹️ Training capped via early stopping convergence.")
                break

    torch.save(model.state_dict(), "grok_gqa_v9_final.pth")
    return model


if __name__ == "__main__":
    # FIXED: Explicitly passing num_layers=8 to prevent default parameters from overriding structural layout
    model = train_model(epochs=40, batch_size=16, lr=3e-4, seq_len=32, num_layers=8)
    logger.info("🎉 System training cycle complete.")
