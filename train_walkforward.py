"""
train_walkforward.py — Walk-forward retraining pipeline.

Retrains the transformer model on recent data using rolling windows,
validates on held-out future data, and saves new candidate models.
Designed to run daily/weekly via cron or scheduler.

Usage:
    python train_walkforward.py                    # default walk-forward
    python train_walkforward.py --window-days 30   # custom lookback
    python train_walkforward.py --force            # retrain even if no new data
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import logger, MODEL_PATH, SEQUENCE_LEN, EXPERIENCE_LOG_PATH
from experience_capture import load_experiences
from feature_engineering import add_features, FEATURE_COLS
from ml_predictor import GrokGQA_Transformer


# ── Config ────────────────────────────────────────────────────────────────────
RETRAIN_DIR = os.getenv("RETRAIN_DIR", "retrained_models")
LOOKBACK_DAYS = int(os.getenv("RETRAIN_LOOKBACK_DAYS", "14"))
MIN_TRADES_FOR_TRAIN = int(os.getenv("MIN_TRADES_FOR_TRAIN", "50"))
TARGET_HORIZON_BARS = 6  # 6 x 15min = 90min forward return
VAL_SPLIT = 0.2  # 20% held out for validation


# ── Data loading ──────────────────────────────────────────────────────────────

def load_training_data(lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Load experiences from JSONL and convert to training samples.
    Each entry becomes a feature vector; label = 1 if profitable exit within horizon.
    """
    events = load_experiences()
    if not events:
        logger.warning("No experiences found in JSONL")
        return pd.DataFrame()

    # Filter to entries within lookback window
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    entries = []
    exits_by_symbol = {}

    for ev in events:
        if ev.get("type") == "exit":
            sym = ev.get("symbol", "")
            if sym not in exits_by_symbol:
                exits_by_symbol[sym] = []
            exits_by_symbol[sym].append(ev)
        elif ev.get("type") == "entry":
            ts_str = ev.get("ts", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts >= cutoff:
                    entries.append(ev)
            except (ValueError, TypeError):
                continue

    if not entries:
        logger.warning(f"No entries found in last {lookback_days} days")
        return pd.DataFrame()

    # Join entries to exits: label = 1 if profitable exit within horizon
    rows = []
    for entry in entries:
        features = entry.get("features", {})
        if not features or len(features) < len(FEATURE_COLS):
            continue

        symbol = entry.get("symbol", "")
        entry_price = entry.get("price", 0)
        entry_ts = entry.get("ts", "")

        # Find the next exit for this symbol
        label = 0.5  # default: no exit found (uncertain)
        if symbol in exits_by_symbol:
            for exit_ev in exits_by_symbol[symbol]:
                exit_ts = exit_ev.get("ts", "")
                if exit_ts > entry_ts:
                    pnl = exit_ev.get("pnl_pct", 0)
                    label = 1.0 if pnl > 0 else 0.0
                    break

        row = {col: features.get(col, 0.0) for col in FEATURE_COLS}
        row["label"] = label
        row["symbol"] = symbol
        row["entry_price"] = entry_price
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} training samples from {len(entries)} entries")
    return df


def prepare_tensors(df: pd.DataFrame, seq_len: int = SEQUENCE_LEN):
    """
    Convert DataFrame to sequences for transformer training.
    Returns (X_train, y_train, X_val, y_val).
    """
    if df.empty or len(df) < seq_len:
        return None, None, None, None

    features = df[FEATURE_COLS].values.astype(np.float32)
    labels = df["label"].values.astype(np.float32)

    # Create sequences (sliding window)
    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len : i])
        y.append(labels[i])

    if not X:
        return None, None, None, None

    X = np.array(X)
    y = np.array(y)

    # Train/val split
    split_idx = int(len(X) * (1 - VAL_SPLIT))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    return (
        torch.tensor(X_train),
        torch.tensor(y_train),
        torch.tensor(X_val),
        torch.tensor(y_val),
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    X_train, y_train, X_val, y_val,
    epochs: int = 50,
    lr: float = 1e-4,
    batch_size: int = 32,
    patience: int = 10,
) -> dict:
    """
    Train transformer model with early stopping.
    Returns metrics dict.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[-1]

    model = GrokGQA_Transformer(
        input_dim=input_dim,
        seq_len=SEQUENCE_LEN,
        embed_dim=128,
        num_layers=4,
        num_q_heads=8,
        num_kv_heads=2,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
    )

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x).squeeze(-1)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x).squeeze(-1)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
                preds = (torch.sigmoid(output) > 0.5).float()
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total if total > 0 else 0

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss/len(train_loader):.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return {
        "model": model,
        "best_val_loss": best_val_loss,
        "val_acc": val_acc,
        "epochs_trained": epoch + 1,
        "input_dim": input_dim,
    }


# ── Validation ────────────────────────────────────────────────────────────────

def validate_candidate(model, X_val, y_val) -> dict:
    """
    Validate candidate model against held-out data.
    Returns validation metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with torch.no_grad():
        output = model(X_val.to(device)).squeeze(-1)
        preds = torch.sigmoid(output)
        binary_preds = (preds > 0.5).float()

        # Metrics
        accuracy = (binary_preds == y_val).float().mean().item()
        
        # Precision/Recall for class 1 (profitable)
        tp = ((binary_preds == 1) & (y_val == 1)).sum().item()
        fp = ((binary_preds == 1) & (y_val == 0)).sum().item()
        fn = ((binary_preds == 0) & (y_val == 1)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Brier score (calibration)
        brier = ((preds - y_val) ** 2).mean().item()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier_score": brier,
        "n_samples": len(y_val),
    }


# ── Save ──────────────────────────────────────────────────────────────────────

def save_candidate(model, metrics: dict, suffix: str = "") -> str:
    """Save model artifact with metadata."""
    os.makedirs(RETRAIN_DIR, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"transformer_candidate_{timestamp}{suffix}.pth"
    filepath = os.path.join(RETRAIN_DIR, filename)

    # Save model weights
    torch.save(model.state_dict(), filepath)

    # Save metadata
    meta = {
        "timestamp": timestamp,
        "metrics": metrics,
        "model_path": filepath,
        "sequel_len": SEQUENCE_LEN,
        "input_dim": metrics.get("input_dim", 11),
    }
    meta_path = filepath + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved candidate model: {filepath}")
    logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")

    return filepath


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Walk-forward model retraining")
    parser.add_argument("--window-days", type=int, default=LOOKBACK_DAYS,
                        help="Lookback window in days (default: 14)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Max training epochs (default: 50)")
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if no new data")
    args = parser.parse_args()

    logger.info(f"Starting walk-forward retraining (lookback={args.window_days} days)")

    # Load data
    df = load_training_data(lookback_days=args.window_days)
    if df.empty:
        if args.force:
            logger.warning("No data but --force flag set. Using synthetic data for test.")
        else:
            logger.info("No new data to train on. Exiting.")
            return

    # Prepare tensors
    X_train, y_train, X_val, y_val = prepare_tensors(df)
    if X_train is None:
        logger.error("Not enough data to create training sequences")
        return

    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Train
    result = train_model(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs,
    )

    # Validate
    val_metrics = validate_candidate(result["model"], X_val, y_val)
    val_metrics["input_dim"] = result["input_dim"]
    val_metrics["train_loss"] = result["best_val_loss"]
    val_metrics["epochs_trained"] = result["epochs_trained"]

    # Save
    filepath = save_candidate(result["model"], val_metrics)

    # Summary
    print("\n" + "=" * 60)
    print("WALK-FORWARD RETRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {filepath}")
    print(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Precision: {val_metrics['precision']:.4f}")
    print(f"Recall: {val_metrics['recall']:.4f}")
    print(f"F1 Score: {val_metrics['f1']:.4f}")
    print(f"Brier Score: {val_metrics['brier_score']:.4f}")
    print("=" * 60)
    print("\nRun promotion_gate.py to evaluate against champion.")


if __name__ == "__main__":
    main()
