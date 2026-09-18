"""
promotion_gate.py — Champion/Challenger promotion and demotion logic.

Evaluates candidate models against the current champion using walk-forward
validation. Promotes candidates that outperform; demotes champions that
degrade below thresholds.

Usage:
    python promotion_gate.py                    # evaluate latest candidate
    python promotion_gate.py --candidate <path> # evaluate specific model
    python promotion_gate.py --force-promote    # skip validation, force promote
    python promotion_gate.py --demote           # force demote current champion
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import logger, MODEL_PATH, SEQUENCE_LEN, RETRAIN_DIR
from ml_predictor import SafeMLPredictor, GrokGQA_Transformer
from train_walkforward import load_training_data, prepare_tensors


# ── Config ────────────────────────────────────────────────────────────────────
PROMOTION_STATE_PATH = os.getenv("PROMOTION_STATE_PATH", "promotion_state.json")
BACKUP_DIR = os.getenv("MODEL_BACKUP_DIR", "model_backups")

# Minimum metrics to promote (must beat champion by this margin)
MIN_IMPROVEMENT = float(os.getenv("PROMOTION_MIN_IMPROVEMENT", "0.02"))
MIN_ACCURACY = float(os.getenv("PROMOTION_MIN_ACCURACY", "0.50"))
MIN_F1 = float(os.getenv("PROMOTION_MIN_F1", "0.45"))

# Demotion thresholds (champion is demoted if below these)
DEMOTE_ACCURACY = float(os.getenv("PROMOTION_DEMOTE_ACCURACY", "0.42"))
DEMOTE_F1 = float(os.getenv("PROMOTION_DEMOTE_F1", "0.38"))


# ── State management ──────────────────────────────────────────────────────────

def load_state() -> dict:
    """Load promotion state."""
    if os.path.exists(PROMOTION_STATE_PATH):
        try:
            with open(PROMOTION_STATE_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load promotion state from {PROMOTION_STATE_PATH}: {e}")
    return {
        "champion_path": MODEL_PATH,
        "champion_metrics": None,
        "last_promotion": None,
        "last_demotion": None,
        "promotion_history": [],
    }


def save_state(state: dict):
    """Save promotion state."""
    try:
        with open(PROMOTION_STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save promotion state: {e}")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_for_eval(path: str) -> nn.Module:
    """Load a model for evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if path.endswith((".joblib", ".pkl")):
        # Sklearn model - load via SafeMLPredictor
        predictor = SafeMLPredictor(model_path=path, seq_len=SEQUENCE_LEN)
        return predictor

    # Torch model
    model = GrokGQA_Transformer(
        input_dim=11,  # FEATURE_COLS count
        seq_len=SEQUENCE_LEN,
        embed_dim=128,
        num_layers=4,
        num_q_heads=8,
        num_kv_heads=2,
        dropout=0.0,
    ).to(device)

    try:
        model.load_state_dict(
            torch.load(path, map_location=device),
            strict=True,
        )
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {path}: {e}")
        raise


def evaluate_model(model, X_val, y_val) -> dict:
    """Evaluate a model on validation data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(model, SafeMLPredictor):
        # Sklearn model
        return _evaluate_sklearn(model, X_val, y_val)

    # Torch model
    with torch.no_grad():
        output = model(X_val.to(device)).squeeze(-1)
        preds = torch.sigmoid(output)
        binary_preds = (preds > 0.5).float()

        # Metrics
        accuracy = (binary_preds == y_val).float().mean().item()

        tp = ((binary_preds == 1) & (y_val == 1)).sum().item()
        fp = ((binary_preds == 1) & (y_val == 0)).sum().item()
        fn = ((binary_preds == 0) & (y_val == 1)).sum().item()
        tn = ((binary_preds == 0) & (y_val == 0)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        brier = ((preds - y_val) ** 2).mean().item()

        # Confusion matrix
        confusion = {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier_score": brier,
        "confusion": confusion,
        "n_samples": len(y_val),
    }


def _evaluate_sklearn(predictor, X_val, y_val) -> dict:
    """Evaluate sklearn model."""
    # Convert tensors to numpy
    X_np = X_val.numpy() if torch.is_tensor(X_val) else X_val
    y_np = y_val.numpy() if torch.is_tensor(y_val) else y_val

    # Get predictions
    all_preds = []
    for i in range(len(X_np)):
        row = X_np[i].reshape(1, -1) if len(X_np[i].shape) == 1 else X_np[i:i+1]
        proba = predictor.clf.predict_proba(row)[0]
        classes = list(getattr(predictor.clf, "classes_", [0, 1]))
        pred = float(proba[classes.index(1)]) if 1 in classes else float(proba[-1])
        all_preds.append(pred)

    preds = np.array(all_preds)
    binary_preds = (preds > 0.5).astype(float)

    accuracy = (binary_preds == y_np).mean()
    tp = ((binary_preds == 1) & (y_np == 1)).sum()
    fp = ((binary_preds == 1) & (y_np == 0)).sum()
    fn = ((binary_preds == 0) & (y_np == 1)).sum()
    tn = ((binary_preds == 0) & (y_np == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    brier = ((preds - y_np) ** 2).mean()

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "brier_score": float(brier),
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "n_samples": len(y_np),
    }


# ── Promotion logic ───────────────────────────────────────────────────────────

def backup_champion(champion_path: str) -> str:
    """Backup current champion before promotion."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_name = f"champion_backup_{timestamp}.pth"
    backup_path = os.path.join(BACKUP_DIR, backup_name)
    shutil.copy2(champion_path, backup_path)
    logger.info(f"Backed up champion to {backup_path}")
    return backup_path


def promote_candidate(candidate_path: str, candidate_metrics: dict) -> bool:
    """Promote candidate to champion."""
    state = load_state()
    champion_path = state.get("champion_path", MODEL_PATH)

    # Backup current champion
    try:
        backup_champion(champion_path)
    except Exception as e:
        logger.warning(f"Failed to backup champion: {e}")

    # Copy candidate to champion location
    try:
        shutil.copy2(candidate_path, champion_path)
        logger.info(f"Promoted {candidate_path} to champion {champion_path}")
    except Exception as e:
        logger.error(f"Failed to promote candidate: {e}")
        return False

    # Update state
    state["champion_path"] = champion_path
    state["champion_metrics"] = candidate_metrics
    state["last_promotion"] = datetime.now(timezone.utc).isoformat()
    state["promotion_history"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candidate": candidate_path,
        "metrics": candidate_metrics,
    })
    save_state(state)

    return True


def demote_champion(reason: str = "performance degradation") -> bool:
    """Demote current champion (force retraining)."""
    state = load_state()
    champion_path = state.get("champion_path", MODEL_PATH)

    # Backup before demotion
    try:
        backup_champion(champion_path)
    except Exception as e:
        logger.warning(f"Failed to backup champion: {e}")

    # Mark as demoted
    state["last_demotion"] = datetime.now(timezone.utc).isoformat()
    state["demotion_reason"] = reason
    state["champion_metrics"] = None
    save_state(state)

    logger.warning(f"Champion demoted: {reason}")
    return True


# ── Main evaluation ───────────────────────────────────────────────────────────

def find_latest_candidate() -> str | None:
    """Find the most recent candidate model."""
    if not os.path.exists(RETRAIN_DIR):
        return None

    candidates = []
    for f in os.listdir(RETRAIN_DIR):
        if f.startswith("transformer_candidate_") and f.endswith(".pth"):
            path = os.path.join(RETRAIN_DIR, f)
            candidates.append(path)

    if not candidates:
        return None

    # Sort by modification time
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def evaluate_candidates(candidate_path: str = None) -> dict:
    """Evaluate candidate against champion."""
    state = load_state()

    # Find candidate
    if candidate_path is None:
        candidate_path = find_latest_candidate()
        if candidate_path is None:
            logger.info("No candidate model found to evaluate")
            return {"error": "No candidate found"}

    logger.info(f"Evaluating candidate: {candidate_path}")

    # Load training data for validation
    df = load_training_data(lookback_days=14)
    if df.empty:
        logger.error("No training data available for evaluation")
        return {"error": "No training data"}

    X_train, y_train, X_val, y_val = prepare_tensors(df)
    if X_train is None:
        logger.error("Not enough data for evaluation")
        return {"error": "Insufficient data"}

    # Evaluate champion
    champion_path = state.get("champion_path", MODEL_PATH)
    try:
        champion = load_model_for_eval(champion_path)
        champion_metrics = evaluate_model(champion, X_val, y_val)
        logger.info(f"Champion metrics: accuracy={champion_metrics['accuracy']:.4f}, f1={champion_metrics['f1']:.4f}")
    except Exception as e:
        logger.error(f"Failed to evaluate champion: {e}")
        champion_metrics = {"accuracy": 0.5, "f1": 0.5, "error": str(e)}

    # Evaluate candidate
    try:
        candidate = load_model_for_eval(candidate_path)
        candidate_metrics = evaluate_model(candidate, X_val, y_val)
        logger.info(f"Candidate metrics: accuracy={candidate_metrics['accuracy']:.4f}, f1={candidate_metrics['f1']:.4f}")
    except Exception as e:
        logger.error(f"Failed to evaluate candidate: {e}")
        return {"error": f"Candidate evaluation failed: {e}"}

    # Compare
    accuracy_diff = candidate_metrics["accuracy"] - champion_metrics["accuracy"]
    f1_diff = candidate_metrics["f1"] - champion_metrics["f1"]

    result = {
        "champion_path": champion_path,
        "candidate_path": candidate_path,
        "champion_metrics": champion_metrics,
        "candidate_metrics": candidate_metrics,
        "accuracy_diff": accuracy_diff,
        "f1_diff": f1_diff,
        "should_promote": False,
        "reason": "",
    }

    # Promotion criteria
    if candidate_metrics["accuracy"] < MIN_ACCURACY:
        result["reason"] = f"Candidate accuracy {candidate_metrics['accuracy']:.4f} < {MIN_ACCURACY}"
    elif candidate_metrics["f1"] < MIN_F1:
        result["reason"] = f"Candidate F1 {candidate_metrics['f1']:.4f} < {MIN_F1}"
    elif accuracy_diff > MIN_IMPROVEMENT and f1_diff > -0.05:
        result["should_promote"] = True
        result["reason"] = f"Candidate improves accuracy by {accuracy_diff:.4f} and F1 by {f1_diff:.4f}"
    else:
        result["reason"] = f"Candidate improvement insufficient (acc={accuracy_diff:+.4f}, f1={f1_diff:+.4f})"

    # Demotion check
    if champion_metrics.get("accuracy", 0) < DEMOTE_ACCURACY:
        result["should_demote"] = True
        result["demote_reason"] = f"Champion accuracy {champion_metrics['accuracy']:.4f} < {DEMOTE_ACCURACY}"
    elif champion_metrics.get("f1", 0) < DEMOTE_F1:
        result["should_demote"] = True
        result["demote_reason"] = f"Champion F1 {champion_metrics['f1']:.4f} < {DEMOTE_F1}"
    else:
        result["should_demote"] = False

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Model promotion gate")
    parser.add_argument("--candidate", type=str, help="Path to candidate model")
    parser.add_argument("--force-promote", action="store_true", help="Force promote candidate")
    parser.add_argument("--demote", action="store_true", help="Force demote champion")
    args = parser.parse_args()

    if args.demote:
        reason = input("Enter demotion reason: ") or "Manual demotion"
        demote_champion(reason)
        print("Champion demoted.")
        return

    # Evaluate
    result = evaluate_candidates(args.candidate)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    # Print report
    print("\n" + "=" * 60)
    print("PROMOTION GATE EVALUATION")
    print("=" * 60)
    print(f"Champion: {result['champion_path']}")
    print(f"Candidate: {result['candidate_path']}")
    print()
    print("Champion Metrics:")
    print(f"  Accuracy: {result['champion_metrics']['accuracy']:.4f}")
    print(f"  F1 Score: {result['champion_metrics']['f1']:.4f}")
    print(f"  Brier:    {result['champion_metrics']['brier_score']:.4f}")
    print()
    print("Candidate Metrics:")
    print(f"  Accuracy: {result['candidate_metrics']['accuracy']:.4f}")
    print(f"  F1 Score: {result['candidate_metrics']['f1']:.4f}")
    print(f"  Brier:    {result['candidate_metrics']['brier_score']:.4f}")
    print()
    print(f"Accuracy Diff: {result['accuracy_diff']:+.4f}")
    print(f"F1 Diff:       {result['f1_diff']:+.4f}")
    print()
    print(f"Should Promote: {'YES' if result['should_promote'] else 'NO'}")
    print(f"Reason: {result['reason']}")
    print()

    if result.get("should_demote"):
        print(f"⚠️  Champion should be DEMOTED: {result['demote_reason']}")
    print("=" * 60)

    # Auto-promote if criteria met
    if result["should_promote"] or args.force_promote:
        if args.force_promote:
            print("\nForce promote requested. Promoting...")
        else:
            response = input("\nPromote candidate? [y/N]: ")
            if response.lower() != "y":
                print("Promotion cancelled.")
                return

        success = promote_candidate(
            result["candidate_path"],
            result["candidate_metrics"],
        )
        if success:
            print("✅ Candidate promoted to champion!")
        else:
            print("❌ Promotion failed.")
    elif result.get("should_demote"):
        response = input("\nDemote champion? [y/N]: ")
        if response.lower() == "y":
            demote_champion(result.get("demote_reason", "performance"))
            print("Champion demoted. Retraining recommended.")


if __name__ == "__main__":
    main()
