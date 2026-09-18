"""
adaptive_monitor.py — Adaptive/self-learning behavior monitoring.

Monitors:
- Feature drift
- Prediction drift
- Win-rate drift
- Expected-value drift
- Calibration drift
- Execution-cost drift

Defines thresholds:
Continue → Reduce Risk → Pause → Retrain → Roll Back

Usage:
    python adaptive_monitor.py                  # full status
    python adaptive_monitor.py --days 7         # analyze period
    python adaptive_monitor.py --action check   # check thresholds
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from config import logger, EXPERIENCE_LOG_PATH
from experience_capture import load_experiences
from feature_engineering import FEATURE_COLS


# ── Configuration ────────────────────────────────────────────────────────────

# What is allowed to change (hard limits)
ADAPTIVE_LIMITS = {
    "max_parameter_change_per_day": 0.05,  # 5% max change per day
    "max_regime_shift_per_week": 0.2,      # 20% max regime shift per week
    "min_data_for_adaptation": 50,         # Minimum trades before adapting
    "max_consecutive_adaptations": 3,       # Max adaptations before human review
    "cooldown_hours_between_adaptations": 24,
}

# Drift thresholds → Actions
DRIFT_THRESHOLDS = {
    "continue": {
        "max_drift_score": 0.3,
        "min_win_rate": 0.45,
        "min_sharpe": 0.5,
        "max_feature_drift": 0.5,
    },
    "reduce_risk": {
        "max_drift_score": 0.5,
        "min_win_rate": 0.40,
        "min_sharpe": 0.3,
        "max_feature_drift": 0.8,
    },
    "pause": {
        "max_drift_score": 0.7,
        "min_win_rate": 0.35,
        "min_sharpe": 0.0,
        "max_feature_drift": 1.0,
    },
    "retrain": {
        "max_drift_score": 0.85,
        "min_win_rate": 0.30,
        "min_sharpe": -0.5,
        "max_feature_drift": 1.5,
    },
    "rollback": {
        "max_drift_score": 1.0,
        "min_win_rate": 0.25,
        "min_sharpe": -1.0,
        "max_feature_drift": 2.0,
    },
}


# ── Drift Detection ──────────────────────────────────────────────────────────

def compute_feature_drift(events: list[dict], baseline_events: list[dict] = None) -> dict:
    """
    Detect feature distribution drift using PSI (Population Stability Index).
    """
    if not events:
        return {"error": "No events"}
    
    drift_scores = {}
    
    for col in FEATURE_COLS:
        # Current distribution
        current_values = []
        for ev in events:
            if ev.get("type") == "entry":
                features = ev.get("features", {})
                if col in features:
                    current_values.append(features[col])
        
        if len(current_values) < 10:
            drift_scores[col] = {"drift": 0, "status": "INSUFFICIENT_DATA"}
            continue
        
        current_arr = np.array(current_values)
        
        # Baseline distribution (if provided)
        if baseline_events:
            baseline_values = []
            for ev in baseline_events:
                if ev.get("type") == "entry":
                    features = ev.get("features", {})
                    if col in features:
                        baseline_values.append(features[col])
            
            if len(baseline_values) >= 10:
                baseline_arr = np.array(baseline_values)
                
                # Compute PSI
                psi = compute_psi(baseline_arr, current_arr)
                
                # Determine status
                if psi < 0.1:
                    status = "STABLE"
                elif psi < 0.25:
                    status = "MODERATE_DRIFT"
                else:
                    status = "SEVERE_DRIFT"
                
                drift_scores[col] = {
                    "psi": round(float(psi), 4),
                    "baseline_mean": round(float(baseline_arr.mean()), 4),
                    "current_mean": round(float(current_arr.mean()), 4),
                    "mean_shift": round(float(abs(current_arr.mean() - baseline_arr.mean())), 4),
                    "status": status,
                }
            else:
                drift_scores[col] = {"psi": 0, "status": "INSUFFICIENT_BASELINE"}
        else:
            # No baseline - use statistical properties
            mean_drift = abs(current_arr.mean())
            std_ratio = current_arr.std() / (abs(current_arr.mean()) + 1e-10)
            
            if mean_drift < 0.5 and std_ratio < 2:
                status = "STABLE"
            elif mean_drift < 1.0:
                status = "MODERATE_DRIFT"
            else:
                status = "SEVERE_DRIFT"
            
            drift_scores[col] = {
                "mean_drift": round(float(mean_drift), 4),
                "std_ratio": round(float(std_ratio), 4),
                "status": status,
            }
    
    return drift_scores


def compute_psi(baseline: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Population Stability Index (PSI).
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.25: Moderate change
    PSI >= 0.25: Significant change
    """
    # Create bins from combined data
    combined = np.concatenate([baseline, current])
    bins = np.linspace(combined.min(), combined.max(), n_bins + 1)
    
    # Compute distributions
    baseline_hist, _ = np.histogram(baseline, bins=bins)
    current_hist, _ = np.histogram(current, bins=bins)
    
    # Normalize to proportions
    baseline_pct = (baseline_hist + 1) / (len(baseline) + n_bins)  # Add 1 to avoid zeros
    current_pct = (current_hist + 1) / (len(current) + n_bins)
    
    # Compute PSI
    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    
    return psi


def compute_prediction_drift(events: list[dict]) -> dict:
    """
    Detect drift in model predictions (signal strength).
    """
    signals = []
    for ev in events:
        if ev.get("type") == "entry":
            signal = ev.get("signal", 0.5)
            signals.append(signal)
    
    if len(signals) < 20:
        return {"error": "Insufficient data"}
    
    arr = np.array(signals)
    
    # Split into halves
    mid = len(arr) // 2
    first_half = arr[:mid]
    second_half = arr[mid:]
    
    # Compare
    mean_shift = abs(second_half.mean() - first_half.mean())
    std_change = abs(second_half.std() - first_half.std())
    
    # KS test for distribution change
    from scipy import stats
    ks_stat, ks_pvalue = stats.ks_2samp(first_half, second_half)
    
    return {
        "mean_signal": round(float(arr.mean()), 4),
        "std_signal": round(float(arr.std()), 4),
        "first_half_mean": round(float(first_half.mean()), 4),
        "second_half_mean": round(float(second_half.mean()), 4),
        "mean_shift": round(float(mean_shift), 4),
        "std_change": round(float(std_change), 4),
        "ks_statistic": round(float(ks_stat), 4),
        "ks_pvalue": round(float(ks_pvalue), 4),
        "is_drifting": ks_pvalue < 0.05,
        "status": "DRIFTING" if ks_pvalue < 0.05 else "STABLE",
    }


def compute_calibration_drift(events: list[dict]) -> dict:
    """
    Detect drift in prediction calibration (predicted vs actual).
    """
    entries = [e for e in events if e.get("type") == "entry"]
    exits = [e for e in events if e.get("type") == "exit"]
    
    if not entries or not exits:
        return {"error": "Insufficient data"}
    
    # Match entries to exits
    matched = []
    for entry in entries:
        symbol = entry.get("symbol", "")
        entry_ts = entry.get("ts", "")
        signal = entry.get("signal", 0.5)
        
        for exit_ev in exits:
            if exit_ev.get("symbol") == symbol and exit_ev.get("ts", "") > entry_ts:
                pnl = exit_ev.get("pnl_pct", 0)
                outcome = 1 if pnl > 0 else 0
                matched.append({"signal": signal, "outcome": outcome})
                break
    
    if len(matched) < 20:
        return {"error": "Insufficient matched trades"}
    
    # Compute calibration by decile
    matched.sort(key=lambda x: x["signal"])
    n = len(matched)
    bin_size = max(1, n // 10)
    
    calibration_error = 0
    for i in range(0, n, bin_size):
        bin_data = matched[i:i + bin_size]
        if not bin_data:
            continue
        
        mean_signal = np.mean([d["signal"] for d in bin_data])
        actual_win_rate = np.mean([d["outcome"] for d in bin_data])
        
        calibration_error += abs(mean_signal - actual_win_rate)
    
    avg_calibration_error = calibration_error / (n // bin_size + 1)
    
    return {
        "n_trades": len(matched),
        "avg_calibration_error": round(float(avg_calibration_error), 4),
        "status": "WELL_CALIBRATED" if avg_calibration_error < 0.1 else "POORLY_CALIBRATED",
    }


def compute_execution_cost_drift(events: list[dict]) -> dict:
    """
    Detect drift in execution costs (slippage, fees).
    """
    entries = [e for e in events if e.get("type") == "entry"]
    
    slippages = []
    for ev in entries:
        price = ev.get("price", 0)
        fill_price = ev.get("fill_price", 0)
        
        if price > 0 and fill_price > 0:
            slippage = (fill_price - price) / price
            slippages.append(slippage)
    
    if not slippages:
        return {"error": "No fill price data"}
    
    arr = np.array(slippages)
    
    return {
        "mean_slippage": round(float(arr.mean()), 6),
        "std_slippage": round(float(arr.std()), 6),
        "max_slippage": round(float(np.max(np.abs(arr))), 6),
        "status": "NORMAL" if abs(arr.mean()) < 0.005 else "ELEVATED",
    }


# ── State Management ─────────────────────────────────────────────────────────

def load_adaptive_state() -> dict:
    """Load adaptive monitoring state."""
    state_path = "adaptive_state.json"
    if os.path.exists(state_path):
        try:
            with open(state_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load adaptive state: {e}")
    
    return {
        "current_action": "continue",
        "drift_history": [],
        "adaptation_count": 0,
        "last_adaptation_ts": None,
        "consecutive_adaptations": 0,
        "baseline_snapshot_ts": None,
        "rollback_available": False,
    }


def save_adaptive_state(state: dict):
    """Save adaptive monitoring state."""
    state_path = "adaptive_state.json"
    try:
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save adaptive state: {e}")


# ── Action Determination ─────────────────────────────────────────────────────

def determine_action(
    drift_score: float,
    win_rate: float,
    sharpe: float,
    feature_drift: float,
    state: dict,
) -> str:
    """
    Determine action based on drift metrics.
    Priority: rollback > retrain > pause > reduce_risk > continue
    """
    # Check if we can adapt
    if state.get("consecutive_adaptations", 0) >= ADAPTIVE_LIMITS["max_consecutive_adaptations"]:
        return "pause"  # Too many adaptations, need human review
    
    last_adaptation = state.get("last_adaptation_ts")
    if last_adaptation:
        try:
            last_dt = datetime.fromisoformat(last_adaptation.replace("Z", "+00:00"))
            hours_since = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
            if hours_since < ADAPTIVE_LIMITS["cooldown_hours_between_adaptations"]:
                return state.get("current_action", "continue")  # Cooldown active
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse last_adaptation_ts: {e}")
    
    # Determine action based on thresholds
    if (drift_score >= DRIFT_THRESHOLDS["rollback"]["max_drift_score"] or
        win_rate <= DRIFT_THRESHOLDS["rollback"]["min_win_rate"] or
        sharpe <= DRIFT_THRESHOLDS["rollback"]["min_sharpe"] or
        feature_drift >= DRIFT_THRESHOLDS["rollback"]["max_feature_drift"]):
        return "rollback"
    
    if (drift_score >= DRIFT_THRESHOLDS["retrain"]["max_drift_score"] or
        win_rate <= DRIFT_THRESHOLDS["retrain"]["min_win_rate"] or
        sharpe <= DRIFT_THRESHOLDS["retrain"]["min_sharpe"]):
        return "retrain"
    
    if (drift_score >= DRIFT_THRESHOLDS["pause"]["max_drift_score"] or
        win_rate <= DRIFT_THRESHOLDS["pause"]["min_win_rate"]):
        return "pause"
    
    if (drift_score >= DRIFT_THRESHOLDS["reduce_risk"]["max_drift_score"] or
        win_rate <= DRIFT_THRESHOLDS["reduce_risk"]["min_win_rate"] or
        sharpe <= DRIFT_THRESHOLDS["reduce_risk"]["min_sharpe"]):
        return "reduce_risk"
    
    return "continue"


# ── Main Analysis ────────────────────────────────────────────────────────────

def analyze_adaptive_status(days: int = 7) -> dict:
    """
    Full adaptive behavior analysis.
    """
    events = load_experiences()
    if not events:
        return {"error": "No experiences found"}
    
    # Filter to recent period
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    recent = []
    for ev in events:
        try:
            ts_str = ev.get("ts", "")
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts >= cutoff:
                recent.append(ev)
        except (ValueError, TypeError):
            continue
    
    # Load state
    state = load_adaptive_state()
    
    # Compute drift metrics
    feature_drift = compute_feature_drift(recent)
    prediction_drift = compute_prediction_drift(recent)
    calibration_drift = compute_calibration_drift(recent)
    execution_cost_drift = compute_execution_cost_drift(recent)
    
    # Compute overall drift score
    entries = [e for e in recent if e.get("type") == "entry"]
    exits = [e for e in recent if e.get("type") == "exit"]
    
    win_rate = 0
    sharpe = 0
    drift_score = 0
    
    if entries and exits:
        # Match trades
        pnls = []
        for entry in entries:
            symbol = entry.get("symbol", "")
            entry_ts = entry.get("ts", "")
            
            for exit_ev in exits:
                if exit_ev.get("symbol") == symbol and exit_ev.get("ts", "") > entry_ts:
                    pnls.append(exit_ev.get("pnl_pct", 0))
                    break
        
        if pnls:
            pnls = np.array(pnls)
            win_rate = np.mean(pnls > 0)
            sharpe = np.mean(pnls) / (np.std(pnls) + 1e-10) if len(pnls) > 1 else 0
            
            # Feature drift score
            feature_drift_score = 0
            for col, stats in feature_drift.items():
                if isinstance(stats, dict):
                    if stats.get("status") == "SEVERE_DRIFT":
                        feature_drift_score += 0.3
                    elif stats.get("status") == "MODERATE_DRIFT":
                        feature_drift_score += 0.1
            
            drift_score = min(1.0, feature_drift_score + 
                            (0.3 if win_rate < 0.45 else 0) +
                            (0.2 if sharpe < 0.5 else 0) +
                            (0.2 if prediction_drift.get("is_drifting") else 0))
    
    # Determine action
    action = determine_action(drift_score, win_rate, sharpe, 
                             sum(1 for v in feature_drift.values() 
                                 if isinstance(v, dict) and v.get("status") == "SEVERE_DRIFT"),
                             state)
    
    return {
        "period_days": days,
        "drift_score": round(float(drift_score), 4),
        "win_rate": round(float(win_rate), 4),
        "sharpe": round(float(sharpe), 4),
        "action": action,
        "feature_drift": feature_drift,
        "prediction_drift": prediction_drift,
        "calibration_drift": calibration_drift,
        "execution_cost_drift": execution_cost_drift,
        "state": state,
        "thresholds": DRIFT_THRESHOLDS,
        "limits": ADAPTIVE_LIMITS,
    }


def print_report(results: dict):
    """Print adaptive monitoring report."""
    print("\n" + "=" * 80)
    print("ADAPTIVE BEHAVIOR MONITORING")
    print("=" * 80)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    print(f"Period: Last {results.get('period_days', 0)} days")
    print(f"Drift Score: {results.get('drift_score', 0):.4f}")
    print(f"Win Rate: {results.get('win_rate', 0):.2%}")
    print(f"Sharpe: {results.get('sharpe', 0):.4f}")
    
    # Action
    action = results.get("action", "continue")
    action_colors = {
        "continue": "🟢",
        "reduce_risk": "🟡",
        "pause": "🟠",
        "retrain": "🔴",
        "rollback": "⛔",
    }
    print(f"\nACTION: {action_colors.get(action, '?')} {action.upper()}")
    
    # Feature Drift
    feature_drift = results.get("feature_drift", {})
    if feature_drift:
        print("\n" + "-" * 80)
        print("FEATURE DRIFT")
        print("-" * 80)
        
        drifting = [(col, stats) for col, stats in feature_drift.items() 
                   if isinstance(stats, dict) and stats.get("status") in ["MODERATE_DRIFT", "SEVERE_DRIFT"]]
        
        if drifting:
            for col, stats in drifting:
                print(f"  {col}: {stats.get('status', 'N/A')} (PSI: {stats.get('psi', 'N/A')})")
        else:
            print("  No significant feature drift detected")
    
    # Prediction Drift
    pred_drift = results.get("prediction_drift", {})
    if "error" not in pred_drift:
        print("\n" + "-" * 80)
        print("PREDICTION DRIFT")
        print("-" * 80)
        print(f"  Status: {pred_drift.get('status', 'N/A')}")
        print(f"  KS p-value: {pred_drift.get('ks_pvalue', 'N/A')}")
    
    # Calibration
    cal_drift = results.get("calibration_drift", {})
    if "error" not in cal_drift:
        print("\n" + "-" * 80)
        print("CALIBRATION DRIFT")
        print("-" * 80)
        print(f"  Status: {cal_drift.get('status', 'N/A')}")
        print(f"  Avg Calibration Error: {cal_drift.get('avg_calibration_error', 'N/A')}")
    
    # Action Thresholds
    print("\n" + "-" * 80)
    print("ACTION THRESHOLDS")
    print("-" * 80)
    thresholds = results.get("thresholds", {})
    for action_name, params in thresholds.items():
        print(f"\n  {action_name.upper()}:")
        for key, value in params.items():
            print(f"    {key}: {value}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Adaptive behavior monitoring")
    parser.add_argument("--days", type=int, default=7, help="Analysis window (days)")
    args = parser.parse_args()
    
    results = analyze_adaptive_status(days=args.days)
    print_report(results)


if __name__ == "__main__":
    main()
