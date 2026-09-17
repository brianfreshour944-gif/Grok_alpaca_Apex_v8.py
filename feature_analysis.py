"""
feature_analysis.py — Feature quality analysis and correlation monitoring.

Analyzes feature importance, correlations, stationarity, and redundancy.
Run periodically to monitor feature degradation.

Usage:
    python feature_analysis.py                    # analyze from experiences
    python feature_analysis.py --csv-dir <dir>    # analyze from historical data
    python feature_analysis.py --importance        # show feature importance
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import logger, EXPERIENCE_LOG_PATH
from experience_capture import load_experiences
from feature_engineering import add_features, FEATURE_COLS


# ── Correlation Analysis ─────────────────────────────────────────────────────

def compute_feature_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix for features."""
    if df.empty or len(df) < 20:
        return pd.DataFrame()
    
    feature_df = df[FEATURE_COLS].dropna()
    if len(feature_df) < 20:
        return pd.DataFrame()
    
    return feature_df.corr()


def find_redundant_features(corr_matrix: pd.DataFrame, threshold: float = 0.7) -> list[tuple[str, str, float]]:
    """Find pairs of features with correlation > threshold."""
    redundant = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col_i = corr_matrix.columns[i]
            col_j = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > threshold:
                redundant.append((col_i, col_j, corr_val))
    return sorted(redundant, key=lambda x: abs(x[2]), reverse=True)


# ── Stationarity Analysis ────────────────────────────────────────────────────

def compute_stationarity_scores(df: pd.DataFrame, window: int = 100) -> dict:
    """
    Compute stationarity scores for each feature.
    Score = 1 - (std_of_rolling_mean / overall_std)
    Higher = more stationary.
    """
    scores = {}
    for col in FEATURE_COLS:
        if col not in df.columns:
            scores[col] = 0.0
            continue
        
        series = df[col].dropna()
        if len(series) < window * 2:
            scores[col] = 0.5  # insufficient data
            continue
        
        # Rolling mean and its variability
        rolling_mean = series.rolling(window).mean().dropna()
        rolling_std = series.rolling(window).std().dropna()
        
        if len(rolling_mean) < 10 or rolling_std.mean() == 0:
            scores[col] = 0.5
            continue
        
        # Stationarity = how stable is the rolling mean relative to the feature's volatility
        mean_variability = rolling_mean.std() / (series.std() + 1e-8)
        score = max(0, min(1, 1 - mean_variability))
        scores[col] = round(score, 4)
    
    return scores


# ── Feature Importance Analysis ──────────────────────────────────────────────

def analyze_feature_importance(events: list[dict]) -> dict:
    """Analyze feature importance from logged experiences."""
    entries = [e for e in events if e.get("type") == "entry" and e.get("feature_importance")]
    
    if not entries:
        return {"error": "No feature importance data found"}
    
    # Aggregate importance by feature
    importance_sums = {col: [] for col in FEATURE_COLS}
    
    for entry in entries:
        imp = entry.get("feature_importance", {})
        for col in FEATURE_COLS:
            if col in imp:
                importance_sums[col].append(imp[col])
    
    # Compute statistics
    results = {}
    for col in FEATURE_COLS:
        values = importance_sums[col]
        if values:
            results[col] = {
                "mean": round(float(np.mean(values)), 6),
                "std": round(float(np.std(values)), 6),
                "min": round(float(np.min(values)), 6),
                "max": round(float(np.max(values)), 6),
                "n_samples": len(values),
            }
        else:
            results[col] = {"mean": 0, "std": 0, "min": 0, "max": 0, "n_samples": 0}
    
    return results


# ── Main Analysis ────────────────────────────────────────────────────────────

def analyze_from_experiences(days: int = 7) -> dict:
    """Run full feature analysis from experience log."""
    events = load_experiences()
    if not events:
        return {"error": "No experiences found"}
    
    # Filter to recent period
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    recent_events = []
    for ev in events:
        try:
            ts_str = ev.get("ts", "")
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts >= cutoff:
                recent_events.append(ev)
        except (ValueError, TypeError):
            continue
    
    # Extract features from entries
    entries = [e for e in recent_events if e.get("type") == "entry" and e.get("features")]
    
    if not entries:
        return {"error": f"No entries with features in last {days} days"}
    
    # Build feature DataFrame
    feature_rows = []
    for entry in entries:
        features = entry.get("features", {})
        if features:
            feature_rows.append(features)
    
    if not feature_rows:
        return {"error": "No valid feature data"}
    
    df = pd.DataFrame(feature_rows)
    
    # Ensure all FEATURE_COLS present
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    
    # Analysis
    corr_matrix = compute_feature_correlations(df)
    redundant = find_redundant_features(corr_matrix)
    stationarity = compute_stationarity_scores(df)
    importance = analyze_feature_importance(recent_events)
    
    # Outliers detection
    outlier_stats = {}
    for col in FEATURE_COLS:
        if col in df.columns:
            series = df[col].dropna()
            if len(series) > 0:
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                outlier_count = ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()
                outlier_stats[col] = {
                    "count": int(outlier_count),
                    "pct": round(float(outlier_count / len(series) * 100), 2),
                }
    
    return {
        "period_days": days,
        "n_entries": len(entries),
        "n_features": len(FEATURE_COLS),
        "correlation_matrix": corr_matrix.to_dict() if not corr_matrix.empty else {},
        "redundant_pairs": redundant,
        "stationarity_scores": stationarity,
        "feature_importance": importance,
        "outlier_stats": outlier_stats,
    }


def print_report(results: dict):
    """Print analysis report."""
    print("\n" + "=" * 70)
    print("FEATURE QUALITY ANALYSIS")
    print("=" * 70)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    print(f"Period: Last {results['period_days']} days")
    print(f"Entries analyzed: {results['n_entries']}")
    print(f"Features: {results['n_features']}")
    
    # Redundant features
    print("\n" + "-" * 70)
    print("REDUNDANT FEATURE PAIRS (correlation > 0.7)")
    print("-" * 70)
    redundant = results.get("redundant_pairs", [])
    if redundant:
        for col_i, col_j, corr in redundant:
            print(f"  {col_i:20s} <-> {col_j:20s} : {corr:+.4f}")
    else:
        print("  No redundant features detected")
    
    # Stationarity scores
    print("\n" + "-" * 70)
    print("STATIONARITY SCORES (1.0 = perfectly stationary)")
    print("-" * 70)
    stationarity = results.get("stationarity_scores", {})
    for col in FEATURE_COLS:
        score = stationarity.get(col, 0)
        status = "✅" if score > 0.7 else "⚠️" if score > 0.4 else "❌"
        print(f"  {status} {col:20s} : {score:.4f}")
    
    # Feature importance
    print("\n" + "-" * 70)
    print("FEATURE IMPORTANCE (gradient-based)")
    print("-" * 70)
    importance = results.get("feature_importance", {})
    if importance and "error" not in importance:
        sorted_imp = sorted(importance.items(), key=lambda x: x[1].get("mean", 0), reverse=True)
        for col, stats in sorted_imp:
            mean_imp = stats.get("mean", 0)
            n = stats.get("n_samples", 0)
            print(f"  {col:20s} : {mean_imp:.6f} (n={n})")
    else:
        print("  No importance data available")
    
    # Outlier stats
    print("\n" + "-" * 70)
    print("OUTLIER STATISTICS (>1.5 IQR from quartiles)")
    print("-" * 70)
    outliers = results.get("outlier_stats", {})
    for col in FEATURE_COLS:
        stats = outliers.get(col, {})
        count = stats.get("count", 0)
        pct = stats.get("pct", 0)
        status = "✅" if pct < 5 else "⚠️" if pct < 10 else "❌"
        print(f"  {status} {col:20s} : {count} outliers ({pct:.1f}%)")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Feature quality analysis")
    parser.add_argument("--days", type=int, default=7, help="Analysis window (days)")
    parser.add_argument("--importance", action="store_true", help="Show feature importance")
    args = parser.parse_args()
    
    results = analyze_from_experiences(days=args.days)
    print_report(results)


if __name__ == "__main__":
    main()
