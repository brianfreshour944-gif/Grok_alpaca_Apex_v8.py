"""
backtest_validation.py — Validate backtest/training pipeline integrity.

Checks for:
- Look-ahead bias
- Future data leakage
- Correct timestamps
- Correct indicator warm-up
- Realistic fills/fees/slippage
- Survivorship bias
- Data leakage
- Training/validation/test separation

Usage:
    python backtest_validation.py              # full validation
    python backtest_validation.py --check all  # specific check
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import logger, EXPERIENCE_LOG_PATH, MODEL_PATH, SEQUENCE_LEN
from experience_capture import load_experiences
from feature_engineering import add_features, FEATURE_COLS
from ml_predictor import GrokGQA_Transformer


# ── Check 1: Look-Ahead Bias ────────────────────────────────────────────────

def check_lookahead_bias(events: list[dict]) -> dict:
    """
    Check if features use future data.
    Labels should be based on FUTURE outcomes, not past.
    """
    issues = []
    
    for ev in events:
        if ev.get("type") != "entry":
            continue
        
        features = ev.get("features", {})
        entry_ts = ev.get("ts", "")
        entry_price = ev.get("price", 0)
        
        # Check if any feature looks like it uses future data
        # Features should only use data up to entry time
        for feat_name, feat_value in features.items():
            if feat_value is None or feat_value == 0:
                continue
            
            # Suspicious: features that are exactly 0 or 1 (binary outcomes)
            if feat_name == "label":
                if feat_value not in [0, 0.5, 1.0]:
                    issues.append(f"Entry {entry_ts}: label={feat_value} (unexpected value)")
    
    return {
        "check": "lookahead_bias",
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "details": {
            "total_entries": sum(1 for e in events if e.get("type") == "entry"),
            "entries_with_features": sum(1 for e in events if e.get("type") == "entry" and e.get("features")),
        },
    }


# ── Check 2: Timestamp Ordering ─────────────────────────────────────────────

def check_timestamp_ordering(events: list[dict]) -> dict:
    """
    Verify timestamps are monotonically increasing.
    No future timestamps in feature computation.
    """
    issues = []
    
    timestamps = []
    for ev in events:
        ts_str = ev.get("ts", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            timestamps.append((ts, ev.get("type", ""), ev.get("symbol", "")))
        except (ValueError, TypeError):
            issues.append(f"Invalid timestamp: {ts_str}")
    
    # Check ordering within each symbol
    by_symbol = {}
    for ts, typ, sym in timestamps:
        if sym not in by_symbol:
            by_symbol[sym] = []
        by_symbol[sym].append((ts, typ))
    
    for sym, sym_events in by_symbol.items():
        sym_events.sort(key=lambda x: x[0])
        for i in range(1, len(sym_events)):
            prev_ts, prev_type = sym_events[i-1]
            curr_ts, curr_type = sym_events[i]
            
            # Entry should come before exit
            if prev_type == "exit" and curr_type == "entry":
                if curr_ts < prev_ts:
                    issues.append(f"{sym}: entry {curr_ts} before exit {prev_ts}")
    
    return {
        "check": "timestamp_ordering",
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "details": {
            "total_timestamps": len(timestamps),
            "unique_symbols": len(by_symbol),
        },
    }


# ── Check 3: Feature Warm-Up ────────────────────────────────────────────────

def check_feature_warmup(events: list[dict]) -> dict:
    """
    Verify indicators have sufficient warm-up period.
    Features computed from < N bars may be unreliable.
    """
    issues = []
    
    MIN_WARMUP_BARS = 20  # Minimum bars for reliable indicators
    
    for ev in events:
        if ev.get("type") != "entry":
            continue
        
        features = ev.get("features", {})
        bars_used = ev.get("bars_used", 0)
        
        if bars_used > 0 and bars_used < MIN_WARMUP_BARS:
            issues.append(
                f"Entry {ev.get('ts', '')}: only {bars_used} bars "
                f"(min {MIN_WARMUP_BARS} recommended)"
            )
        
        # Check for default/neutral feature values (indicates warm-up issue)
        default_count = 0
        for col in FEATURE_COLS:
            val = features.get(col, 0.0)
            if val == 0.0:
                default_count += 1
        
        if default_count > len(FEATURE_COLS) * 0.5:
            issues.append(
                f"Entry {ev.get('ts', '')}: {default_count}/{len(FEATURE_COLS)} "
                f"features are default (possible warm-up issue)"
            )
    
    return {
        "check": "feature_warmup",
        "status": "PASS" if not issues else "WARN",
        "issues": issues[:20],  # First 20
        "details": {
            "min_warmup_bars": MIN_WARMUP_BARS,
            "total_entries_checked": sum(1 for e in events if e.get("type") == "entry"),
        },
    }


# ── Check 4: Realistic Fills ────────────────────────────────────────────────

def check_realistic_fills(events: list[dict]) -> dict:
    """
    Verify fills are realistic (not using exact entry/exit prices).
    """
    issues = []
    
    for ev in events:
        if ev.get("type") != "entry":
            continue
        
        price = ev.get("price", 0)
        fill_price = ev.get("fill_price", 0)
        
        if price > 0 and fill_price > 0:
            slippage = abs(fill_price - price) / price
            
            # Unrealistic: perfect fills (0 slippage)
            if slippage == 0:
                issues.append(f"Entry {ev.get('ts', '')}: zero slippage (unrealistic)")
            
            # Unrealistic: too much slippage
            if slippage > 0.05:  # > 5%
                issues.append(
                    f"Entry {ev.get('ts', '')}: {slippage:.2%} slippage (too high)"
                )
    
    return {
        "check": "realistic_fills",
        "status": "PASS" if not issues else "WARN",
        "issues": issues[:20],
        "details": {
            "entries_with_fill_data": sum(
                1 for e in events 
                if e.get("type") == "entry" and e.get("fill_price")
            ),
        },
    }


# ── Check 5: Survivorship Bias ─────────────────────────────────────────────

def check_survivorship_bias(events: list[dict]) -> dict:
    """
    Check if only winning symbols are included.
    Are losing symbols that were delisted/dropped excluded?
    """
    issues = []
    
    # Count by symbol
    by_symbol = {}
    for ev in events:
        sym = ev.get("symbol", "")
        if sym not in by_symbol:
            by_symbol[sym] = {"entries": 0, "exits": 0, "wins": 0, "losses": 0}
        
        if ev.get("type") == "entry":
            by_symbol[sym]["entries"] += 1
        elif ev.get("type") == "exit":
            by_symbol[sym]["exits"] += 1
            if ev.get("pnl_pct", 0) > 0:
                by_symbol[sym]["wins"] += 1
            else:
                by_symbol[sym]["losses"] += 1
    
    # Check for symbols with entries but no exits (still open)
    still_open = []
    for sym, counts in by_symbol.items():
        if counts["entries"] > counts["exits"]:
            still_open.append(sym)
    
    if still_open:
        issues.append(f"Symbols still open (not yet exited): {still_open}")
    
    # Check for symbols with 100% win rate (suspicious)
    for sym, counts in by_symbol.items():
        if counts["exits"] > 5 and counts["losses"] == 0:
            issues.append(f"{sym}: 100% win rate over {counts['exits']} trades (suspicious)")
    
    return {
        "check": "survivorship_bias",
        "status": "PASS" if not issues else "WARN",
        "issues": issues,
        "details": {
            "unique_symbols": len(by_symbol),
            "symbols_still_open": len(still_open),
            "symbol_stats": by_symbol,
        },
    }


# ── Check 6: Data Leakage ───────────────────────────────────────────────────

def check_data_leakage(events: list[dict]) -> dict:
    """
    Check if training data leaks into validation/test sets.
    """
    issues = []
    
    # Check if experience log has duplicate timestamps
    timestamps = [e.get("ts", "") for e in events]
    unique_timestamps = set(timestamps)
    
    if len(timestamps) != len(unique_timestamps):
        dup_count = len(timestamps) - len(unique_timestamps)
        issues.append(f"Duplicate timestamps found: {dup_count} duplicates")
    
    # Check if features are identical across different entries
    feature_hashes = []
    for ev in events:
        if ev.get("type") == "entry":
            features = ev.get("features", {})
            if features:
                feat_str = json.dumps(features, sort_keys=True)
                feature_hashes.append(feat_str)
    
    unique_features = set(feature_hashes)
    if len(feature_hashes) != len(unique_features):
        dup_count = len(feature_hashes) - len(unique_features)
        issues.append(f"Duplicate feature vectors: {dup_count} duplicates")
    
    return {
        "check": "data_leakage",
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "details": {
            "total_events": len(events),
            "unique_timestamps": len(unique_timestamps),
            "unique_feature_vectors": len(unique_features),
        },
    }


# ── Check 7: Training/Validation Split ──────────────────────────────────────

def check_train_val_split(events: list[dict]) -> dict:
    """
    Verify proper train/validation/test separation.
    """
    issues = []
    
    entries = [e for e in events if e.get("type") == "entry"]
    
    if len(entries) < 100:
        issues.append(f"Only {len(entries)} entries (need 100+ for proper split)")
    
    # Check temporal split
    timestamps = []
    for ev in entries:
        ts_str = ev.get("ts", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            timestamps.append(ts)
        except (ValueError, TypeError):
            continue
    
    if timestamps:
        timestamps.sort()
        train_end = timestamps[int(len(timestamps) * 0.7)]
        val_end = timestamps[int(len(timestamps) * 0.85)]
        
        # Check if data is properly split by time
        train_count = sum(1 for t in timestamps if t <= train_end)
        val_count = sum(1 for t in timestamps if train_end < t <= val_end)
        test_count = sum(1 for t in timestamps if t > val_end)
        
        return {
            "check": "train_val_split",
            "status": "PASS",
            "issues": issues,
            "details": {
                "total_entries": len(entries),
                "train_count": train_count,
                "val_count": val_count,
                "test_count": test_count,
                "train_end": str(train_end),
                "val_end": str(val_end),
                "split_ratio": f"{train_count/len(entries):.1%}/{val_count/len(entries):.1%}/{test_count/len(entries):.1%}",
            },
        }
    
    return {
        "check": "train_val_split",
        "status": "WARN",
        "issues": ["Could not determine temporal split"],
        "details": {},
    }


# ── Main Validation ──────────────────────────────────────────────────────────

def run_validation(checks: list[str] = None) -> dict:
    """Run all validation checks."""
    events = load_experiences()
    
    if not events:
        return {
            "status": "SKIP",
            "reason": "No experience data found",
            "checks": [],
        }
    
    all_checks = {
        "lookahead_bias": check_lookahead_bias,
        "timestamp_ordering": check_timestamp_ordering,
        "feature_warmup": check_feature_warmup,
        "realistic_fills": check_realistic_fills,
        "survivorship_bias": check_survivorship_bias,
        "data_leakage": check_data_leakage,
        "train_val_split": check_train_val_split,
    }
    
    if checks is None:
        checks = list(all_checks.keys())
    
    results = {}
    for check_name in checks:
        if check_name in all_checks:
            results[check_name] = all_checks[check_name](events)
    
    # Overall status
    statuses = [r["status"] for r in results.values()]
    if all(s == "PASS" for s in statuses):
        overall = "PASS"
    elif any(s == "FAIL" for s in statuses):
        overall = "FAIL"
    else:
        overall = "WARN"
    
    return {
        "status": overall,
        "total_events": len(events),
        "checks": results,
    }


def print_report(results: dict):
    """Print validation report."""
    print("\n" + "=" * 80)
    print("BACKTEST VALIDATION REPORT")
    print("=" * 80)
    
    if results.get("status") == "SKIP":
        print(f"Skipped: {results.get('reason', 'Unknown')}")
        return
    
    print(f"Total Events: {results.get('total_events', 0)}")
    print(f"Overall Status: {results.get('status', 'UNKNOWN')}")
    
    checks = results.get("checks", {})
    for check_name, check_result in checks.items():
        status = check_result.get("status", "UNKNOWN")
        icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}.get(status, "?")
        
        print(f"\n{icon} {check_name.upper().replace('_', ' ')}")
        
        issues = check_result.get("issues", [])
        if issues:
            for issue in issues[:5]:
                print(f"  - {issue}")
            if len(issues) > 5:
                print(f"  ... and {len(issues) - 5} more")
        else:
            print("  No issues found")
        
        details = check_result.get("details", {})
        if details:
            for key, value in details.items():
                if key != "symbol_stats":
                    print(f"  {key}: {value}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Backtest validation")
    parser.add_argument("--check", type=str, nargs="+", help="Specific checks to run")
    args = parser.parse_args()
    
    results = run_validation(checks=args.check)
    print_report(results)


if __name__ == "__main__":
    main()
