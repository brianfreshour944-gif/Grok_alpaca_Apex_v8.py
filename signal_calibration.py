"""
signal_calibration.py — Signal strength and calibration analysis.

Analyzes whether the ML signal's predicted probability matches actual outcomes.
Tests threshold sensitivity and identifies optimal signal filtering.

Usage:
    python signal_calibration.py                    # analyze last 7 days
    python signal_calibration.py --days 30          # longer window
    python signal_calibration.py --threshold-test   # test optimal thresholds
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from config import logger, EXPERIENCE_LOG_PATH, BUY_SIGNAL, SELL_SIGNAL
from experience_capture import load_experiences


# ── Signal Calibration ────────────────────────────────────────────────────────

def compute_calibration(events: list[dict], n_bins: int = 10) -> dict:
    """
    Compute calibration: predicted probability vs actual win frequency.
    Bins signals into deciles and compares mean predicted vs actual outcome.
    """
    # Extract entry-exit pairs
    entries = {}
    exits = {}
    
    for ev in events:
        if ev.get("type") == "entry":
            sym = ev.get("symbol", "")
            ts = ev.get("ts", "")
            signal = ev.get("signal", 0.5)
            entries[(sym, ts)] = {"signal": signal, "symbol": sym, "ts": ts}
        elif ev.get("type") == "exit":
            sym = ev.get("symbol", "")
            ts = ev.get("ts", "")
            exits[(sym, ts)] = {"pnl_pct": ev.get("pnl_pct", 0), "symbol": sym, "ts": ts}
    
    # Match entries to exits
    matched = []
    for (sym, entry_ts), entry in entries.items():
        for (sym2, exit_ts), exit_ev in exits.items():
            if sym == sym2 and exit_ts > entry_ts:
                matched.append({
                    "signal": entry["signal"],
                    "pnl_pct": exit_ev["pnl_pct"],
                    "outcome": 1 if exit_ev["pnl_pct"] > 0 else 0,
                    "symbol": sym,
                    "entry_ts": entry_ts,
                    "exit_ts": exit_ts,
                })
                break
    
    if not matched:
        return {"error": "No matched entry-exit pairs"}
    
    # Sort by signal strength
    matched.sort(key=lambda x: x["signal"])
    
    # Compute calibration bins
    n = len(matched)
    bin_size = max(1, n // n_bins)
    bins = []
    
    for i in range(0, n, bin_size):
        bin_data = matched[i:i + bin_size]
        if not bin_data:
            continue
        
        signals = [d["signal"] for d in bin_data]
        outcomes = [d["outcome"] for d in bin_data]
        pnls = [d["pnl_pct"] for d in bin_data]
        
        bins.append({
            "signal_range": (min(signals), max(signals)),
            "mean_signal": np.mean(signals),
            "n_trades": len(bin_data),
            "actual_win_rate": np.mean(outcomes),
            "mean_pnl": np.mean(pnls),
            "median_pnl": np.median(pnls),
            "std_pnl": np.std(pnls) if len(pnls) > 1 else 0,
        })
    
    # Overall calibration metrics
    all_signals = [d["signal"] for d in matched]
    all_outcomes = [d["outcome"] for d in matched]
    all_pnls = [d["pnl_pct"] for d in matched]
    
    # Brier score: mean((predicted - actual)^2)
    brier = np.mean([(s - o) ** 2 for s, o in zip(all_signals, all_outcomes)])
    
    # Log-loss (calibration quality)
    eps = 1e-10
    log_loss = -np.mean([
        o * np.log(max(s, eps)) + (1 - o) * np.log(max(1 - s, eps))
        for s, o in zip(all_signals, all_outcomes)
    ])
    
    return {
        "n_trades": n,
        "overall_win_rate": np.mean(all_outcomes),
        "mean_signal": np.mean(all_signals),
        "mean_pnl": np.mean(all_pnls),
        "brier_score": round(float(brier), 4),
        "log_loss": round(float(log_loss), 4),
        "calibration_bins": bins,
        "matched_trades": matched,
    }


# ── Threshold Testing ────────────────────────────────────────────────────────

def test_thresholds(events: list[dict], thresholds: list[float] = None) -> list[dict]:
    """
    Test performance at different signal thresholds.
    Returns metrics for each threshold.
    """
    if thresholds is None:
        thresholds = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.65, 0.70, 0.75, 0.80]
    
    # Extract entry-exit pairs
    entries = {}
    exits = {}
    
    for ev in events:
        if ev.get("type") == "entry":
            sym = ev.get("symbol", "")
            ts = ev.get("ts", "")
            signal = ev.get("signal", 0.5)
            entries[(sym, ts)] = {"signal": signal, "symbol": sym, "ts": ts}
        elif ev.get("type") == "exit":
            sym = ev.get("symbol", "")
            ts = ev.get("ts", "")
            exits[(sym, ts)] = {"pnl_pct": ev.get("pnl_pct", 0), "symbol": sym, "ts": ts}
    
    # Match entries to exits
    matched = []
    for (sym, entry_ts), entry in entries.items():
        for (sym2, exit_ts), exit_ev in exits.items():
            if sym == sym2 and exit_ts > entry_ts:
                matched.append({
                    "signal": entry["signal"],
                    "pnl_pct": exit_ev["pnl_pct"],
                    "outcome": 1 if exit_ev["pnl_pct"] > 0 else 0,
                })
                break
    
    if not matched:
        return [{"error": "No matched trades"}]
    
    # Test each threshold
    results = []
    baseline_win_rate = np.mean([d["outcome"] for d in matched])
    baseline_pnl = np.mean([d["pnl_pct"] for d in matched])
    
    for threshold in thresholds:
        filtered = [d for d in matched if d["signal"] >= threshold]
        
        if not filtered:
            results.append({
                "threshold": threshold,
                "n_trades": 0,
                "n_filtered": len(matched),
                "note": "No trades above threshold"
            })
            continue
        
        outcomes = [d["outcome"] for d in filtered]
        pnls = [d["pnl_pct"] for d in filtered]
        
        win_rate = np.mean(outcomes)
        mean_pnl = np.mean(pnls)
        total_pnl = np.sum(pnls)
        sharpe = np.mean(pnls) / (np.std(pnls) + 1e-10) if len(pnls) > 1 else 0
        
        # Expectancy per trade
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        
        results.append({
            "threshold": threshold,
            "n_trades": len(filtered),
            "n_filtered_out": len(matched) - len(filtered),
            "pct_of_total": round(len(filtered) / len(matched) * 100, 1),
            "win_rate": round(float(win_rate), 4),
            "win_rate_vs_baseline": round(float(win_rate - baseline_win_rate), 4),
            "mean_pnl": round(float(mean_pnl), 6),
            "total_pnl": round(float(total_pnl), 4),
            "sharpe": round(float(sharpe), 4),
            "avg_win": round(float(avg_win), 6),
            "avg_loss": round(float(avg_loss), 6),
            "expectancy": round(float(expectancy), 6),
            "profit_factor": round(float(avg_win / abs(avg_loss)), 4) if avg_loss != 0 else float("inf"),
        })
    
    return results


# ── Signal Distribution ──────────────────────────────────────────────────────

def analyze_signal_distribution(events: list[dict]) -> dict:
    """Analyze distribution of signal strengths."""
    signals = [e.get("signal", 0.5) for e in events if e.get("type") == "entry"]
    
    if not signals:
        return {"error": "No signals"}
    
    arr = np.array(signals)
    
    return {
        "count": len(arr),
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "median": round(float(np.median(arr)), 4),
        "p10": round(float(np.percentile(arr, 10)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "histogram": {
            f"{i/10:.1f}-{(i+1)/10:.1f}": int(((arr >= i/10) & (arr < (i+1)/10)).sum())
            for i in range(10)
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def analyze_from_experiences(days: int = 7) -> dict:
    """Run full signal analysis from experience log."""
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
    
    entries = [e for e in recent if e.get("type") == "entry"]
    exits = [e for e in recent if e.get("type") == "exit"]
    
    if not entries:
        return {"error": f"No entries in last {days} days"}
    
    calibration = compute_calibration(recent)
    threshold_tests = test_thresholds(recent)
    signal_dist = analyze_signal_distribution(recent)
    
    return {
        "period_days": days,
        "n_entries": len(entries),
        "n_exits": len(exits),
        "calibration": calibration,
        "threshold_tests": threshold_tests,
        "signal_distribution": signal_dist,
    }


def print_report(results: dict):
    """Print analysis report."""
    print("\n" + "=" * 70)
    print("SIGNAL CALIBRATION & THRESHOLD ANALYSIS")
    print("=" * 70)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    print(f"Period: Last {results['period_days']} days")
    print(f"Entries: {results['n_entries']} | Exits: {results['n_exits']}")
    
    # Calibration
    cal = results.get("calibration", {})
    if "error" not in cal:
        print("\n" + "-" * 70)
        print("SIGNAL CALIBRATION (Predicted vs Actual)")
        print("-" * 70)
        print(f"Overall Win Rate: {cal.get('overall_win_rate', 0):.2%}")
        print(f"Mean Signal: {cal.get('mean_signal', 0):.4f}")
        print(f"Mean PnL: {cal.get('mean_pnl', 0):.4%}")
        print(f"Brier Score: {cal.get('brier_score', 0):.4f} (lower = better)")
        print(f"Log Loss: {cal.get('log_loss', 0):.4f} (lower = better)")
        
        print("\nCalibration Bins (Predicted Signal → Actual Win Rate):")
        print(f"{'Signal Range':<15} {'N':>5} {'Pred':>8} {'Actual':>8} {'PnL':>10}")
        print("-" * 50)
        for b in cal.get("calibration_bins", []):
            sr = b.get("signal_range", (0, 0))
            print(f"{sr[0]:.2f}-{sr[1]:.2f}     {b['n_trades']:>5} "
                  f"{b['mean_signal']:>8.4f} {b['actual_win_rate']:>8.2%} "
                  f"{b['mean_pnl']:>10.4%}")
    
    # Threshold tests
    thresholds = results.get("threshold_tests", [])
    if thresholds and "error" not in thresholds[0]:
        print("\n" + "-" * 70)
        print("THRESHOLD SENSITIVITY ANALYSIS")
        print("-" * 70)
        print(f"{'Threshold':>10} {'Trades':>7} {'%':>6} {'WinRate':>8} {'ΔBaseline':>10} "
              f"{'MeanPnL':>10} {'Sharpe':>8} {'Exp/Trade':>10}")
        print("-" * 70)
        
        baseline_wr = None
        for t in thresholds:
            if t.get("n_trades", 0) == 0:
                continue
            
            if baseline_wr is None:
                baseline_wr = t.get("win_rate", 0)
            
            delta = t.get("win_rate_vs_baseline", 0)
            print(f"{t['threshold']:>10.2f} {t['n_trades']:>7} {t['pct_of_total']:>5.1f}% "
                  f"{t['win_rate']:>8.2%} {delta:>+10.2%} "
                  f"{t['mean_pnl']:>10.4%} {t['sharpe']:>8.2f} "
                  f"{t['expectancy']:>10.6f}")
    
    # Signal distribution
    dist = results.get("signal_distribution", {})
    if "error" not in dist:
        print("\n" + "-" * 70)
        print("SIGNAL DISTRIBUTION")
        print("-" * 70)
        print(f"Count: {dist.get('count', 0)}")
        print(f"Mean: {dist.get('mean', 0):.4f} ± {dist.get('std', 0):.4f}")
        print(f"Range: [{dist.get('min', 0):.4f}, {dist.get('max', 0):.4f}]")
        print(f"Median: {dist.get('median', 0):.4f}")
        
        print("\nHistogram:")
        hist = dist.get("histogram", {})
        for bucket, count in sorted(hist.items()):
            bar = "█" * min(50, count)
            print(f"  {bucket:>8}: {count:>5} {bar}")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Signal calibration analysis")
    parser.add_argument("--days", type=int, default=7, help="Analysis window (days)")
    parser.add_argument("--threshold-test", action="store_true", help="Run threshold tests")
    args = parser.parse_args()
    
    results = analyze_from_experiences(days=args.days)
    print_report(results)


if __name__ == "__main__":
    main()
