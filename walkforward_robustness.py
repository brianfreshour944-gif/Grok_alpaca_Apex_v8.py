"""
walkforward_robustness.py — Walk-forward robustness testing.

Tests strategy stability across multiple rolling periods.
Checks if performance remains stable or if parameters are chasing noise.

Usage:
    python walkforward_robustness.py              # full analysis
    python walkforward_robustness.py --periods 12 # custom periods
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


# ── Rolling Period Analysis ──────────────────────────────────────────────────

def compute_rolling_metrics(events: list[dict], period_days: int = 7) -> dict:
    """
    Compute performance metrics for each rolling period.
    """
    entries = [e for e in events if e.get("type") == "entry"]
    exits = [e for e in events if e.get("type") == "exit"]
    
    if not entries:
        return {"error": "No entries"}
    
    # Get date range
    timestamps = []
    for ev in entries:
        ts_str = ev.get("ts", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            timestamps.append(ts)
        except (ValueError, TypeError):
            continue
    
    if not timestamps:
        return {"error": "No valid timestamps"}
    
    min_ts = min(timestamps)
    max_ts = max(timestamps)
    
    # Create rolling periods
    periods = []
    current_start = min_ts
    
    while current_start < max_ts:
        current_end = current_start + timedelta(days=period_days)
        
        # Get events in this period
        period_entries = []
        period_exits = []
        
        for ev in entries:
            ts_str = ev.get("ts", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if current_start <= ts < current_end:
                    period_entries.append(ev)
            except (ValueError, TypeError):
                continue
        
        for ev in exits:
            ts_str = ev.get("ts", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if current_start <= ts < current_end:
                    period_exits.append(ev)
            except (ValueError, TypeError):
                continue
        
        # Compute period metrics
        if period_entries:
            pnls = [e.get("pnl_pct", 0) for e in period_exits]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            
            win_rate = len(wins) / len(pnls) if pnls else 0
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
            sharpe = np.mean(pnls) / (np.std(pnls) + 1e-10) if len(pnls) > 1 else 0
            
            # Feature importance (average signal strength)
            signals = [e.get("signal", 0.5) for e in period_entries]
            avg_signal = np.mean(signals)
            
            # Drawdown
            cumulative = np.cumsum(pnls) if pnls else [0]
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = cumulative - running_max
            max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0
            
            periods.append({
                "start": str(current_start),
                "end": str(current_end),
                "n_trades": len(period_exits),
                "win_rate": round(float(win_rate), 4),
                "avg_win": round(float(avg_win), 6),
                "avg_loss": round(float(avg_loss), 6),
                "expectancy": round(float(expectancy), 6),
                "sharpe": round(float(sharpe), 4),
                "max_drawdown": round(float(max_dd), 4),
                "avg_signal": round(float(avg_signal), 4),
                "total_pnl": round(float(sum(pnls)), 4) if pnls else 0,
            })
        
        current_start = current_end
    
    return {
        "period_days": period_days,
        "total_periods": len(periods),
        "date_range": {
            "start": str(min_ts),
            "end": str(max_ts),
            "total_days": (max_ts - min_ts).days,
        },
        "periods": periods,
    }


# ── Stability Analysis ──────────────────────────────────────────────────────

def analyze_stability(periods: list[dict]) -> dict:
    """
    Analyze stability of metrics across periods.
    """
    if not periods:
        return {"error": "No periods"}
    
    # Extract metrics
    win_rates = [p["win_rate"] for p in periods]
    expectancies = [p["expectancy"] for p in periods]
    sharpes = [p["sharpe"] for p in periods]
    max_dds = [p["max_drawdown"] for p in periods]
    trade_counts = [p["n_trades"] for p in periods]
    signals = [p["avg_signal"] for p in periods]
    
    def compute_stability(values, name):
        if not values:
            return {name: "N/A"}
        
        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr)
        cv = std / abs(mean) if abs(mean) > 1e-10 else float("inf")
        
        # Coefficient of Variation interpretation
        if cv < 0.2:
            stability = "STABLE"
        elif cv < 0.5:
            stability = "MODERATE"
        else:
            stability = "UNSTABLE"
        
        return {
            "mean": round(float(mean), 6),
            "std": round(float(std), 6),
            "cv": round(float(cv), 4),
            "min": round(float(np.min(arr)), 6),
            "max": round(float(np.max(arr)), 6),
            "range": round(float(np.max(arr) - np.min(arr)), 6),
            "stability": stability,
        }
    
    return {
        "win_rate": compute_stability(win_rates, "win_rate"),
        "expectancy": compute_stability(expectancies, "expectancy"),
        "sharpe": compute_stability(sharpes, "sharpe"),
        "max_drawdown": compute_stability(max_dds, "max_drawdown"),
        "trade_frequency": compute_stability(trade_counts, "trade_frequency"),
        "signal_strength": compute_stability(signals, "signal_strength"),
        "overall_assessment": assess_overall_stability(
            win_rates, expectancies, sharpes, max_dds, trade_counts
        ),
    }


def assess_overall_stability(
    win_rates, expectancies, sharpes, max_dds, trade_counts
) -> dict:
    """
    Provide overall stability assessment.
    """
    issues = []
    
    # Check win rate stability
    wr_arr = np.array(win_rates)
    if np.std(wr_arr) > 0.15:
        issues.append(f"Win rate varies widely: {np.min(wr_arr):.1%} to {np.max(wr_arr):.1%}")
    
    # Check expectancy consistency
    exp_arr = np.array(expectancies)
    negative_periods = np.sum(exp_arr < 0)
    if negative_periods > len(exp_arr) * 0.3:
        issues.append(f"{negative_periods}/{len(exp_arr)} periods have negative expectancy")
    
    # Check for regime changes
    if len(exp_arr) >= 4:
        first_half = exp_arr[:len(exp_arr)//2]
        second_half = exp_arr[len(exp_arr)//2:]
        if np.mean(first_half) > 0 and np.mean(second_half) < 0:
            issues.append("Strategy performance degraded in recent periods")
        elif np.mean(first_half) < 0 and np.mean(second_half) > 0:
            issues.append("Strategy improved in recent periods (check for overfitting)")
    
    # Check trade frequency
    tc_arr = np.array(trade_counts)
    if np.max(tc_arr) > np.min(tc_arr) * 3:
        issues.append(f"Trade frequency varies 3x: {np.min(tc_arr)} to {np.max(tc_arr)} trades")
    
    # Check drawdown
    dd_arr = np.array(max_dds)
    if np.min(dd_arr) < -0.05:
        issues.append(f"Severe drawdown in some periods: {np.min(dd_arr):.2%}")
    
    if not issues:
        return {
            "status": "ROBUST",
            "message": "Strategy shows stable performance across periods",
        }
    elif len(issues) <= 2:
        return {
            "status": "MODERATE",
            "message": "Strategy has some instability",
            "issues": issues,
        }
    else:
        return {
            "status": "FRAGILE",
            "message": "Strategy may be chasing noise",
            "issues": issues,
        }


# ── Parameter Stability ─────────────────────────────────────────────────────

def analyze_parameter_stability(events: list[dict]) -> dict:
    """
    Analyze if key parameters remain stable across periods.
    """
    entries = [e for e in events if e.get("type") == "entry"]
    
    if not entries:
        return {"error": "No entries"}
    
    # Group by week
    weekly_data = {}
    for ev in entries:
        ts_str = ev.get("ts", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            week_key = ts.strftime("%Y-W%W")
            if week_key not in weekly_data:
                weekly_data[week_key] = []
            weekly_data[week_key].append(ev)
        except (ValueError, TypeError):
            continue
    
    # Compute weekly parameters
    weekly_params = []
    for week, evs in sorted(weekly_data.items()):
        signals = [e.get("signal", 0.5) for e in evs]
        regimes = [e.get("regime", "unknown") for e in evs]
        atr_pcts = [e.get("atr_pct", 0) for e in evs]
        
        # Regime distribution
        regime_counts = {}
        for r in regimes:
            regime_counts[r] = regime_counts.get(r, 0) + 1
        
        dominant_regime = max(regime_counts, key=regime_counts.get) if regime_counts else "unknown"
        
        weekly_params.append({
            "week": week,
            "n_trades": len(evs),
            "avg_signal": round(float(np.mean(signals)), 4),
            "signal_std": round(float(np.std(signals)), 4),
            "avg_atr_pct": round(float(np.mean(atr_pcts)), 2),
            "dominant_regime": dominant_regime,
            "regime_distribution": regime_counts,
        })
    
    # Analyze parameter drift
    if len(weekly_params) >= 2:
        signals = [p["avg_signal"] for p in weekly_params]
        atrs = [p["avg_atr_pct"] for p in weekly_params]
        
        signal_drift = np.polyfit(range(len(signals)), signals, 1)[0]
        atr_drift = np.polyfit(range(len(atrs)), atrs, 1)[0]
        
        return {
            "n_weeks": len(weekly_params),
            "signal_drift_per_week": round(float(signal_drift), 6),
            "atr_drift_per_week": round(float(atr_drift), 4),
            "signal_trending": "UP" if signal_drift > 0.01 else "DOWN" if signal_drift < -0.01 else "STABLE",
            "atr_trending": "UP" if atr_drift > 0.1 else "DOWN" if atr_drift < -0.1 else "STABLE",
            "weekly_params": weekly_params,
        }
    
    return {
        "n_weeks": len(weekly_params),
        "weekly_params": weekly_params,
    }


# ── Main Analysis ────────────────────────────────────────────────────────────

def analyze_from_experiences(days: int = 30, period_days: int = 7) -> dict:
    """Run full walk-forward robustness analysis."""
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
    
    # Run analyses
    rolling = compute_rolling_metrics(recent, period_days=period_days)
    
    stability = {}
    if "error" not in rolling:
        periods = rolling.get("periods", [])
        stability = analyze_stability(periods)
    
    params = analyze_parameter_stability(recent)
    
    return {
        "analysis_days": days,
        "period_days": period_days,
        "rolling_metrics": rolling,
        "stability": stability,
        "parameter_stability": params,
    }


def print_report(results: dict):
    """Print robustness analysis report."""
    print("\n" + "=" * 80)
    print("WALK-FORWARD ROBUSTNESS ANALYSIS")
    print("=" * 80)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    print(f"Analysis Period: Last {results['analysis_days']} days")
    print(f"Period Length: {results['period_days']} days")
    
    # ── Rolling Metrics ──────────────────────────────────────────────────
    rolling = results.get("rolling_metrics", {})
    if "error" not in rolling:
        print("\n" + "-" * 80)
        print("ROLLING PERIOD PERFORMANCE")
        print("-" * 80)
        
        periods = rolling.get("periods", [])
        if periods:
            print(f"{'Period':<25} {'Trades':>7} {'WinRate':>8} {'Expect':>10} {'Sharpe':>8} {'MaxDD':>8}")
            print("-" * 70)
            
            for p in periods:
                start = p["start"][:10]
                end = p["end"][:10]
                print(f"{start} to {end}  {p['n_trades']:>7} {p['win_rate']:>8.1%} "
                      f"{p['expectancy']:>10.4f} {p['sharpe']:>8.2f} {p['max_drawdown']:>8.2%}")
    
    # ── Stability Analysis ───────────────────────────────────────────────
    stability = results.get("stability", {})
    if "error" not in stability:
        print("\n" + "-" * 80)
        print("STABILITY ANALYSIS")
        print("-" * 80)
        
        overall = stability.get("overall_assessment", {})
        print(f"Overall: {overall.get('status', 'UNKNOWN')} - {overall.get('message', '')}")
        
        if overall.get("issues"):
            for issue in overall["issues"]:
                print(f"  ⚠ {issue}")
        
        print("\nMetric Stability:")
        for metric in ["win_rate", "expectancy", "sharpe", "max_drawdown", "trade_frequency"]:
            m = stability.get(metric, {})
            if m:
                print(f"  {metric:<20} CV: {m.get('cv', 0):.2f} [{m.get('stability', 'N/A')}]")
    
    # ── Parameter Stability ──────────────────────────────────────────────
    params = results.get("parameter_stability", {})
    if "error" not in params:
        print("\n" + "-" * 80)
        print("PARAMETER STABILITY")
        print("-" * 80)
        
        print(f"Signal Drift: {params.get('signal_drift_per_week', 0):.6f}/week [{params.get('signal_trending', 'N/A')}]")
        print(f"ATR Drift: {params.get('atr_drift_per_week', 0):.4f}/week [{params.get('atr_trending', 'N/A')}]")
        
        weekly = params.get("weekly_params", [])
        if weekly:
            print("\nWeekly Summary:")
            print(f"{'Week':<10} {'Trades':>7} {'AvgSignal':>10} {'AvgATR':>8} {'Regime':<12}")
            print("-" * 50)
            for w in weekly:
                print(f"{w['week']:<10} {w['n_trades']:>7} {w['avg_signal']:>10.4f} "
                      f"{w['avg_atr_pct']:>8.2f} {w['dominant_regime']:<12}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Walk-forward robustness analysis")
    parser.add_argument("--days", type=int, default=30, help="Analysis window (days)")
    parser.add_argument("--periods", type=int, default=7, help="Period length (days)")
    args = parser.parse_args()
    
    results = analyze_from_experiences(days=args.days, period_days=args.periods)
    print_report(results)


if __name__ == "__main__":
    main()
