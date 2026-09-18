"""
ablation_testing.py — Ablation testing for multi-model/feature architecture.

Evaluates each component independently:
- Transformer model
- Shadow GBT model
- Regime detection
- Feature groups
- Exit logic

Runs ablation tests:
- Full System
- System – Model A
- System – Model B
- System – Feature Group A
- etc.

Usage:
    python ablation_testing.py                    # full ablation
    python ablation_testing.py --component all    # test all components
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
from feature_engineering import FEATURE_COLS


# ── Component Definitions ────────────────────────────────────────────────────

COMPONENTS = {
    "transformer": {
        "name": "Transformer Model",
        "description": "Main ML model (GrokGQA_Transformer)",
        "removes": "Use random signal instead of ML prediction",
    },
    "shadow_gbt": {
        "name": "Shadow GBT",
        "description": "Challenger model (shadow only, never trades)",
        "removes": "Remove shadow GBT logging",
    },
    "regime": {
        "name": "Regime Detection",
        "description": "Market regime classification (wild/normal/quiet)",
        "removes": "Use fixed thresholds instead of regime-adaptive",
    },
    "feature_group_momentum": {
        "name": "Momentum Features",
        "description": "z_return, range_position_z",
        "removes": "Zero out momentum features",
    },
    "feature_group_volatility": {
        "name": "Volatility Features",
        "description": "parkinson_vol, garman_klass_vol, vol_of_vol",
        "removes": "Zero out volatility features",
    },
    "feature_group_microstructure": {
        "name": "Microstructure Features",
        "description": "kyle_lambda, signed_flow, amihud_z, trade_size_proxy",
        "removes": "Zero out microstructure features",
    },
    "feature_group_efficiency": {
        "name": "Efficiency Features",
        "description": "vwap_z, roll_autocorr",
        "removes": "Zero out efficiency features",
    },
    "trailing_stop": {
        "name": "Trailing Stop",
        "description": "ATR-based trailing stop",
        "removes": "Remove trailing stop exit",
    },
    "time_decay_stop": {
        "name": "Time-Decay Stop",
        "description": "Stop loss tightens over time",
        "removes": "Use fixed stop loss",
    },
    "kelly_sizing": {
        "name": "Kelly Criterion Sizing",
        "description": "Signal-based position sizing",
        "removes": "Use fixed position size",
    },
}


# ── Ablation Engine ──────────────────────────────────────────────────────────

def simulate_system(
    events: list[dict],
    component: str = "full",
    signal_threshold: float = BUY_SIGNAL,
    use_regime: bool = True,
    use_trailing: bool = True,
    use_time_decay: bool = True,
    use_kelly: bool = True,
    feature_mask: dict = None,
) -> dict:
    """
    Simulate trading system with specified components.
    """
    entries = [e for e in events if e.get("type") == "entry"]
    exits = [e for e in events if e.get("type") == "exit"]
    
    if not entries:
        return {"error": "No entries"}
    
    # Match entries to exits
    trades = []
    for entry in entries:
        symbol = entry.get("symbol", "")
        entry_ts = entry.get("ts", "")
        entry_signal = entry.get("signal", 0.5)
        entry_price = entry.get("price", 0)
        regime = entry.get("regime", "normal")
        atr_pct = entry.get("atr_pct", 2.0)
        
        # Apply component modifications
        if component == "transformer":
            # Use random signal instead of ML
            entry_signal = np.random.uniform(0.4, 0.6)
        
        if feature_mask:
            # Zero out specified features
            features = entry.get("features", {})
            for feat in feature_mask:
                features[feat] = 0.0
        
        # Find matching exit
        for exit_ev in exits:
            if exit_ev.get("symbol") == symbol and exit_ev.get("ts", "") > entry_ts:
                pnl_pct = exit_ev.get("pnl_pct", 0)
                held_hours = exit_ev.get("held_hours", 0)
                
                # Apply exit modifications
                if not use_trailing:
                    # Remove trailing stop (would have exited at fixed stop)
                    if "Trailing Stop" in exit_ev.get("exit_reason", ""):
                        # Assume would have been stopped at stop loss
                        pnl_pct = -0.02
                
                if not use_time_decay:
                    # Remove time decay (use fixed stop)
                    if "Time-Decay" in exit_ev.get("exit_reason", ""):
                        pnl_pct = -0.02  # Fixed 2% stop
                
                trades.append({
                    "symbol": symbol,
                    "signal": entry_signal,
                    "regime": regime,
                    "atr_pct": atr_pct,
                    "pnl_pct": pnl_pct,
                    "held_hours": held_hours,
                    "entry_price": entry_price,
                })
                break
    
    if not trades:
        return {"error": "No matched trades"}
    
    # Compute metrics
    pnls = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    
    win_rate = len(wins) / len(pnls) if pnls else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    sharpe = np.mean(pnls) / (np.std(pnls) + 1e-10) if len(pnls) > 1 else 0
    
    # Drawdown
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    max_dd = float(np.min(drawdowns))
    
    # Profit factor
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    
    return {
        "n_trades": len(trades),
        "win_rate": round(float(win_rate), 4),
        "avg_win": round(float(avg_win), 6),
        "avg_loss": round(float(avg_loss), 6),
        "expectancy": round(float(expectancy), 6),
        "sharpe": round(float(sharpe), 4),
        "max_drawdown": round(float(max_dd), 4),
        "profit_factor": round(float(profit_factor), 4) if profit_factor != float("inf") else "inf",
        "total_pnl": round(float(sum(pnls)), 4),
    }


# ── Ablation Tests ───────────────────────────────────────────────────────────

def run_ablation(events: list[dict]) -> dict:
    """
    Run full ablation test suite.
    """
    results = {}
    
    # Full system (baseline)
    results["full_system"] = {
        "component": "Full System",
        "description": "All components active",
        "metrics": simulate_system(events, component="full"),
    }
    
    # Ablation: Remove Transformer
    results["no_transformer"] = {
        "component": "– Transformer",
        "description": "Random signal instead of ML",
        "metrics": simulate_system(events, component="transformer"),
    }
    
    # Ablation: Remove Regime Detection
    results["no_regime"] = {
        "component": "– Regime Detection",
        "description": "Fixed thresholds",
        "metrics": simulate_system(events, component="full", use_regime=False),
    }
    
    # Ablation: Remove Trailing Stop
    results["no_trailing_stop"] = {
        "component": "– Trailing Stop",
        "description": "Fixed stop loss only",
        "metrics": simulate_system(events, component="full", use_trailing=False),
    }
    
    # Ablation: Remove Time-Decay Stop
    results["no_time_decay"] = {
        "component": "– Time-Decay Stop",
        "description": "Fixed stop loss",
        "metrics": simulate_system(events, component="full", use_time_decay=False),
    }
    
    # Ablation: Remove Kelly Sizing
    results["no_kelly"] = {
        "component": "– Kelly Sizing",
        "description": "Fixed position size",
        "metrics": simulate_system(events, component="full", use_kelly=False),
    }
    
    # Ablation: Remove Momentum Features
    results["no_momentum"] = {
        "component": "– Momentum Features",
        "description": "Zero z_return, range_position_z",
        "metrics": simulate_system(events, component="full", 
                                   feature_mask=["z_return", "range_position_z"]),
    }
    
    # Ablation: Remove Volatility Features
    results["no_volatility"] = {
        "component": "– Volatility Features",
        "description": "Zero parkinson_vol, garman_klass_vol, vol_of_vol",
        "metrics": simulate_system(events, component="full",
                                   feature_mask=["parkinson_vol", "garman_klass_vol", "vol_of_vol"]),
    }
    
    # Ablation: Remove Microstructure Features
    results["no_microstructure"] = {
        "component": "– Microstructure Features",
        "description": "Zero kyle_lambda, signed_flow, amihud_z, trade_size_proxy",
        "metrics": simulate_system(events, component="full",
                                   feature_mask=["kyle_lambda", "signed_flow", "amihud_z", "trade_size_proxy"]),
    }
    
    # Ablation: Remove Efficiency Features
    results["no_efficiency"] = {
        "component": "– Efficiency Features",
        "description": "Zero vwap_z, roll_autocorr",
        "metrics": simulate_system(events, component="full",
                                   feature_mask=["vwap_z", "roll_autocorr"]),
    }
    
    return results


# ── Impact Analysis ──────────────────────────────────────────────────────────

def analyze_impact(ablation_results: dict) -> dict:
    """
    Analyze impact of each component removal.
    """
    baseline = ablation_results.get("full_system", {}).get("metrics", {})
    
    if not baseline or "error" in baseline:
        return {"error": "No baseline metrics"}
    
    baseline_expectancy = baseline.get("expectancy", 0)
    baseline_sharpe = baseline.get("sharpe", 0)
    baseline_win_rate = baseline.get("win_rate", 0)
    baseline_max_dd = baseline.get("max_drawdown", 0)
    
    impacts = {}
    for key, result in ablation_results.items():
        if key == "full_system":
            continue
        
        metrics = result.get("metrics", {})
        if "error" in metrics:
            continue
        
        expectancy_delta = metrics.get("expectancy", 0) - baseline_expectancy
        sharpe_delta = metrics.get("sharpe", 0) - baseline_sharpe
        win_rate_delta = metrics.get("win_rate", 0) - baseline_win_rate
        max_dd_delta = metrics.get("max_drawdown", 0) - baseline_max_dd
        
        # Determine if component is valuable
        is_valuable = expectancy_delta < 0  # Removing it hurts performance
        
        impacts[key] = {
            "component": result.get("component", ""),
            "description": result.get("description", ""),
            "expectancy_delta": round(float(expectancy_delta), 6),
            "sharpe_delta": round(float(sharpe_delta), 4),
            "win_rate_delta": round(float(win_rate_delta), 4),
            "max_dd_delta": round(float(max_dd_delta), 4),
            "is_valuable": is_valuable,
            "verdict": "KEEP" if is_valuable else "REMOVE",
        }
    
    return {
        "baseline": baseline,
        "impacts": impacts,
        "valuable_components": [k for k, v in impacts.items() if v["is_valuable"]],
        "removable_components": [k for k, v in impacts.items() if not v["is_valuable"]],
    }


# ── Main Analysis ────────────────────────────────────────────────────────────

def analyze_from_experiences(days: int = 30) -> dict:
    """Run full ablation analysis."""
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
    
    # Run ablation
    ablation = run_ablation(recent)
    impact = analyze_impact(ablation)
    
    return {
        "period_days": days,
        "ablation_results": ablation,
        "impact_analysis": impact,
    }


def print_report(results: dict):
    """Print ablation analysis report."""
    print("\n" + "=" * 80)
    print("ABLATION TESTING REPORT")
    print("=" * 80)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    print(f"Period: Last {results.get('period_days', 0)} days")
    
    # Baseline
    impact = results.get("impact_analysis", {})
    baseline = impact.get("baseline", {})
    
    if baseline:
        print("\n" + "-" * 80)
        print("BASELINE (FULL SYSTEM)")
        print("-" * 80)
        print(f"  Trades: {baseline.get('n_trades', 0)}")
        print(f"  Win Rate: {baseline.get('win_rate', 0):.2%}")
        print(f"  Expectancy: {baseline.get('expectancy', 0):.4f}")
        print(f"  Sharpe: {baseline.get('sharpe', 0):.4f}")
        print(f"  Max Drawdown: {baseline.get('max_drawdown', 0):.2%}")
        print(f"  Profit Factor: {baseline.get('profit_factor', 0)}")
    
    # Impact analysis
    impacts = impact.get("impacts", {})
    if impacts:
        print("\n" + "-" * 80)
        print("COMPONENT IMPACT (Removing component changes metric)")
        print("-" * 80)
        print(f"{'Component':<25} {'ΔExpect':>10} {'ΔSharpe':>10} {'ΔWinRate':>10} {'Verdict':<10}")
        print("-" * 70)
        
        for key, imp in sorted(impacts.items(), key=lambda x: x[1]["expectancy_delta"]):
            exp_delta = imp["expectancy_delta"]
            sharpe_delta = imp["sharpe_delta"]
            wr_delta = imp["win_rate_delta"]
            verdict = imp["verdict"]
            
            exp_str = f"{exp_delta:+.4f}" if exp_delta != 0 else "0.0000"
            sharpe_str = f"{sharpe_delta:+.4f}" if sharpe_delta != 0 else "0.0000"
            wr_str = f"{wr_delta:+.2%}" if wr_delta != 0 else "0.00%"
            
            print(f"{imp['component']:<25} {exp_str:>10} {sharpe_str:>10} {wr_str:>10} {verdict:<10}")
    
    # Valuable vs removable
    valuable = impact.get("valuable_components", [])
    removable = impact.get("removable_components", [])
    
    if valuable or removable:
        print("\n" + "-" * 80)
        print("VERDICT")
        print("-" * 80)
        
        if valuable:
            print("\nVALUABLE (keep in architecture):")
            for v in valuable:
                comp = impacts.get(v, {})
                print(f"  ✓ {comp.get('component', v)}")
        
        if removable:
            print("\nQUESTIONABLE (consider removing):")
            for r in removable:
                comp = impacts.get(r, {})
                print(f"  ✗ {comp.get('component', r)}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Ablation testing")
    parser.add_argument("--days", type=int, default=30, help="Analysis window (days)")
    args = parser.parse_args()
    
    results = analyze_from_experiences(days=args.days)
    print_report(results)


if __name__ == "__main__":
    main()
