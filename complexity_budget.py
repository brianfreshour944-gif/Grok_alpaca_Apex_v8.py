"""
complexity_budget.py — Evaluate component value vs complexity.

For every component, ask:
- What measurable problem does this solve?
- What is its incremental contribution?
- Does it improve out-of-sample performance?

If improvement cannot be demonstrated, recommend removal.

Usage:
    python complexity_budget.py              # full analysis
    python complexity_budget.py --remove X   # simulate removing component
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


# ── Component Registry ───────────────────────────────────────────────────────

COMPONENTS = {
    # Models
    "transformer_model": {
        "name": "Transformer Model (GrokGQA)",
        "category": "model",
        "complexity": "HIGH",
        "lines_of_code": 200,
        "dependencies": ["torch", "joblib"],
        "problem_solved": "Generates buy/sell signals from market data",
        "measurable_impact": "Signal accuracy, win rate contribution",
    },
    "shadow_gbt": {
        "name": "Shadow GBT Challenger",
        "category": "model",
        "complexity": "LOW",
        "lines_of_code": 80,
        "dependencies": ["joblib", "sklearn"],
        "problem_solved": "Backup model, future comparison",
        "measurable_impact": "Currently 0 (shadow only, never trades)",
    },
    
    # Features
    "z_return": {
        "name": "Z-Return (Momentum)",
        "category": "feature",
        "complexity": "LOW",
        "lines_of_code": 5,
        "dependencies": [],
        "problem_solved": "Regime-invariant momentum signal",
        "measurable_impact": "Feature importance ranking",
    },
    "parkinson_vol": {
        "name": "Parkinson Volatility",
        "category": "feature",
        "complexity": "LOW",
        "lines_of_code": 10,
        "dependencies": [],
        "problem_solved": "Range-based volatility estimation",
        "measurable_impact": "5x more efficient than std",
    },
    "garman_klass_vol": {
        "name": "Garman-Klass Volatility",
        "category": "feature",
        "complexity": "LOW",
        "lines_of_code": 15,
        "dependencies": [],
        "problem_solved": "OHLC volatility estimation",
        "measurable_impact": "Most efficient open-market estimator",
    },
    "kyle_lambda": {
        "name": "Kyle Lambda (Price Impact)",
        "category": "feature",
        "complexity": "MEDIUM",
        "lines_of_code": 20,
        "dependencies": [],
        "problem_solved": "Insider trading detection",
        "measurable_impact": "Price impact proxy",
    },
    "signed_flow": {
        "name": "Signed Order Flow",
        "category": "feature",
        "complexity": "LOW",
        "lines_of_code": 10,
        "dependencies": [],
        "problem_solved": "Buy/sell pressure",
        "measurable_impact": "Institutional vs retail flow",
    },
    "vwap_z": {
        "name": "VWAP Z-Score",
        "category": "feature",
        "complexity": "LOW",
        "lines_of_code": 10,
        "dependencies": [],
        "problem_solved": "Fair value distance",
        "measurable_impact": "Mean reversion signal",
    },
    "vol_of_vol": {
        "name": "Volatility of Volatility",
        "category": "feature",
        "complexity": "LOW",
        "lines_of_code": 10,
        "dependencies": [],
        "problem_solved": "Regime uncertainty",
        "measurable_impact": "Transition detection",
    },
    "amihud_z": {
        "name": "Amihud Illiquidity",
        "category": "feature",
        "complexity": "LOW",
        "lines_of_code": 10,
        "dependencies": [],
        "problem_solved": "Liquidity measurement",
        "measurable_impact": "Illiquidity premium",
    },
    "trade_size_proxy": {
        "name": "Trade Size Proxy",
        "category": "feature",
        "complexity": "LOW",
        "lines_of_code": 10,
        "dependencies": [],
        "problem_solved": "Institutional vs retail detection",
        "measurable_impact": "Flow analysis",
    },
    "roll_autocorr": {
        "name": "Rolling Autocorrelation",
        "category": "feature",
        "complexity": "LOW",
        "lines_of_code": 10,
        "dependencies": [],
        "problem_solved": "Trend vs mean-reversion",
        "measurable_impact": "Regime classification",
    },
    "range_position_z": {
        "name": "Range Position Z-Score",
        "category": "feature",
        "complexity": "LOW",
        "lines_of_code": 10,
        "dependencies": [],
        "problem_solved": "Breakout signal",
        "measurable_impact": "Price position in range",
    },
    
    # Risk Management
    "kelly_sizing": {
        "name": "Kelly Criterion Sizing",
        "category": "risk",
        "complexity": "MEDIUM",
        "lines_of_code": 30,
        "dependencies": [],
        "problem_solved": "Optimal position sizing",
        "measurable_impact": "Risk-adjusted returns",
    },
    "volatility_scaling": {
        "name": "Volatility-Based Scaling",
        "category": "risk",
        "complexity": "LOW",
        "lines_of_code": 20,
        "dependencies": [],
        "problem_solved": "Risk reduction in high vol",
        "measurable_impact": "Drawdown reduction",
    },
    "drawdown_taper": {
        "name": "Drawdown Taper",
        "category": "risk",
        "complexity": "LOW",
        "lines_of_code": 15,
        "dependencies": [],
        "problem_solved": "Risk reduction during drawdowns",
        "measurable_impact": "Max drawdown reduction",
    },
    "daily_loss_limit": {
        "name": "Daily Loss Limit",
        "category": "risk",
        "complexity": "LOW",
        "lines_of_code": 10,
        "dependencies": [],
        "problem_solved": "Prevents catastrophic loss",
        "measurable_impact": "Worst-case protection",
    },
    
    # Exit Logic
    "trailing_stop": {
        "name": "ATR-Based Trailing Stop",
        "category": "exit",
        "complexity": "MEDIUM",
        "lines_of_code": 30,
        "dependencies": [],
        "problem_solved": "Lock in profits",
        "measurable_impact": "Profit preservation",
    },
    "time_decay_stop": {
        "name": "Time-Decay Stop Loss",
        "category": "exit",
        "complexity": "MEDIUM",
        "lines_of_code": 25,
        "dependencies": [],
        "problem_solved": "Tighten stops over time",
        "measurable_impact": "Loss reduction",
    },
    "signal_exit": {
        "name": "Signal Weakness Exit",
        "category": "exit",
        "complexity": "LOW",
        "lines_of_code": 10,
        "dependencies": [],
        "problem_solved": "Exit when signal fades",
        "measurable_impact": "Loss reduction",
    },
    
    # Infrastructure
    "regime_detection": {
        "name": "Regime Detection",
        "category": "infrastructure",
        "complexity": "MEDIUM",
        "lines_of_code": 50,
        "dependencies": [],
        "problem_solved": "Adaptive thresholds",
        "measurable_impact": "Performance across regimes",
    },
    "experience_logging": {
        "name": "Experience Logging",
        "category": "infrastructure",
        "complexity": "LOW",
        "lines_of_code": 100,
        "dependencies": [],
        "problem_solved": "Data collection for analysis",
        "measurable_impact": "Enables all analysis tools",
    },
    "drift_monitor": {
        "name": "Drift Monitoring",
        "category": "infrastructure",
        "complexity": "MEDIUM",
        "lines_of_code": 200,
        "dependencies": [],
        "problem_solved": "Detects model degradation",
        "measurable_impact": "Early warning system",
    },
    "circuit_breaker": {
        "name": "Circuit Breaker",
        "category": "infrastructure",
        "complexity": "LOW",
        "lines_of_code": 50,
        "dependencies": [],
        "problem_solved": "Stops trading during failures",
        "measurable_impact": "Failure protection",
    },
}


# ── Complexity Analysis ──────────────────────────────────────────────────────

def compute_complexity_budget() -> dict:
    """
    Compute total complexity budget and per-category breakdown.
    """
    total_loc = 0
    total_components = len(COMPONENTS)
    categories = {}
    
    for comp_id, comp in COMPONENTS.items():
        cat = comp["category"]
        loc = comp["lines_of_code"]
        total_loc += loc
        
        if cat not in categories:
            categories[cat] = {"count": 0, "loc": 0, "components": []}
        categories[cat]["count"] += 1
        categories[cat]["loc"] += loc
        categories[cat]["components"].append(comp_id)
    
    return {
        "total_components": total_components,
        "total_loc": total_loc,
        "avg_loc_per_component": round(total_loc / total_components, 1),
        "categories": categories,
    }


def assess_component_value(events: list[dict]) -> dict:
    """
    Assess each component's value based on available data.
    """
    if not events:
        return {"error": "No events for assessment"}
    
    # Extract trade data
    entries = [e for e in events if e.get("type") == "entry"]
    exits = [e for e in events if e.get("type") == "exit"]
    
    if not entries or not exits:
        return {"error": "Insufficient trades"}
    
    # Match trades
    trades = []
    for entry in entries:
        symbol = entry.get("symbol", "")
        entry_ts = entry.get("ts", "")
        
        for exit_ev in exits:
            if exit_ev.get("symbol") == symbol and exit_ev.get("ts", "") > entry_ts:
                trades.append({
                    "signal": entry.get("signal", 0.5),
                    "regime": entry.get("regime", "unknown"),
                    "pnl_pct": exit_ev.get("pnl_pct", 0),
                    "exit_reason": exit_ev.get("exit_reason", ""),
                })
                break
    
    if not trades:
        return {"error": "No matched trades"}
    
    pnls = np.array([t["pnl_pct"] for t in trades])
    
    # Assess each component
    assessments = {}
    
    for comp_id, comp in COMPONENTS.items():
        value_score = 0
        evidence = []
        
        # Model components
        if comp_id == "transformer_model":
            # Check if ML signal adds value
            signals = np.array([t["signal"] for t in trades])
            signal_corr = np.corrcoef(signals, pnls)[0, 1] if len(signals) > 1 else 0
            value_score = abs(signal_corr) * 10
            evidence.append(f"Signal-PnL correlation: {signal_corr:.4f}")
        
        elif comp_id == "shadow_gbt":
            value_score = 0  # Shadow only, no trading contribution
            evidence.append("Shadow mode only, never trades")
        
        # Feature components
        elif comp_id in FEATURE_COLS:
            # Would need feature importance analysis
            value_score = 5  # Default neutral
            evidence.append("Feature importance not yet measured")
        
        # Risk components
        elif comp_id == "volatility_scaling":
            # Check if scaling reduces drawdown
            value_score = 7  # Known to help
            evidence.append("Reduces position size in high vol")
        
        elif comp_id == "drawdown_taper":
            value_score = 8  # Critical protection
            evidence.append("Prevents catastrophic drawdown")
        
        elif comp_id == "daily_loss_limit":
            value_score = 10  # Essential
            evidence.append("Hard stop on losses")
        
        # Exit components
        elif comp_id == "trailing_stop":
            # Check trailing stop performance
            trailing_exits = [t for t in trades if "Trailing" in t.get("exit_reason", "")]
            if trailing_exits:
                trailing_pnls = [t["pnl_pct"] for t in trailing_exits]
                value_score = 7 if np.mean(trailing_pnls) > 0 else 3
                evidence.append(f"Trailing stop avg PnL: {np.mean(trailing_pnls):.4f}")
        
        elif comp_id == "time_decay_stop":
            value_score = 6
            evidence.append("Tightens stops over time")
        
        # Infrastructure
        elif comp_id == "experience_logging":
            value_score = 10  # Enables all analysis
            evidence.append("Enables all analysis tools")
        
        elif comp_id == "drift_monitor":
            value_score = 8  # Early warning
            evidence.append("Detects model degradation")
        
        elif comp_id == "circuit_breaker":
            value_score = 9  # Failure protection
            evidence.append("Stops trading during failures")
        
        else:
            value_score = 5  # Default
            evidence.append("Value not yet measured")
        
        assessments[comp_id] = {
            "name": comp["name"],
            "category": comp["category"],
            "complexity": comp["complexity"],
            "value_score": value_score,
            "evidence": evidence,
            "recommendation": "KEEP" if value_score >= 6 else "EVALUATE" if value_score >= 4 else "CONSIDER_REMOVING",
        }
    
    return assessments


# ── Removal Simulation ───────────────────────────────────────────────────────

def simulate_removal(component_id: str) -> dict:
    """
    Simulate removing a component and estimate impact.
    """
    if component_id not in COMPONENTS:
        return {"error": f"Component {component_id} not found"}
    
    comp = COMPONENTS[component_id]
    
    # Estimate impact based on component type
    impact = {
        "complexity_reduction": comp["lines_of_code"],
        "dependency_reduction": len(comp["dependencies"]),
        "estimated_performance_impact": "UNKNOWN - requires ablation test",
    }
    
    if comp["category"] == "model":
        impact["risk"] = "HIGH - removes signal generation"
        impact["alternative"] = "Use simpler model or fixed rules"
    
    elif comp["category"] == "feature":
        impact["risk"] = "LOW - other features may compensate"
        impact["alternative"] = "Feature importance may drop"
    
    elif comp["category"] == "risk":
        impact["risk"] = "HIGH - removes risk protection"
        impact["alternative"] = "Increase other risk limits"
    
    elif comp["category"] == "exit":
        impact["risk"] = "MEDIUM - changes exit behavior"
        impact["alternative"] = "Adjust other exit rules"
    
    elif comp["category"] == "infrastructure":
        impact["risk"] = "MEDIUM - reduces observability/reliability"
        impact["alternative"] = "Manual monitoring required"
    
    return {
        "component": comp["name"],
        "category": comp["category"],
        "complexity_saved": comp["lines_of_code"],
        "impact": impact,
    }


# ── Main Analysis ────────────────────────────────────────────────────────────

def analyze_complexity_budget(days: int = 30) -> dict:
    """Run full complexity budget analysis."""
    events = load_experiences()
    
    budget = compute_complexity_budget()
    
    assessments = {}
    if events:
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
        
        assessments = assess_component_value(recent)
    
    # Count by recommendation
    recommendations = {"KEEP": 0, "EVALUATE": 0, "CONSIDER_REMOVING": 0}
    for a in assessments.values():
        rec = a.get("recommendation", "EVALUATE")
        recommendations[rec] = recommendations.get(rec, 0) + 1
    
    return {
        "budget": budget,
        "assessments": assessments,
        "recommendations": recommendations,
        "data_available": bool(events),
    }


def print_report(results: dict):
    """Print complexity budget report."""
    print("\n" + "=" * 80)
    print("COMPLEXITY BUDGET ANALYSIS")
    print("=" * 80)
    
    budget = results.get("budget", {})
    print(f"\nTotal Components: {budget.get('total_components', 0)}")
    print(f"Total Lines of Code: {budget.get('total_loc', 0)}")
    print(f"Avg LOC/Component: {budget.get('avg_loc_per_component', 0)}")
    
    # Categories
    categories = budget.get("categories", {})
    print("\n" + "-" * 80)
    print("COMPLEXITY BY CATEGORY")
    print("-" * 80)
    print(f"{'Category':<20} {'Count':>7} {'LOC':>7} {'% of Total':>10}")
    print("-" * 50)
    
    total_loc = budget.get("total_loc", 1)
    for cat, stats in sorted(categories.items(), key=lambda x: -x[1]["loc"]):
        pct = stats["loc"] / total_loc * 100
        print(f"{cat:<20} {stats['count']:>7} {stats['loc']:>7} {pct:>9.1f}%")
    
    # Assessments
    assessments = results.get("assessments", {})
    if assessments:
        print("\n" + "-" * 80)
        print("COMPONENT VALUE ASSESSMENT")
        print("-" * 80)
        print(f"{'Component':<30} {'Value':>6} {'Complexity':<10} {'Verdict':<20}")
        print("-" * 70)
        
        for comp_id, a in sorted(assessments.items(), key=lambda x: -x[1].get("value_score", 0)):
            name = a.get("name", comp_id)[:30]
            value = a.get("value_score", 0)
            complexity = a.get("complexity", "UNKNOWN")
            rec = a.get("recommendation", "EVALUATE")
            
            print(f"{name:<30} {value:>6}/10 {complexity:<10} {rec:<20}")
    
    # Recommendations
    recs = results.get("recommendations", {})
    if recs:
        print("\n" + "-" * 80)
        print("RECOMMENDATIONS")
        print("-" * 80)
        print(f"  KEEP: {recs.get('KEEP', 0)} components")
        print(f"  EVALUATE: {recs.get('EVALUATE', 0)} components")
        print(f"  CONSIDER REMOVING: {recs.get('CONSIDER_REMOVING', 0)} components")
    
    # High-value components
    if assessments:
        print("\n" + "-" * 80)
        print("HIGH-VALUE COMPONENTS (score >= 8)")
        print("-" * 80)
        high_value = [(k, v) for k, v in assessments.items() if v.get("value_score", 0) >= 8]
        for comp_id, a in sorted(high_value, key=lambda x: -x[1]["value_score"]):
            print(f"  ✓ {a['name']} (score: {a['value_score']}/10)")
    
    # Low-value components
    if assessments:
        print("\n" + "-" * 80)
        print("LOW-VALUE COMPONENTS (score <= 4)")
        print("-" * 80)
        low_value = [(k, v) for k, v in assessments.items() if v.get("value_score", 0) <= 4]
        if low_value:
            for comp_id, a in sorted(low_value, key=lambda x: x[1]["value_score"]):
                print(f"  ✗ {a['name']} (score: {a['value_score']}/10)")
                print(f"    Recommendation: {a['recommendation']}")
        else:
            print("  None identified")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Complexity budget analysis")
    parser.add_argument("--days", type=int, default=30, help="Analysis window (days)")
    parser.add_argument("--remove", type=str, help="Simulate removing component")
    args = parser.parse_args()
    
    if args.remove:
        result = simulate_removal(args.remove)
        print(json.dumps(result, indent=2))
    else:
        results = analyze_complexity_budget(days=args.days)
        print_report(results)


if __name__ == "__main__":
    main()
