"""
monte_carlo_failure.py — Monte Carlo simulation for failure analysis.

Goal: Determine how badly the strategy can fail, not prove it will succeed.

Tests:
- Randomized trade ordering
- Bootstrapped trades
- Randomized slippage/fees/delays
- Parameter perturbation

Usage:
    python monte_carlo_failure.py                    # default 10k simulations
    python monte_carlo_failure.py --simulations 100000
    python monte_carlo_failure.py --perturbation      # parameter perturbation
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from config import logger, EXPERIENCE_LOG_PATH, BASE_RISK_PERCENT
from experience_capture import load_experiences


# ── Monte Carlo Engine ───────────────────────────────────────────────────────

def run_monte_carlo(
    trades: list[dict],
    n_simulations: int = 10000,
    n_trades_per_sim: int = 252,
    initial_equity: float = 10000,
    randomize_order: bool = True,
    randomize_slippage: bool = True,
    randomize_fees: bool = True,
    randomize_delays: bool = True,
) -> dict:
    """
    Run Monte Carlo simulations to determine failure probability.
    """
    if not trades:
        return {"error": "No trades"}
    
    pnls = np.array([t.get("pnl_pct", 0) for t in trades])
    signals = np.array([t.get("signal", 0.5) for t in trades])
    
    # Base metrics
    base_win_rate = np.mean(pnls > 0)
    base_avg_pnl = np.mean(pnls)
    
    # Simulation results
    final_equities = []
    max_drawdowns = []
    losing_streaks = []
    min_equities = []
    ruin_count = 0
    
    for _ in range(n_simulations):
        equity = initial_equity
        peak = initial_equity
        max_dd = 0
        current_streak = 0
        max_streak = 0
        min_eq = initial_equity
        
        # Randomize trade order if enabled
        if randomize_order:
            indices = np.random.permutation(len(pnls))
        else:
            indices = np.arange(len(pnls))
        
        # Select trades (bootstrap if needed)
        selected = []
        for i in range(n_trades_per_sim):
            idx = indices[i % len(indices)]
            pnl = pnls[idx]
            
            # Apply randomization
            if randomize_slippage:
                slippage = np.random.normal(0, 0.001)  # 0.1% std
                pnl += slippage
            
            if randomize_fees:
                fee = np.random.uniform(0, 0.002)  # 0-0.2% fee
                pnl -= fee
            
            if randomize_delays:
                # Delays can cause slippage
                delay_slippage = np.random.exponential(0.0005)  # 0.05% avg
                pnl -= delay_slippage
            
            selected.append(pnl)
        
        # Simulate
        for pnl in selected:
            trade_size = equity * BASE_RISK_PERCENT
            equity += trade_size * pnl
            
            if equity <= 0:
                ruin_count += 1
                break
            
            peak = max(peak, equity)
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
            min_eq = min(min_eq, equity)
            
            if pnl < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        final_equities.append(equity)
        max_drawdowns.append(max_dd)
        losing_streaks.append(max_streak)
        min_equities.append(min_eq)
    
    # Compute statistics
    final_arr = np.array(final_equities)
    dd_arr = np.array(max_drawdowns)
    streak_arr = np.array(losing_streaks)
    min_arr = np.array(min_equities)
    
    return {
        "simulations": n_simulations,
        "trades_per_sim": n_trades_per_sim,
        "initial_equity": initial_equity,
        "ruin_probability": round(float(ruin_count / n_simulations), 6),
        "ruin_count": ruin_count,
        "final_equity": {
            "mean": round(float(np.mean(final_arr)), 2),
            "median": round(float(np.median(final_arr)), 2),
            "std": round(float(np.std(final_arr)), 2),
            "min": round(float(np.min(final_arr)), 2),
            "max": round(float(np.max(final_arr)), 2),
            "p5": round(float(np.percentile(final_arr, 5)), 2),
            "p10": round(float(np.percentile(final_arr, 10)), 2),
            "p25": round(float(np.percentile(final_arr, 25)), 2),
            "p75": round(float(np.percentile(final_arr, 75)), 2),
            "p95": round(float(np.percentile(final_arr, 95)), 2),
        },
        "return": {
            "mean": round(float((np.mean(final_arr) - initial_equity) / initial_equity), 4),
            "median": round(float((np.median(final_arr) - initial_equity) / initial_equity), 4),
            "p5": round(float((np.percentile(final_arr, 5) - initial_equity) / initial_equity), 4),
            "p95": round(float((np.percentile(final_arr, 95) - initial_equity) / initial_equity), 4),
        },
        "drawdown": {
            "mean": round(float(np.mean(dd_arr)), 4),
            "median": round(float(np.median(dd_arr)), 4),
            "max": round(float(np.max(dd_arr)), 4),
            "p95": round(float(np.percentile(dd_arr, 95)), 4),
            "p99": round(float(np.percentile(dd_arr, 99)), 4),
        },
        "losing_streak": {
            "mean": round(float(np.mean(streak_arr)), 1),
            "median": round(float(np.median(streak_arr)), 1),
            "max": round(float(np.max(streak_arr)), 1),
            "p95": round(float(np.percentile(streak_arr, 95)), 1),
        },
        "probability_of_losing_money": round(float(np.mean(final_arr < initial_equity)), 4),
        "probability_of_ruin": round(float(ruin_count / n_simulations), 6),
        "expected_max_drawdown": round(float(np.mean(dd_arr)), 4),
        "worst_case_drawdown": round(float(np.max(dd_arr)), 4),
        "expected_losing_streak": round(float(np.mean(streak_arr)), 1),
        "strategy_failure_probability": round(float(np.mean(final_arr < initial_equity * 0.5)), 4),
    }


# ── Parameter Perturbation ──────────────────────────────────────────────────

def run_parameter_perturbation(
    trades: list[dict],
    n_simulations: int = 5000,
    n_trades_per_sim: int = 252,
    initial_equity: float = 10000,
) -> dict:
    """
    Test strategy under perturbed parameters.
    Simulates model drift and parameter uncertainty.
    """
    if not trades:
        return {"error": "No trades"}
    
    pnls = np.array([t.get("pnl_pct", 0) for t in trades])
    
    results = []
    
    for _ in range(n_simulations):
        # Perturb parameters
        win_rate_perturbation = np.random.normal(0, 0.05)  # ±5%
        avg_pnl_perturbation = np.random.normal(0, 0.002)  # ±0.2%
        
        # Apply perturbation to trades
        perturbed_pnls = pnls + avg_pnl_perturbation
        
        # Simulate
        equity = initial_equity
        peak = initial_equity
        max_dd = 0
        
        for pnl in perturbed_pnls[:n_trades_per_sim]:
            trade_size = equity * BASE_RISK_PERCENT
            equity += trade_size * pnl
            
            if equity <= 0:
                break
            
            peak = max(peak, equity)
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
        
        results.append({
            "final_equity": equity,
            "max_drawdown": max_dd,
            "return": (equity - initial_equity) / initial_equity,
        })
    
    final_equities = np.array([r["final_equity"] for r in results])
    drawdowns = np.array([r["max_drawdown"] for r in results])
    returns = np.array([r["return"] for r in results])
    
    return {
        "simulations": n_simulations,
        "parameter_perturbation": {
            "win_rate_std": 0.05,
            "avg_pnl_std": 0.002,
        },
        "final_equity": {
            "mean": round(float(np.mean(final_equities)), 2),
            "std": round(float(np.std(final_equities)), 2),
            "p5": round(float(np.percentile(final_equities, 5)), 2),
            "p95": round(float(np.percentile(final_equities, 95)), 2),
        },
        "drawdown": {
            "mean": round(float(np.mean(drawdowns)), 4),
            "p95": round(float(np.percentile(drawdowns, 95)), 4),
            "p99": round(float(np.percentile(drawdowns, 99)), 4),
        },
        "failure_probability": round(float(np.mean(final_equities < initial_equity * 0.5)), 4),
        "negative_return_probability": round(float(np.mean(returns < 0)), 4),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def run_from_experiences(days: int = 30, simulations: int = 10000) -> dict:
    """Run full Monte Carlo failure analysis."""
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
    
    # Extract exits with PnL
    exits = [e for e in recent if e.get("type") == "exit" and e.get("pnl_pct") is not None]
    
    if not exits:
        return {"error": "No completed trades"}
    
    # Run simulations
    mc_results = run_monte_carlo(
        exits,
        n_simulations=simulations,
        n_trades_per_sim=min(len(exits) * 10, 252),
    )
    
    perturbation_results = run_parameter_perturbation(
        exits,
        n_simulations=simulations // 2,
        n_trades_per_sim=min(len(exits) * 10, 252),
    )
    
    return {
        "period_days": days,
        "total_trades": len(exits),
        "base_metrics": {
            "win_rate": round(float(np.mean([e.get("pnl_pct", 0) > 0 for e in exits])), 4),
            "avg_pnl": round(float(np.mean([e.get("pnl_pct", 0) for e in exits])), 6),
            "total_pnl": round(float(sum([e.get("pnl_pct", 0) for e in exits])), 4),
        },
        "monte_carlo": mc_results,
        "parameter_perturbation": perturbation_results,
    }


def print_report(results: dict):
    """Print Monte Carlo failure analysis report."""
    print("\n" + "=" * 80)
    print("MONTE CARLO FAILURE ANALYSIS")
    print("=" * 80)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    print(f"Period: Last {results.get('period_days', 0)} days")
    print(f"Trades Analyzed: {results.get('total_trades', 0)}")
    
    # Base metrics
    base = results.get("base_metrics", {})
    print(f"Base Win Rate: {base.get('win_rate', 0):.2%}")
    print(f"Base Avg PnL: {base.get('avg_pnl', 0):.4%}")
    
    # Monte Carlo results
    mc = results.get("monte_carlo", {})
    if "error" not in mc:
        print("\n" + "-" * 80)
        print("MONTE CARLO SIMULATION RESULTS")
        print("-" * 80)
        print(f"Simulations: {mc.get('simulations', 0):,}")
        print(f"Trades Per Sim: {mc.get('trades_per_sim', 0)}")
        
        print(f"\n{'Metric':<30} {'Value':>15} {'Interpretation':<25}")
        print("-" * 70)
        
        metrics = [
            ("Probability of Losing Money", f"{mc.get('probability_of_losing_money', 0):.2%}",
             "HIGH" if mc.get('probability_of_losing_money', 0) > 0.3 else "MODERATE" if mc.get('probability_of_losing_money', 0) > 0.15 else "LOW"),
            ("Probability of Ruin", f"{mc.get('probability_of_ruin', 0):.4%}",
             "CRITICAL" if mc.get('probability_of_ruin', 0) > 0.01 else "ACCEPTABLE"),
            ("Expected Max Drawdown", f"{mc.get('expected_max_drawdown', 0):.2%}",
             "SEVERE" if mc.get('expected_max_drawdown', 0) > 0.2 else "MODERATE" if mc.get('expected_max_drawdown', 0) > 0.1 else "MILD"),
            ("Worst Case Drawdown", f"{mc.get('worst_case_drawdown', 0):.2%}",
             "CATASTROPHIC" if mc.get('worst_case_drawdown', 0) > 0.5 else "SEVERE" if mc.get('worst_case_drawdown', 0) > 0.3 else "MODERATE"),
            ("Expected Losing Streak", f"{mc.get('expected_losing_streak', 0):.0f} trades",
             "LONG" if mc.get('expected_losing_streak', 0) > 10 else "MODERATE" if mc.get('expected_losing_streak', 0) > 5 else "SHORT"),
            ("Strategy Failure (50% loss)", f"{mc.get('strategy_failure_probability', 0):.2%}",
             "HIGH RISK" if mc.get('strategy_failure_probability', 0) > 0.1 else "MODERATE" if mc.get('strategy_failure_probability', 0) > 0.05 else "LOW RISK"),
        ]
        
        for name, value, interp in metrics:
            print(f"{name:<30} {value:>15} {interp:<25}")
        
        # Equity distribution
        eq = mc.get("final_equity", {})
        print(f"\nFinal Equity Distribution:")
        print(f"  Mean: ${eq.get('mean', 0):,.2f}")
        print(f"  Median: ${eq.get('median', 0):,.2f}")
        print(f"  5th Percentile: ${eq.get('p5', 0):,.2f}")
        print(f"  95th Percentile: ${eq.get('p95', 0):,.2f}")
        
        # Drawdown distribution
        dd = mc.get("drawdown", {})
        print(f"\nDrawdown Distribution:")
        print(f"  Mean: {dd.get('mean', 0):.2%}")
        print(f"  95th Percentile: {dd.get('p95', 0):.2%}")
        print(f"  99th Percentile: {dd.get('p99', 0):.2%}")
    
    # Parameter perturbation
    pert = results.get("parameter_perturbation", {})
    if "error" not in pert:
        print("\n" + "-" * 80)
        print("PARAMETER PERTURBATION RESULTS")
        print("-" * 80)
        print(f"Simulations: {pert.get('simulations', 0):,}")
        
        print(f"\nFailure Probability: {pert.get('failure_probability', 0):.2%}")
        print(f"Negative Return Probability: {pert.get('negative_return_probability', 0):.2%}")
        
        eq = pert.get("final_equity", {})
        print(f"\nEquity (with perturbation):")
        print(f"  Mean: ${eq.get('mean', 0):,.2f}")
        print(f"  5th Percentile: ${eq.get('p5', 0):,.2f}")
        print(f"  95th Percentile: ${eq.get('p95', 0):,.2f}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo failure analysis")
    parser.add_argument("--days", type=int, default=30, help="Analysis window (days)")
    parser.add_argument("--simulations", type=int, default=10000, help="Number of simulations")
    args = parser.parse_args()
    
    results = run_from_experiences(days=args.days, simulations=args.simulations)
    print_report(results)


if __name__ == "__main__":
    main()
