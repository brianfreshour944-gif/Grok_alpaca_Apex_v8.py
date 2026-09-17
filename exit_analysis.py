"""
exit_analysis.py — Stop-loss and exit logic analysis.

Analyzes whether exits are:
- Too tight (normal noise causes unnecessary losses)
- Too wide (losses become unnecessarily large)

Answers:
- How far did price move against before recovering?
- How many stopped-out trades would have become winners?
- How much additional profit was available after exit?
- Is the bot exiting too early?

Usage:
    python exit_analysis.py                     # full exit analysis
    python exit_analysis.py --days 30           # longer window
    python exit_analysis.py --stop-test         # test stop levels
"""

import argparse
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from config import (
    logger, EXPERIENCE_LOG_PATH,
    PROFIT_TARGET_PCT, STOP_LOSS_PCT, MAX_HOLD_HOURS,
    MIN_HOLD_HOURS_BEFORE_SIGNAL, SELL_SIGNAL,
    TRAILING_STOP_ATR_MULTIPLIER, MIN_TRAILING_STOP_PCT, MAX_TRAILING_STOP_PCT,
)
from experience_capture import load_experiences


# ── Exit Type Analysis ──────────────────────────────────────────────────────

def classify_exit_type(event: dict) -> str:
    """
    Classify the exit type based on event data.
    """
    exit_reason = event.get("exit_reason", "")
    
    if "Trailing Stop" in exit_reason:
        return "trailing_stop"
    elif "Stop loss" in exit_reason or "Time-Decay" in exit_reason:
        return "stop_loss"
    elif "Max hold" in exit_reason:
        return "max_hold"
    elif "Signal weak" in exit_reason:
        return "signal_weakness"
    else:
        return "other"


# ── Stopped-Out Trade Analysis ──────────────────────────────────────────────

def analyze_stopped_trades(events: list[dict]) -> dict:
    """
    Analyze trades that were stopped out.
    Key question: Would they have recovered if held longer?
    """
    exits = [e for e in events if e.get("type") == "exit"]
    
    stopped_trades = []
    for ev in exits:
        exit_type = classify_exit_type(ev)
        if exit_type == "stop_loss":
            pnl_pct = ev.get("pnl_pct", 0)
            held_hours = ev.get("held_hours", 0)
            symbol = ev.get("symbol", "")
            
            # Find the entry to get initial signal
            entry_signal = None
            for entry in events:
                if (entry.get("type") == "entry" and 
                    entry.get("symbol") == symbol and 
                    entry.get("ts", "") < ev.get("ts", "")):
                    entry_signal = entry.get("signal", 0.5)
                    break
            
            stopped_trades.append({
                "symbol": symbol,
                "pnl_pct": pnl_pct,
                "held_hours": held_hours,
                "entry_signal": entry_signal,
                "exit_reason": ev.get("exit_reason", ""),
                "ts": ev.get("ts", ""),
            })
    
    if not stopped_trades:
        return {"error": "No stopped-out trades"}
    
    pnls = [t["pnl_pct"] for t in stopped_trades]
    
    return {
        "n_stopped": len(stopped_trades),
        "total_pnl": round(float(sum(pnls)), 4),
        "avg_pnl": round(float(np.mean(pnls)), 6),
        "worst_stop": round(float(min(pnls)), 4),
        "best_stop": round(float(max(pnls)), 4),
        "avg_held_hours": round(float(np.mean([t["held_hours"] for t in stopped_trades])), 2),
        "trades": stopped_trades[:20],  # First 20 for inspection
    }


# ── Unrealized Opportunity Analysis ─────────────────────────────────────────

def analyze_unrealized_opportunity(events: list[dict]) -> dict:
    """
    Analyze winning trades that exited.
    Key question: How much additional profit was available after exit?
    """
    exits = [e for e in events if e.get("type") == "exit"]
    
    winning_exits = []
    for ev in exits:
        if ev.get("pnl_pct", 0) > 0:
            winning_exits.append({
                "symbol": ev.get("symbol", ""),
                "pnl_pct": ev.get("pnl_pct", 0),
                "held_hours": ev.get("held_hours", 0),
                "exit_reason": ev.get("exit_reason", ""),
                "highest_seen": ev.get("highest_seen", 0),
                "exit_price": ev.get("exit_price", 0),
                "ts": ev.get("ts", ""),
            })
    
    if not winning_exits:
        return {"error": "No winning exits"}
    
    pnls = [e["pnl_pct"] for e in winning_exits]
    
    # Estimate "would have been" by looking at highest_seen vs exit
    # If trailing stop exited at X, but highest_seen was Y, the difference
    # is profit given back
    profit_given_back = []
    for e in winning_exits:
        highest = e.get("highest_seen", 0)
        exit_price = e.get("exit_price", 0)
        if highest > 0 and exit_price > 0:
            given_back = (highest - exit_price) / highest
            profit_given_back.append(given_back)
    
    return {
        "n_winning_exits": len(winning_exits),
        "avg_pnl": round(float(np.mean(pnls)), 6),
        "total_pnl": round(float(sum(pnls)), 4),
        "avg_profit_given_back": round(float(np.mean(profit_given_back)), 6) if profit_given_back else 0,
        "total_profit_given_back": round(float(sum(profit_given_back)), 4) if profit_given_back else 0,
        "trades": winning_exits[:20],
    }


# ── Price Movement Analysis ─────────────────────────────────────────────────

def analyze_price_movement(events: list[dict]) -> dict:
    """
    Analyze how far price moved against positions before recovering or stopping.
    """
    entries = [e for e in events if e.get("type") == "entry"]
    exits = [e for e in events if e.get("type") == "exit"]
    
    if not entries or not exits:
        return {"error": "Insufficient data"}
    
    # Match entries to exits
    trade_pairs = []
    for entry in entries:
        symbol = entry.get("symbol", "")
        entry_ts = entry.get("ts", "")
        entry_price = entry.get("price", 0)
        
        for exit_ev in exits:
            if (exit_ev.get("symbol") == symbol and 
                exit_ev.get("ts", "") > entry_ts):
                
                exit_price = exit_ev.get("exit_price", 0)
                highest = exit_ev.get("highest_seen", entry_price)
                lowest = exit_ev.get("lowest_seen", entry_price)
                
                if entry_price > 0:
                    # Maximum adverse excursion (MAE)
                    max_adverse = (lowest - entry_price) / entry_price if lowest > 0 else 0
                    
                    # Maximum favorable excursion (MFE)
                    max_favorable = (highest - entry_price) / entry_price if highest > 0 else 0
                    
                    trade_pairs.append({
                        "symbol": symbol,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "highest": highest,
                        "lowest": lowest,
                        "mae": max_adverse,  # How far against
                        "mfe": max_favorable,  # How far in favor
                        "pnl_pct": exit_ev.get("pnl_pct", 0),
                        "exit_type": classify_exit_type(exit_ev),
                        "held_hours": exit_ev.get("held_hours", 0),
                    })
                break
    
    if not trade_pairs:
        return {"error": "No matched trade pairs"}
    
    # Analyze by exit type
    by_exit_type = {}
    for trade in trade_pairs:
        exit_type = trade["exit_type"]
        if exit_type not in by_exit_type:
            by_exit_type[exit_type] = []
        by_exit_type[exit_type].append(trade)
    
    exit_type_metrics = {}
    for exit_type, trades in by_exit_type.items():
        maes = [t["mae"] for t in trades]
        mfes = [t["mfe"] for t in trades]
        pnls = [t["pnl_pct"] for t in trades]
        
        exit_type_metrics[exit_type] = {
            "n_trades": len(trades),
            "avg_mae": round(float(np.mean(maes)), 6),
            "max_mae": round(float(min(maes)), 6),
            "avg_mfe": round(float(np.mean(mfes)), 6),
            "max_mfe": round(float(max(mfes)), 6),
            "avg_pnl": round(float(np.mean(pnls)), 6),
            "mae_to_pnl_ratio": round(float(abs(np.mean(maes)) / abs(np.mean(pnls))), 2) if np.mean(pnls) != 0 else float("inf"),
        }
    
    # Overall metrics
    all_maes = [t["mae"] for t in trade_pairs]
    all_mfes = [t["mfe"] for t in trade_pairs]
    
    return {
        "n_trades": len(trade_pairs),
        "overall": {
            "avg_mae": round(float(np.mean(all_maes)), 6),
            "max_mae": round(float(min(all_maes)), 6),
            "avg_mfe": round(float(np.mean(all_mfes)), 6),
            "max_mfe": round(float(max(all_mfes)), 6),
        },
        "by_exit_type": exit_type_metrics,
        "trades": trade_pairs[:20],
    }


# ── Stop Level Testing ──────────────────────────────────────────────────────

def test_stop_levels(events: list[dict], stop_levels: list[float] = None) -> list[dict]:
    """
    Test performance at different stop loss levels.
    """
    if stop_levels is None:
        stop_levels = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
    
    # Extract trade data
    entries = {}
    exits = {}
    
    for ev in events:
        if ev.get("type") == "entry":
            key = (ev.get("symbol", ""), ev.get("ts", ""))
            entries[key] = {
                "symbol": ev.get("symbol", ""),
                "price": ev.get("price", 0),
                "ts": ev.get("ts", ""),
                "signal": ev.get("signal", 0.5),
            }
        elif ev.get("type") == "exit":
            key = (ev.get("symbol", ""), ev.get("ts", ""))
            exits[key] = {
                "pnl_pct": ev.get("pnl_pct", 0),
                "highest_seen": ev.get("highest_seen", 0),
                "lowest_seen": ev.get("lowest_seen", 0),
                "held_hours": ev.get("held_hours", 0),
                "symbol": ev.get("symbol", ""),
                "ts": ev.get("ts", ""),
            }
    
    # Match entries to exits
    matched = []
    for (sym, entry_ts), entry in entries.items():
        for (sym2, exit_ts), exit_ev in exits.items():
            if sym == sym2 and exit_ts > entry_ts:
                matched.append({
                    "entry_price": entry["price"],
                    "exit_price": exit_ev.get("pnl_pct", 0) * entry["price"] + entry["price"],
                    "highest": exit_ev.get("highest_seen", entry["price"]),
                    "lowest": exit_ev.get("lowest_seen", entry["price"]),
                    "signal": entry["signal"],
                    "pnl_pct": exit_ev["pnl_pct"],
                    "held_hours": exit_ev.get("held_hours", 0),
                })
                break
    
    if not matched:
        return [{"error": "No matched trades"}]
    
    # Test each stop level
    results = []
    baseline_win_rate = np.mean([1 if m["pnl_pct"] > 0 else 0 for m in matched])
    
    for stop_pct in stop_levels:
        # Simulate trades with this stop level
        simulated = []
        for trade in matched:
            entry_price = trade["entry_price"]
            lowest = trade["lowest"]
            
            # Check if stop would have triggered
            stop_price = entry_price * (1 - stop_pct)
            would_stop = lowest <= stop_price
            
            if would_stop:
                # Stopped out at stop level (with slippage)
                pnl = -stop_pct - 0.001  # 0.1% slippage
            else:
                # Held to original exit
                pnl = trade["pnl_pct"]
            
            simulated.append({
                "pnl": pnl,
                "would_stop": would_stop,
                "original_pnl": trade["pnl_pct"],
            })
        
        outcomes = [1 if s["pnl"] > 0 else 0 for s in simulated]
        pnls = [s["pnl"] for s in simulated]
        
        win_rate = np.mean(outcomes)
        mean_pnl = np.mean(pnls)
        total_pnl = sum(pnls)
        
        # How many trades would have been stopped that were winners?
        stopped_that_were_winners = sum(
            1 for s in simulated 
            if s["would_stop"] and s["original_pnl"] > 0
        )
        
        # How many trades would have been stopped that were losers anyway?
        stopped_that_were_losers = sum(
            1 for s in simulated 
            if s["would_stop"] and s["original_pnl"] <= 0
        )
        
        results.append({
            "stop_pct": stop_pct,
            "n_trades": len(simulated),
            "n_stopped": sum(1 for s in simulated if s["would_stop"]),
            "pct_stopped": round(sum(1 for s in simulated if s["would_stop"]) / len(simulated) * 100, 1),
            "win_rate": round(float(win_rate), 4),
            "win_rate_vs_baseline": round(float(win_rate - baseline_win_rate), 4),
            "mean_pnl": round(float(mean_pnl), 6),
            "total_pnl": round(float(total_pnl), 4),
            "stopped_winners": stopped_that_were_winners,
            "stopped_losers": stopped_that_were_losers,
            "stop_efficiency": round(float(stopped_that_were_losers / max(stopped_that_were_winners + stopped_that_were_losers, 1)), 4),
        })
    
    return results


# ── Main Analysis ────────────────────────────────────────────────────────────

def analyze_from_experiences(days: int = 7) -> dict:
    """Run full exit analysis from experience log."""
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
    stopped_trades = analyze_stopped_trades(recent)
    unrealized = analyze_unrealized_opportunity(recent)
    price_movement = analyze_price_movement(recent)
    stop_tests = test_stop_levels(recent)
    
    # Exit type breakdown
    exits = [e for e in recent if e.get("type") == "exit"]
    exit_types = {}
    for ev in exits:
        et = classify_exit_type(ev)
        if et not in exit_types:
            exit_types[et] = 0
        exit_types[et] += 1
    
    return {
        "period_days": days,
        "exit_type_breakdown": exit_types,
        "stopped_trades": stopped_trades,
        "unrealized_opportunity": unrealized,
        "price_movement": price_movement,
        "stop_level_tests": stop_tests,
    }


def print_report(results: dict):
    """Print exit analysis report."""
    print("\n" + "=" * 80)
    print("STOP-LOSS & EXIT LOGIC ANALYSIS")
    print("=" * 80)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    print(f"Period: Last {results['period_days']} days")
    
    # ── Exit Type Breakdown ──────────────────────────────────────────────
    exit_types = results.get("exit_type_breakdown", {})
    if exit_types:
        print("\n" + "-" * 80)
        print("EXIT TYPE BREAKDOWN")
        print("-" * 80)
        total = sum(exit_types.values())
        for et, count in sorted(exit_types.items(), key=lambda x: -x[1]):
            pct = count / total * 100 if total > 0 else 0
            print(f"  {et:<20} {count:>5} ({pct:.1f}%)")
    
    # ── Stopped Trades ───────────────────────────────────────────────────
    stopped = results.get("stopped_trades", {})
    if "error" not in stopped:
        print("\n" + "-" * 80)
        print("STOPPED-OUT TRADES")
        print("-" * 80)
        print(f"  Trades Stopped: {stopped.get('n_stopped', 0)}")
        print(f"  Total PnL: {stopped.get('total_pnl', 0):.4%}")
        print(f"  Avg PnL: {stopped.get('avg_pnl', 0):.4%}")
        print(f"  Worst Stop: {stopped.get('worst_stop', 0):.4%}")
        print(f"  Best Stop: {stopped.get('best_stop', 0):.4%}")
        print(f"  Avg Held: {stopped.get('avg_held_hours', 0):.2f}h")
    
    # ── Unrealized Opportunity ───────────────────────────────────────────
    unrealized = results.get("unrealized_opportunity", {})
    if "error" not in unrealized:
        print("\n" + "-" * 80)
        print("UNREALIZED OPPORTUNITY (WINNING TRADES)")
        print("-" * 80)
        print(f"  Winning Exits: {unrealized.get('n_winning_exits', 0)}")
        print(f"  Avg PnL: {unrealized.get('avg_pnl', 0):.4%}")
        print(f"  Total PnL: {unrealized.get('total_pnl', 0):.4%}")
        print(f"  Avg Profit Given Back: {unrealized.get('avg_profit_given_back', 0):.4%}")
        print(f"  Total Profit Given Back: {unrealized.get('total_profit_given_back', 0):.4%}")
    
    # ── Price Movement ───────────────────────────────────────────────────
    price = results.get("price_movement", {})
    if "error" not in price:
        print("\n" + "-" * 80)
        print("PRICE MOVEMENT ANALYSIS")
        print("-" * 80)
        
        overall = price.get("overall", {})
        print(f"  Avg Max Adverse Excursion (MAE): {overall.get('avg_mae', 0):.4%}")
        print(f"  Worst MAE: {overall.get('max_mae', 0):.4%}")
        print(f"  Avg Max Favorable Excursion (MFE): {overall.get('avg_mfe', 0):.4%}")
        print(f"  Best MFE: {overall.get('max_mfe', 0):.4%}")
        
        by_exit = price.get("by_exit_type", {})
        if by_exit:
            print("\n  By Exit Type:")
            for exit_type, metrics in by_exit.items():
                print(f"    {exit_type}:")
                print(f"      MAE: {metrics['avg_mae']:.4%} | MFE: {metrics['avg_mfe']:.4%} | "
                      f"PnL: {metrics['avg_pnl']:.4%} | MAE/PnL: {metrics['mae_to_pnl_ratio']:.2f}")
    
    # ── Stop Level Tests ─────────────────────────────────────────────────
    stop_tests = results.get("stop_level_tests", [])
    if stop_tests and "error" not in stop_tests[0]:
        print("\n" + "-" * 80)
        print("STOP LEVEL SENSITIVITY")
        print("-" * 80)
        print(f"{'Stop%':>6} {'Stopped':>8} {'WinRate':>8} {'ΔBase':>8} {'MeanPnL':>10} {'TotalPnL':>10} {'Efficiency':>10}")
        print("-" * 70)
        
        baseline = None
        for t in stop_tests:
            if baseline is None:
                baseline = t.get("win_rate", 0)
            
            print(f"{t['stop_pct']:>6.1%} {t['pct_stopped']:>7.1f}% {t['win_rate']:>8.2%} "
                  f"{t['win_rate_vs_baseline']:>+8.2%} {t['mean_pnl']:>10.4%} "
                  f"{t['total_pnl']:>10.4f} {t['stop_efficiency']:>10.2%}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Exit logic analysis")
    parser.add_argument("--days", type=int, default=7, help="Analysis window (days)")
    parser.add_argument("--stop-test", action="store_true", help="Run stop level tests")
    args = parser.parse_args()
    
    results = analyze_from_experiences(days=args.days)
    print_report(results)


if __name__ == "__main__":
    main()
