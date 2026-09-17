"""
execution_cost_analysis.py — Realistic execution cost analysis.

Measures actual trading costs:
- Exchange fees
- Bid/ask spread
- Slippage
- Market impact
- Latency
- Partial fills
- Rejected orders
- Limit-order fill rate

Tests profitability under different cost scenarios.

Usage:
    python execution_cost_analysis.py                    # full analysis
    python execution_cost_analysis.py --days 30          # longer window
    python execution_cost_analysis.py --stress-test      # cost stress test
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
    BASE_RISK_PERCENT, MAX_SINGLE_TRADE_USD,
    PROFIT_TARGET_PCT, STOP_LOSS_PCT,
)
from experience_capture import load_experiences


# ── Cost Model Assumptions ───────────────────────────────────────────────────

# Realistic cost assumptions for Alpaca Crypto
COST_ASSUMPTIONS = {
    "alpaca_crypto": {
        "maker_fee": 0.0,           # 0% maker fee (limit orders)
        "taker_fee": 0.0025,        # 0.25% taker fee (market orders)
        "avg_spread_bps": 15,       # 15 basis points average spread
        "avg_slippage_bps": 5,      # 5 basis points average slippage
        "partial_fill_rate": 0.05,  # 5% of orders partially filled
        "reject_rate": 0.01,        # 1% of orders rejected
        "cancel_replace_delay_ms": 500,  # 500ms for cancel/replace
        "limit_fill_rate": 0.70,    # 70% of limit orders fill
        "latency_ms": 100,          # 100ms average latency
    },
    "aggressive": {
        "maker_fee": 0.0,
        "taker_fee": 0.004,         # 0.4% taker fee
        "avg_spread_bps": 30,       # 30 bps spread
        "avg_slippage_bps": 15,     # 15 bps slippage
        "partial_fill_rate": 0.10,
        "reject_rate": 0.02,
        "cancel_replace_delay_ms": 1000,
        "limit_fill_rate": 0.50,
        "latency_ms": 200,
    },
}


# ── Slippage Analysis ────────────────────────────────────────────────────────

def analyze_slippage(events: list[dict]) -> dict:
    """
    Analyze actual slippage from filled orders.
    Compares expected price vs actual fill price.
    """
    entries = [e for e in events if e.get("type") == "entry"]
    
    if not entries:
        return {"error": "No entries"}
    
    # Extract slippage data
    slippages = []
    for ev in entries:
        expected_price = ev.get("price", 0)
        fill_price = ev.get("fill_price", 0)
        
        if expected_price > 0 and fill_price > 0:
            slippage_pct = (fill_price - expected_price) / expected_price
            slippages.append({
                "symbol": ev.get("symbol", ""),
                "expected": expected_price,
                "fill": fill_price,
                "slippage_pct": slippage_pct,
                "side": "buy",  # Entries are buys
            })
    
    # Also check exits
    exits = [e for e in events if e.get("type") == "exit"]
    for ev in exits:
        expected_price = ev.get("exit_price", 0)
        fill_price = ev.get("fill_price", 0)
        
        if expected_price > 0 and fill_price > 0:
            slippage_pct = (fill_price - expected_price) / expected_price
            slippages.append({
                "symbol": ev.get("symbol", ""),
                "expected": expected_price,
                "fill": fill_price,
                "slippage_pct": slippage_pct,
                "side": "sell",
            })
    
    if not slippages:
        return {
            "error": "No fill price data",
            "note": "Bot may not be logging fill_price. Using estimated slippage.",
            "estimated_slippage_bps": COST_ASSUMPTIONS["alpaca_crypto"]["avg_slippage_bps"],
        }
    
    slippage_pcts = [s["slippage_pct"] for s in slippages]
    
    # Analyze by side
    buys = [s for s in slippages if s["side"] == "buy"]
    sells = [s for s in slippages if s["side"] == "sell"]
    
    return {
        "n_trades": len(slippages),
        "overall": {
            "mean_slippage_pct": round(float(np.mean(slippage_pcts)), 6),
            "median_slippage_pct": round(float(np.median(slippage_pcts)), 6),
            "std_slippage_pct": round(float(np.std(slippage_pcts)), 6),
            "worst_slippage_pct": round(float(np.max(np.abs(slippage_pcts))), 6),
        },
        "buys": {
            "count": len(buys),
            "mean_slippage_pct": round(float(np.mean([s["slippage_pct"] for s in buys])), 6) if buys else 0,
        },
        "sells": {
            "count": len(sells),
            "mean_slippage_pct": round(float(np.mean([s["slippage_pct"] for s in sells])), 6) if sells else 0,
        },
        "trades": slippages[:20],
    }


# ── Fee Analysis ─────────────────────────────────────────────────────────────

def analyze_fees(events: list[dict]) -> dict:
    """
    Analyze actual fees paid.
    """
    exits = [e for e in events if e.get("type") == "exit"]
    
    if not exits:
        return {"error": "No exits"}
    
    fees = [e.get("fee", 0) for e in exits if e.get("fee", 0) > 0]
    pnls = [e.get("pnl_pct", 0) for e in exits]
    
    if not fees:
        return {
            "error": "No fee data",
            "note": "Alpaca API does not expose fees in order objects. Using estimated fees.",
            "estimated_fee_per_trade": COST_ASSUMPTIONS["alpaca_crypto"]["taker_fee"],
        }
    
    return {
        "n_trades_with_fees": len(fees),
        "total_fees": round(float(sum(fees)), 4),
        "avg_fee": round(float(np.mean(fees)), 4),
        "fee_as_pct_of_pnl": round(float(sum(fees) / abs(sum(pnls)) * 100), 2) if sum(pnls) != 0 else 0,
    }


# ── Limit Order Fill Rate ────────────────────────────────────────────────────

def analyze_order_fills(events: list[dict]) -> dict:
    """
    Analyze order fill rates and partial fills.
    """
    # This would require order-level data
    # For now, use estimated values
    return {
        "estimated_fill_rates": {
            "market_orders": 0.99,   # 99% fill rate for market orders
            "limit_orders": 0.70,    # 70% fill rate for limit orders
            "stop_orders": 0.95,     # 95% fill rate for stop orders
        },
        "partial_fill_rate": COST_ASSUMPTIONS["alpaca_crypto"]["partial_fill_rate"],
        "reject_rate": COST_ASSUMPTIONS["alpaca_crypto"]["reject_rate"],
        "note": "Actual fill rate data requires order-level logging. Using industry estimates.",
    }


# ── Cost Scenario Testing ────────────────────────────────────────────────────

def test_cost_scenarios(events: list[dict], scenarios: dict = None) -> dict:
    """
    Test profitability under different cost scenarios.
    """
    if scenarios is None:
        scenarios = {
            "normal": {
                "name": "Normal Costs",
                "fee_pct": 0.0025,      # 0.25%
                "spread_pct": 0.0015,   # 0.15%
                "slippage_pct": 0.0005, # 0.05%
            },
            "double": {
                "name": "2× Normal Costs",
                "fee_pct": 0.005,       # 0.50%
                "spread_pct": 0.003,    # 0.30%
                "slippage_pct": 0.001,  # 0.10%
            },
            "triple": {
                "name": "3× Normal Costs",
                "fee_pct": 0.0075,      # 0.75%
                "spread_pct": 0.0045,   # 0.45%
                "slippage_pct": 0.0015, # 0.15%
            },
            "extreme": {
                "name": "Extreme Costs",
                "fee_pct": 0.01,        # 1.00%
                "spread_pct": 0.006,    # 0.60%
                "slippage_pct": 0.003,  # 0.30%
            },
        }
    
    # Extract trade data
    entries = {}
    exits = {}
    
    for ev in events:
        if ev.get("type") == "entry":
            key = (ev.get("symbol", ""), ev.get("ts", ""))
            entries[key] = {
                "price": ev.get("price", 0),
                "signal": ev.get("signal", 0.5),
            }
        elif ev.get("type") == "exit":
            key = (ev.get("symbol", ""), ev.get("ts", ""))
            exits[key] = {
                "pnl_pct": ev.get("pnl_pct", 0),
            }
    
    # Match trades
    matched = []
    for (sym, entry_ts), entry in entries.items():
        for (sym2, exit_ts), exit_ev in exits.items():
            if sym == sym2 and exit_ts > entry_ts:
                matched.append({
                    "entry_price": entry["price"],
                    "signal": entry["signal"],
                    "raw_pnl_pct": exit_ev["pnl_pct"],
                })
                break
    
    if not matched:
        return {"error": "No matched trades"}
    
    # Test each scenario
    results = {}
    for scenario_name, costs in scenarios.items():
        total_cost_pct = costs["fee_pct"] + costs["spread_pct"] + costs["slippage_pct"]
        
        # Adjust PnL for costs
        adjusted_pnls = []
        for trade in matched:
            # Round-trip cost: entry cost + exit cost
            round_trip_cost = total_cost_pct * 2
            adjusted_pnl = trade["raw_pnl_pct"] - round_trip_cost
            adjusted_pnls.append(adjusted_pnl)
        
        outcomes = [1 if p > 0 else p for p in adjusted_pnls]
        win_rate = np.mean([1 if p > 0 else 0 for p in adjusted_pnls])
        mean_pnl = np.mean(adjusted_pnls)
        total_pnl = sum(adjusted_pnls)
        
        # Sharpe-like ratio
        sharpe = np.mean(adjusted_pnls) / (np.std(adjusted_pnls) + 1e-10) if len(adjusted_pnls) > 1 else 0
        
        results[scenario_name] = {
            "name": costs["name"],
            "total_cost_pct": round(float(total_cost_pct * 100), 4),
            "round_trip_cost_pct": round(float(total_cost_pct * 2 * 100), 4),
            "win_rate": round(float(win_rate), 4),
            "mean_pnl": round(float(mean_pnl), 6),
            "total_pnl": round(float(total_pnl), 4),
            "sharpe": round(float(sharpe), 4),
            "edge_erosion": round(float(total_cost_pct * 2 * len(matched)), 4),
        }
    
    # Baseline (no costs)
    raw_pnls = [t["raw_pnl_pct"] for t in matched]
    results["baseline"] = {
        "name": "No Costs (Baseline)",
        "total_cost_pct": 0,
        "win_rate": round(float(np.mean([1 if p > 0 else 0 for p in raw_pnls])), 4),
        "mean_pnl": round(float(np.mean(raw_pnls)), 6),
        "total_pnl": round(float(sum(raw_pnls)), 4),
        "sharpe": round(float(np.mean(raw_pnls) / (np.std(raw_pnls) + 1e-10)), 4),
    }
    
    return {
        "n_trades": len(matched),
        "scenarios": results,
        "break_even_cost": round(float(np.mean(raw_pnls) / 2 * 100), 4) if np.mean(raw_pnls) > 0 else 0,
    }


# ── Order Type Analysis ─────────────────────────────────────────────────────

def analyze_order_types(events: list[dict]) -> dict:
    """
    Analyze which order types should be used when.
    """
    # Current bot uses limit orders
    # Analyze when market vs limit would be better
    
    entries = [e for e in events if e.get("type") == "entry"]
    exits = [e for e in events if e.get("type") == "exit"]
    
    # Analyze signal strength vs execution quality
    strong_signal_entries = [e for e in entries if e.get("signal", 0.5) > 0.65]
    weak_signal_entries = [e for e in entries if e.get("signal", 0.5) <= 0.65]
    
    return {
        "current_order_type": "limit",
        "recommendations": {
            "market_orders": {
                "when": "Strong signal (>0.65) + high urgency (regime change, stop loss)",
                "pros": "Guaranteed fill, faster execution",
                "cons": "Higher fees (taker), slippage",
                "use_case": "Stop losses, time-sensitive exits",
            },
            "limit_orders": {
                "when": "Normal entries, mean-reversion trades",
                "pros": "Lower fees (maker), price control",
                "cons": "May not fill, adverse selection risk",
                "use_case": "Normal entries, non-urgent exits",
            },
            "post_only": {
                "when": "Non-urgent entries in liquid markets",
                "pros": "Guaranteed maker fee, no slippage",
                "cons": "May not fill if price moves away",
                "use_case": "Accumulation, patient entries",
            },
            "stop_limit": {
                "when": "Protective stops with price control",
                "pros": "Limits slippage vs market stop",
                "cons": "May not fill in fast markets",
                "use_case": "Stop losses where slippage control matters",
            },
        },
        "signal_strength_analysis": {
            "strong_signal_count": len(strong_signal_entries),
            "weak_signal_count": len(weak_signal_entries),
            "recommendation": "Use market orders for strong signals, limit for weak",
        },
    }


# ── Main Analysis ────────────────────────────────────────────────────────────

def analyze_from_experiences(days: int = 7) -> dict:
    """Run full execution cost analysis from experience log."""
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
    slippage = analyze_slippage(recent)
    fees = analyze_fees(recent)
    fills = analyze_order_fills(recent)
    cost_scenarios = test_cost_scenarios(recent)
    order_types = analyze_order_types(recent)
    
    return {
        "period_days": days,
        "slippage": slippage,
        "fees": fees,
        "order_fills": fills,
        "cost_scenarios": cost_scenarios,
        "order_types": order_types,
    }


def print_report(results: dict):
    """Print execution cost analysis report."""
    print("\n" + "=" * 80)
    print("REALISTIC EXECUTION COST ANALYSIS")
    print("=" * 80)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    print(f"Period: Last {results['period_days']} days")
    
    # ── Slippage ─────────────────────────────────────────────────────────
    slippage = results.get("slippage", {})
    print("\n" + "-" * 80)
    print("SLIPPAGE ANALYSIS")
    print("-" * 80)
    
    if "error" not in slippage:
        overall = slippage.get("overall", {})
        print(f"  Mean Slippage: {overall.get('mean_slippage_pct', 0):.4%}")
        print(f"  Median Slippage: {overall.get('median_slippage_pct', 0):.4%}")
        print(f"  Worst Slippage: {overall.get('worst_slippage_pct', 0):.4%}")
        
        buys = slippage.get("buys", {})
        sells = slippage.get("sells", {})
        print(f"  Buys: {buys.get('count', 0)} trades, mean: {buys.get('mean_slippage_pct', 0):.4%}")
        print(f"  Sells: {sells.get('count', 0)} trades, mean: {sells.get('mean_slippage_pct', 0):.4%}")
    else:
        print(f"  {slippage.get('note', slippage.get('error', 'Unknown'))}")
        print(f"  Estimated Slippage: {slippage.get('estimated_slippage_bps', 0)} bps")
    
    # ── Fees ─────────────────────────────────────────────────────────────
    fees = results.get("fees", {})
    print("\n" + "-" * 80)
    print("FEE ANALYSIS")
    print("-" * 80)
    
    if "error" not in fees:
        print(f"  Total Fees: ${fees.get('total_fees', 0):.4f}")
        print(f"  Avg Fee Per Trade: ${fees.get('avg_fee', 0):.4f}")
        print(f"  Fees as % of PnL: {fees.get('fee_as_pct_of_pnl', 0):.2f}%")
    else:
        print(f"  {fees.get('note', fees.get('error', 'Unknown'))}")
        print(f"  Estimated Fee: {fees.get('estimated_fee_per_trade', 0):.4%}")
    
    # ── Cost Scenarios ───────────────────────────────────────────────────
    scenarios = results.get("cost_scenarios", {})
    if "error" not in scenarios:
        print("\n" + "-" * 80)
        print("COST SCENARIO ANALYSIS")
        print("-" * 80)
        print(f"  Trades Analyzed: {scenarios.get('n_trades', 0)}")
        
        sc = scenarios.get("scenarios", {})
        print(f"\n  {'Scenario':<20} {'Cost':>8} {'WinRate':>8} {'MeanPnL':>10} {'TotalPnL':>10} {'Sharpe':>8}")
        print("  " + "-" * 70)
        
        for name in ["baseline", "normal", "double", "triple", "extreme"]:
            if name in sc:
                s = sc[name]
                print(f"  {s['name']:<20} {s['total_cost_pct']:>7.2f}% {s['win_rate']:>8.2%} "
                      f"{s['mean_pnl']:>10.4%} {s['total_pnl']:>10.4f} {s['sharpe']:>8.2f}")
        
        be = scenarios.get("break_even_cost", 0)
        print(f"\n  Break-Even Cost: {be:.2f}% (costs above this eliminate edge)")
    
    # ── Order Types ──────────────────────────────────────────────────────
    order_types = results.get("order_types", {})
    print("\n" + "-" * 80)
    print("ORDER TYPE RECOMMENDATIONS")
    print("-" * 80)
    
    recs = order_types.get("recommendations", {})
    for order_type, details in recs.items():
        print(f"\n  {order_type.upper()}:")
        print(f"    When: {details['when']}")
        print(f"    Use Case: {details['use_case']}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Execution cost analysis")
    parser.add_argument("--days", type=int, default=7, help="Analysis window (days)")
    parser.add_argument("--stress-test", action="store_true", help="Run cost stress test")
    args = parser.parse_args()
    
    results = analyze_from_experiences(days=args.days)
    print_report(results)


if __name__ == "__main__":
    main()
