"""
performance_dashboard.py — Unified performance metrics dashboard.

Tracks all key metrics in one view:
- Returns
- Risk
- Risk-Adjusted
- Trade Quality
- Trading Efficiency
- Robustness

Usage:
    python performance_dashboard.py              # full dashboard
    python performance_dashboard.py --days 7     # specific period
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


# ── Metrics Computation ──────────────────────────────────────────────────────

def compute_all_metrics(events: list[dict], initial_equity: float = 10000) -> dict:
    """
    Compute comprehensive performance metrics.
    """
    entries = [e for e in events if e.get("type") == "entry"]
    exits = [e for e in events if e.get("type") == "exit"]
    
    if not entries or not exits:
        return {"error": "Insufficient data"}
    
    # Match trades
    trades = []
    for entry in entries:
        symbol = entry.get("symbol", "")
        entry_ts = entry.get("ts", "")
        entry_price = entry.get("price", 0)
        signal = entry.get("signal", 0.5)
        regime = entry.get("regime", "unknown")
        atr_pct = entry.get("atr_pct", 0)
        
        for exit_ev in exits:
            if exit_ev.get("symbol") == symbol and exit_ev.get("ts", "") > entry_ts:
                pnl_pct = exit_ev.get("pnl_pct", 0)
                held_hours = exit_ev.get("held_hours", 0)
                exit_reason = exit_ev.get("exit_reason", "")
                
                trades.append({
                    "symbol": symbol,
                    "entry_price": entry_price,
                    "signal": signal,
                    "regime": regime,
                    "atr_pct": atr_pct,
                    "pnl_pct": pnl_pct,
                    "held_hours": held_hours,
                    "exit_reason": exit_reason,
                    "entry_ts": entry_ts,
                    "exit_ts": exit_ev.get("ts", ""),
                })
                break
    
    if not trades:
        return {"error": "No matched trades"}
    
    pnls = np.array([t["pnl_pct"] for t in trades])
    held_hours = np.array([t["held_hours"] for t in trades])
    
    # ── Returns ──────────────────────────────────────────────────────────
    total_return = float(np.sum(pnls))
    avg_trade = float(np.mean(pnls))
    
    # Annualize (assuming 252 trading days, avg 4 trades/day)
    trades_per_day = len(trades) / max(1, (datetime.fromisoformat(trades[-1]["exit_ts"].replace("Z", "+00:00")) - 
                                           datetime.fromisoformat(trades[0]["entry_ts"].replace("Z", "+00:00"))).days)
    annualized_return = total_return * 365 / max(1, len(trades) / max(1, trades_per_day))
    
    # Expectancy
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    # ── Risk ─────────────────────────────────────────────────────────────
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    max_drawdown = float(np.min(drawdowns))
    
    # Drawdown duration
    in_drawdown = drawdowns < 0
    dd_durations = []
    current_dd_start = None
    for i, is_dd in enumerate(in_drawdown):
        if is_dd and current_dd_start is None:
            current_dd_start = i
        elif not is_dd and current_dd_start is not None:
            dd_durations.append(i - current_dd_start)
            current_dd_start = None
    if current_dd_start is not None:
        dd_durations.append(len(in_drawdown) - current_dd_start)
    max_dd_duration = max(dd_durations) if dd_durations else 0
    
    # Downside deviation
    downside_returns = pnls[pnls < 0]
    downside_deviation = float(np.std(downside_returns)) if len(downside_returns) > 1 else 0
    
    # VaR and Expected Shortfall
    var_95 = float(np.percentile(pnls, 5))
    cvar_95 = float(np.mean(pnls[pnls <= var_95])) if np.any(pnls <= var_95) else var_95
    
    # ── Risk-Adjusted ────────────────────────────────────────────────────
    sharpe = float(np.mean(pnls) / (np.std(pnls) + 1e-10)) if len(pnls) > 1 else 0
    
    # Sortino (using downside deviation)
    sortino = float(np.mean(pnls) / (downside_deviation + 1e-10)) if downside_deviation > 0 else 0
    
    # Calmar (return / max drawdown)
    calmar = float(abs(total_return / max_drawdown)) if max_drawdown != 0 else 0
    
    # ── Trade Quality ────────────────────────────────────────────────────
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    
    # Max consecutive losses
    max_consecutive_losses = 0
    current_streak = 0
    for pnl in pnls:
        if pnl < 0:
            current_streak += 1
            max_consecutive_losses = max(max_consecutive_losses, current_streak)
        else:
            current_streak = 0
    
    # ── Trading Efficiency ───────────────────────────────────────────────
    total_days = max(1, (datetime.fromisoformat(trades[-1]["exit_ts"].replace("Z", "+00:00")) - 
                        datetime.fromisoformat(trades[0]["entry_ts"].replace("Z", "+00:00"))).days)
    trades_per_day = len(trades) / total_days
    avg_holding_time = float(np.mean(held_hours))
    
    # Fees and slippage (estimated)
    fee_per_trade = 0.0025  # 0.25% taker fee
    slippage_per_trade = 0.0005  # 0.05% avg slippage
    total_fees = fee_per_trade * len(trades)
    total_slippage = slippage_per_trade * len(trades)
    
    # ── Regime Stability ─────────────────────────────────────────────────
    regime_performance = {}
    for t in trades:
        regime = t.get("regime", "unknown")
        if regime not in regime_performance:
            regime_performance[regime] = []
        regime_performance[regime].append(t["pnl_pct"])
    
    regime_stability = {}
    for regime, regime_pnls in regime_performance.items():
        regime_arr = np.array(regime_pnls)
        regime_stability[regime] = {
            "trades": len(regime_arr),
            "win_rate": round(float(np.mean(regime_arr > 0)), 4),
            "avg_pnl": round(float(np.mean(regime_arr)), 6),
            "sharpe": round(float(np.mean(regime_arr) / (np.std(regime_arr) + 1e-10)), 4),
        }
    
    # ── Exit Analysis ────────────────────────────────────────────────────
    exit_reasons = {}
    for t in trades:
        reason = t.get("exit_reason", "unknown")
        if "Trailing Stop" in reason:
            reason = "trailing_stop"
        elif "Stop loss" in reason or "Time-Decay" in reason:
            reason = "stop_loss"
        elif "Max hold" in reason:
            reason = "max_hold"
        elif "Signal weak" in reason:
            reason = "signal_weakness"
        
        if reason not in exit_reasons:
            exit_reasons[reason] = {"count": 0, "total_pnl": 0}
        exit_reasons[reason]["count"] += 1
        exit_reasons[reason]["total_pnl"] += t["pnl_pct"]
    
    return {
        "n_trades": len(trades),
        "total_days": total_days,
        "returns": {
            "total_return": round(total_return, 4),
            "annualized_return": round(annualized_return, 4),
            "avg_trade": round(avg_trade, 6),
            "expectancy": round(expectancy, 6),
        },
        "risk": {
            "max_drawdown": round(max_drawdown, 4),
            "max_dd_duration": max_dd_duration,
            "downside_deviation": round(downside_deviation, 6),
            "var_95": round(var_95, 4),
            "cvar_95": round(cvar_95, 4),
        },
        "risk_adjusted": {
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
            "calmar": round(calmar, 4),
        },
        "trade_quality": {
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 6),
            "avg_loss": round(avg_loss, 6),
            "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else "inf",
            "payoff_ratio": round(payoff_ratio, 4) if payoff_ratio != float("inf") else "inf",
            "max_consecutive_losses": max_consecutive_losses,
        },
        "trading_efficiency": {
            "trades_per_day": round(trades_per_day, 2),
            "avg_holding_time_hours": round(avg_holding_time, 2),
            "fee_per_trade_pct": round(fee_per_trade * 100, 2),
            "slippage_per_trade_pct": round(slippage_per_trade * 100, 2),
            "total_fees_pct": round(total_fees * 100, 2),
            "total_slippage_pct": round(total_slippage * 100, 2),
        },
        "regime_stability": regime_stability,
        "exit_analysis": exit_reasons,
    }


# ── Dashboard Display ────────────────────────────────────────────────────────

def print_dashboard(results: dict):
    """Print unified performance dashboard."""
    print("\n" + "=" * 80)
    print("PERFORMANCE DASHBOARD")
    print("=" * 80)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    print(f"Period: {results.get('total_days', 0)} days | {results.get('n_trades', 0)} trades")
    
    # ── Returns ──────────────────────────────────────────────────────────
    ret = results.get("returns", {})
    print("\n" + "-" * 80)
    print("RETURNS")
    print("-" * 80)
    print(f"  Total Return:     {ret.get('total_return', 0):>10.2%}")
    print(f"  Annualized:       {ret.get('annualized_return', 0):>10.2%}")
    print(f"  Average Trade:    {ret.get('avg_trade', 0):>10.4%}")
    print(f"  Expectancy:       {ret.get('expectancy', 0):>10.4%}")
    
    # ── Risk ─────────────────────────────────────────────────────────────
    risk = results.get("risk", {})
    print("\n" + "-" * 80)
    print("RISK")
    print("-" * 80)
    print(f"  Max Drawdown:     {risk.get('max_drawdown', 0):>10.2%}")
    print(f"  DD Duration:      {risk.get('max_dd_duration', 0):>10} periods")
    print(f"  Downside Dev:     {risk.get('downside_deviation', 0):>10.4%}")
    print(f"  VaR (95%):        {risk.get('var_95', 0):>10.4%}")
    print(f"  CVaR (95%):       {risk.get('cvar_95', 0):>10.4%}")
    
    # ── Risk-Adjusted ────────────────────────────────────────────────────
    ra = results.get("risk_adjusted", {})
    print("\n" + "-" * 80)
    print("RISK-ADJUSTED")
    print("-" * 80)
    print(f"  Sharpe:           {ra.get('sharpe', 0):>10.4f}")
    print(f"  Sortino:          {ra.get('sortino', 0):>10.4f}")
    print(f"  Calmar:           {ra.get('calmar', 0):>10.4f}")
    
    # ── Trade Quality ────────────────────────────────────────────────────
    tq = results.get("trade_quality", {})
    print("\n" + "-" * 80)
    print("TRADE QUALITY")
    print("-" * 80)
    print(f"  Win Rate:         {tq.get('win_rate', 0):>10.2%}")
    print(f"  Avg Winner:       {tq.get('avg_win', 0):>10.4%}")
    print(f"  Avg Loser:        {tq.get('avg_loss', 0):>10.4%}")
    print(f"  Profit Factor:    {tq.get('profit_factor', 0):>10}")
    print(f"  Payoff Ratio:     {tq.get('payoff_ratio', 0):>10}")
    print(f"  Max Consec Loss:  {tq.get('max_consecutive_losses', 0):>10}")
    
    # ── Trading Efficiency ───────────────────────────────────────────────
    te = results.get("trading_efficiency", {})
    print("\n" + "-" * 80)
    print("TRADING EFFICIENCY")
    print("-" * 80)
    print(f"  Trades/Day:       {te.get('trades_per_day', 0):>10.2f}")
    print(f"  Avg Hold Time:    {te.get('avg_holding_time_hours', 0):>10.2f}h")
    print(f"  Fees/Trade:       {te.get('fee_per_trade_pct', 0):>10.2f}%")
    print(f"  Slippage/Trade:   {te.get('slippage_per_trade_pct', 0):>10.2f}%")
    
    # ── Regime Stability ─────────────────────────────────────────────────
    regime = results.get("regime_stability", {})
    if regime:
        print("\n" + "-" * 80)
        print("REGIME STABILITY")
        print("-" * 80)
        print(f"  {'Regime':<12} {'Trades':>7} {'WinRate':>8} {'AvgPnL':>10} {'Sharpe':>8}")
        print("  " + "-" * 50)
        for regime_name, stats in sorted(regime.items()):
            print(f"  {regime_name:<12} {stats['trades']:>7} {stats['win_rate']:>8.2%} "
                  f"{stats['avg_pnl']:>10.4%} {stats['sharpe']:>8.2f}")
    
    # ── Exit Analysis ────────────────────────────────────────────────────
    exits = results.get("exit_analysis", {})
    if exits:
        print("\n" + "-" * 80)
        print("EXIT ANALYSIS")
        print("-" * 80)
        print(f"  {'Reason':<20} {'Count':>7} {'Total PnL':>10}")
        print("  " + "-" * 40)
        for reason, stats in sorted(exits.items(), key=lambda x: -x[1]["count"]):
            print(f"  {reason:<20} {stats['count']:>7} {stats['total_pnl']:>10.4%}")
    
    print("=" * 80)


# ── Main ─────────────────────────────────────────────────────────────────────

def analyze_from_experiences(days: int = 30) -> dict:
    """Run full performance analysis."""
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
    
    return compute_all_metrics(recent)


def main():
    parser = argparse.ArgumentParser(description="Performance dashboard")
    parser.add_argument("--days", type=int, default=30, help="Analysis window (days)")
    args = parser.parse_args()
    
    results = analyze_from_experiences(days=args.days)
    print_dashboard(results)


if __name__ == "__main__":
    main()
