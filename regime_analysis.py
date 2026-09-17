"""
regime_analysis.py — Comprehensive market regime analysis.

Evaluates strategy performance across multiple regime dimensions:
- Trend (bull/bear/range)
- Volatility (low/normal/high/extreme)
- Volume (high/low)
- Structure (trending/range-bound/breakout)

Usage:
    python regime_analysis.py                    # analyze last 7 days
    python regime_analysis.py --days 30          # longer window
    python regime_analysis.py --matrix           # show regime matrix
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


# ── Regime Classification ────────────────────────────────────────────────────

def classify_extended_regime(df: pd.DataFrame) -> dict:
    """
    Classify market regime along multiple dimensions.
    Input: OHLCV DataFrame with at least 20 bars.
    Returns: dict with regime labels for each dimension.
    """
    if df is None or len(df) < 20:
        return {"error": "Insufficient data"}
    
    try:
        close = pd.to_numeric(df["close"], errors="coerce").astype(float)
        high = pd.to_numeric(df["high"], errors="coerce").astype(float)
        low = pd.to_numeric(df["low"], errors="coerce").astype(float)
        volume = pd.to_numeric(df["volume"], errors="coerce").astype(float)
        
        current_price = close.iloc[-1]
        
        # ── Trend Regime ─────────────────────────────────────────────────
        # EMA-50 for trend direction
        ema50 = close.ewm(span=50).mean().iloc[-1] if len(close) >= 50 else close.mean()
        
        # ADX approximation: |close - ema50| / atr
        atr = (high - low).rolling(14).mean().iloc[-1]
        trend_strength = abs(current_price - ema50) / (atr + 1e-10)
        
        if current_price > ema50 * 1.02:
            trend = "strong_bull"
        elif current_price > ema50:
            trend = "weak_bull"
        elif current_price < ema50 * 0.98:
            trend = "strong_bear"
        elif current_price < ema50:
            trend = "weak_bear"
        else:
            trend = "range"
        
        # ── Volatility Regime ────────────────────────────────────────────
        atr_pct = (atr / current_price) * 100 if current_price > 0 else 0
        
        if atr_pct > 6.0:
            volatility = "extreme"
        elif atr_pct > 4.0:
            volatility = "high"
        elif atr_pct > 2.0:
            volatility = "normal"
        else:
            volatility = "low"
        
        # ── Volume Regime ────────────────────────────────────────────────
        vol_sma20 = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
        vol_ratio = volume.iloc[-1] / (vol_sma20 + 1e-10)
        
        if vol_ratio > 1.5:
            volume_regime = "high"
        elif vol_ratio < 0.5:
            volume_regime = "low"
        else:
            volume_regime = "normal"
        
        # ── Market Structure ─────────────────────────────────────────────
        # Check for breakout: price outside 20-bar range
        high_20 = high.rolling(20).max().iloc[-1] if len(high) >= 20 else high.max()
        low_20 = low.rolling(20).min().iloc[-1] if len(low) >= 20 else low.min()
        range_size = high_20 - low_20
        
        if current_price >= high_20 * 0.99:
            structure = "breakout_up"
        elif current_price <= low_20 * 1.01:
            structure = "breakout_down"
        elif range_size / (current_price + 1e-10) < 0.03:
            structure = "tight_range"
        else:
            structure = "range_bound"
        
        # ── Reversal Detection ───────────────────────────────────────────
        # Check for potential reversal: strong move against trend
        pct_change_5 = (current_price - close.iloc[-6]) / (close.iloc[-6] + 1e-10) * 100 if len(close) >= 6 else 0
        
        if abs(pct_change_5) > 5:
            reversal = "strong_reversal"
        elif abs(pct_change_5) > 2:
            reversal = "weak_reversal"
        else:
            reversal = "none"
        
        return {
            "trend": trend,
            "trend_strength": round(float(trend_strength), 2),
            "volatility": volatility,
            "atr_pct": round(float(atr_pct), 2),
            "volume_regime": volume_regime,
            "volume_ratio": round(float(vol_ratio), 2),
            "structure": structure,
            "reversal": reversal,
            "pct_change_5": round(float(pct_change_5), 2),
        }
    
    except Exception as e:
        return {"error": str(e)}


# ── Performance Analysis ─────────────────────────────────────────────────────

def analyze_regime_performance(events: list[dict]) -> dict:
    """
    Analyze strategy performance broken down by regime.
    Returns metrics for each regime combination.
    """
    # Build trade records with regime data
    trades = []
    entries_by_key = {}
    
    for ev in events:
        if ev.get("type") == "entry":
            key = (ev.get("symbol", ""), ev.get("ts", ""))
            entries_by_key[key] = {
                "symbol": ev.get("symbol", ""),
                "signal": ev.get("signal", 0.5),
                "regime": ev.get("regime", "unknown"),
                "trend": ev.get("trend", "unknown"),
                "atr_pct": ev.get("atr_pct", 0),
                "price": ev.get("price", 0),
                "ts": ev.get("ts", ""),
            }
        elif ev.get("type") == "exit":
            sym = ev.get("symbol", "")
            exit_ts = ev.get("ts", "")
            
            # Find matching entry
            for (entry_sym, entry_ts), entry in entries_by_key.items():
                if entry_sym == sym and entry_ts < exit_ts:
                    pnl_pct = ev.get("pnl_pct", 0)
                    held_hours = ev.get("held_hours", 0)
                    
                    trades.append({
                        "symbol": sym,
                        "signal": entry["signal"],
                        "regime": entry["regime"],
                        "trend": entry["trend"],
                        "atr_pct": entry["atr_pct"],
                        "pnl_pct": pnl_pct,
                        "outcome": 1 if pnl_pct > 0 else 0,
                        "held_hours": held_hours,
                    })
                    break
    
    if not trades:
        return {"error": "No completed trades"}
    
    # Compute metrics for each regime dimension
    results = {}
    
    # ── By Volatility Regime ─────────────────────────────────────────────
    vol_groups = {}
    for t in trades:
        atr = t["atr_pct"]
        if atr > 6.0:
            vol = "extreme"
        elif atr > 4.0:
            vol = "high"
        elif atr > 2.0:
            vol = "normal"
        else:
            vol = "low"
        
        if vol not in vol_groups:
            vol_groups[vol] = []
        vol_groups[vol].append(t)
    
    results["volatility"] = _compute_group_metrics(vol_groups)
    
    # ── By Trend Regime ──────────────────────────────────────────────────
    trend_groups = {}
    for t in trades:
        trend = t.get("trend", "unknown")
        if trend not in trend_groups:
            trend_groups[trend] = []
        trend_groups[trend].append(t)
    
    results["trend"] = _compute_group_metrics(trend_groups)
    
    # ── By Original Regime (wild/normal/quiet) ───────────────────────────
    regime_groups = {}
    for t in trades:
        regime = t.get("regime", "unknown")
        if regime not in regime_groups:
            regime_groups[regime] = []
        regime_groups[regime].append(t)
    
    results["original_regime"] = _compute_group_metrics(regime_groups)
    
    # ── Combined: Trend x Volatility ─────────────────────────────────────
    combined_groups = {}
    for t in trades:
        trend = t.get("trend", "unknown")
        atr = t["atr_pct"]
        if atr > 4.0:
            vol = "high_vol"
        elif atr > 2.0:
            vol = "normal_vol"
        else:
            vol = "low_vol"
        
        key = f"{trend}_{vol}"
        if key not in combined_groups:
            combined_groups[key] = []
        combined_groups[key].append(t)
    
    results["combined"] = _compute_group_metrics(combined_groups)
    
    # ── By Signal Strength ───────────────────────────────────────────────
    signal_groups = {}
    for t in trades:
        sig = t["signal"]
        if sig >= 0.70:
            sig_group = "strong_70plus"
        elif sig >= 0.60:
            sig_group = "medium_60_70"
        elif sig >= 0.55:
            sig_group = "weak_55_60"
        else:
            sig_group = "bare_50_55"
        
        if sig_group not in signal_groups:
            signal_groups[sig_group] = []
        signal_groups[sig_group].append(t)
    
    results["signal_strength"] = _compute_group_metrics(signal_groups)
    
    # ── Summary Statistics ───────────────────────────────────────────────
    all_pnls = [t["pnl_pct"] for t in trades]
    all_outcomes = [t["outcome"] for t in trades]
    
    results["summary"] = {
        "total_trades": len(trades),
        "overall_win_rate": round(float(np.mean(all_outcomes)), 4),
        "overall_mean_pnl": round(float(np.mean(all_pnls)), 6),
        "overall_total_pnl": round(float(sum(all_pnls)), 4),
        "overall_sharpe": round(float(np.mean(all_pnls) / (np.std(all_pnls) + 1e-10)), 4),
    }
    
    return results


def _compute_group_metrics(groups: dict) -> dict:
    """Compute metrics for each group of trades."""
    results = {}
    
    for group_name, trades in groups.items():
        pnls = [t["pnl_pct"] for t in trades]
        outcomes = [t["outcome"] for t in trades]
        
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        win_rate = np.mean(outcomes) if outcomes else 0
        mean_pnl = np.mean(pnls) if pnls else 0
        total_pnl = sum(pnls)
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
        
        # Max drawdown within group
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0
        
        sharpe = np.mean(pnls) / (np.std(pnls) + 1e-10) if len(pnls) > 1 else 0
        
        results[group_name] = {
            "n_trades": len(trades),
            "win_rate": round(float(win_rate), 4),
            "mean_pnl": round(float(mean_pnl), 6),
            "total_pnl": round(float(total_pnl), 4),
            "avg_win": round(float(avg_win), 6),
            "avg_loss": round(float(avg_loss), 6),
            "profit_factor": round(float(profit_factor), 4) if profit_factor != float("inf") else "inf",
            "max_drawdown": round(float(max_dd), 4),
            "sharpe": round(float(sharpe), 4),
        }
    
    return results


# ── Regime Recommendations ───────────────────────────────────────────────────

def generate_recommendations(results: dict) -> list[str]:
    """Generate actionable recommendations based on regime analysis."""
    recs = []
    
    # Check which regimes are losing
    vol_metrics = results.get("volatility", {})
    for regime, metrics in vol_metrics.items():
        if isinstance(metrics, dict) and metrics.get("mean_pnl", 0) < 0:
            recs.append(f"AVOID: {regime} volatility regime (mean PnL: {metrics['mean_pnl']:.4%})")
    
    trend_metrics = results.get("trend", {})
    for regime, metrics in trend_metrics.items():
        if isinstance(metrics, dict) and metrics.get("mean_pnl", 0) < 0:
            recs.append(f"AVOID: {regime} trend regime (mean PnL: {metrics['mean_pnl']:.4%})")
    
    # Check which regimes are profitable
    combined = results.get("combined", {})
    best_regime = max(combined.items(), key=lambda x: x[1].get("mean_pnl", 0) if isinstance(x[1], dict) else 0)
    if isinstance(best_regime[1], dict) and best_regime[1].get("mean_pnl", 0) > 0:
        recs.append(f"FOCUS: {best_regime[0]} is most profitable (mean PnL: {best_regime[1]['mean_pnl']:.4%})")
    
    # Check signal strength
    signal_metrics = results.get("signal_strength", {})
    for group, metrics in signal_metrics.items():
        if isinstance(metrics, dict):
            if metrics.get("mean_pnl", 0) > 0 and metrics.get("n_trades", 0) >= 5:
                recs.append(f"STRENGTHEN: {group} signals are profitable (win rate: {metrics['win_rate']:.2%})")
            elif metrics.get("mean_pnl", 0) < 0 and metrics.get("n_trades", 0) >= 5:
                recs.append(f"WEAKEN: {group} signals are unprofitable (win rate: {metrics['win_rate']:.2%})")
    
    return recs


# ── Main ──────────────────────────────────────────────────────────────────────

def analyze_from_experiences(days: int = 7) -> dict:
    """Run full regime analysis from experience log."""
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
    
    performance = analyze_regime_performance(recent)
    recommendations = generate_recommendations(performance) if "error" not in performance else []
    
    return {
        "period_days": days,
        "performance": performance,
        "recommendations": recommendations,
    }


def print_report(results: dict):
    """Print regime analysis report."""
    print("\n" + "=" * 80)
    print("MARKET REGIME ANALYSIS")
    print("=" * 80)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    print(f"Period: Last {results['period_days']} days")
    
    perf = results.get("performance", {})
    summary = perf.get("summary", {})
    
    print(f"\nOverall: {summary.get('total_trades', 0)} trades | "
          f"Win Rate: {summary.get('overall_win_rate', 0):.2%} | "
          f"Mean PnL: {summary.get('overall_mean_pnl', 0):.4%} | "
          f"Total PnL: {summary.get('overall_total_pnl', 0):.4%}")
    
    # ── Regime Matrix ────────────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("REGIME MATRIX")
    print("-" * 80)
    
    # Volatility regimes
    vol = perf.get("volatility", {})
    if vol:
        print("\nVolatility Regime:")
        print(f"{'Regime':<12} {'Trades':>7} {'WinRate':>8} {'MeanPnL':>10} {'TotalPnL':>10} {'Sharpe':>8}")
        print("-" * 60)
        for regime in ["low", "normal", "high", "extreme"]:
            if regime in vol:
                m = vol[regime]
                print(f"{regime:<12} {m['n_trades']:>7} {m['win_rate']:>8.2%} "
                      f"{m['mean_pnl']:>10.4%} {m['total_pnl']:>10.4f} {m['sharpe']:>8.2f}")
    
    # Trend regimes
    trend = perf.get("trend", {})
    if trend:
        print("\nTrend Regime:")
        print(f"{'Regime':<12} {'Trades':>7} {'WinRate':>8} {'MeanPnL':>10} {'TotalPnL':>10} {'Sharpe':>8}")
        print("-" * 60)
        for regime in ["strong_bull", "weak_bull", "range", "weak_bear", "strong_bear"]:
            if regime in trend:
                m = trend[regime]
                print(f"{regime:<12} {m['n_trades']:>7} {m['win_rate']:>8.2%} "
                      f"{m['mean_pnl']:>10.4%} {m['total_pnl']:>10.4f} {m['sharpe']:>8.2f}")
    
    # Combined regimes
    combined = perf.get("combined", {})
    if combined:
        print("\nCombined Regime (Trend x Volatility):")
        print(f"{'Regime':<20} {'Trades':>7} {'WinRate':>8} {'MeanPnL':>10} {'TotalPnL':>10} {'Sharpe':>8}")
        print("-" * 70)
        for regime in sorted(combined.keys()):
            m = combined[regime]
            print(f"{regime:<20} {m['n_trades']:>7} {m['win_rate']:>8.2%} "
                  f"{m['mean_pnl']:>10.4%} {m['total_pnl']:>10.4f} {m['sharpe']:>8.2f}")
    
    # Signal strength
    signal = perf.get("signal_strength", {})
    if signal:
        print("\nSignal Strength:")
        print(f"{'Group':<12} {'Trades':>7} {'WinRate':>8} {'MeanPnL':>10} {'TotalPnL':>10} {'Sharpe':>8}")
        print("-" * 60)
        for group in ["bare_50_55", "weak_55_60", "medium_60_70", "strong_70plus"]:
            if group in signal:
                m = signal[group]
                print(f"{group:<12} {m['n_trades']:>7} {m['win_rate']:>8.2%} "
                      f"{m['mean_pnl']:>10.4%} {m['total_pnl']:>10.4f} {m['sharpe']:>8.2f}")
    
    # Recommendations
    recs = results.get("recommendations", [])
    if recs:
        print("\n" + "-" * 80)
        print("RECOMMENDATIONS")
        print("-" * 80)
        for rec in recs:
            print(f"  • {rec}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Market regime analysis")
    parser.add_argument("--days", type=int, default=7, help="Analysis window (days)")
    parser.add_argument("--matrix", action="store_true", help="Show regime matrix")
    args = parser.parse_args()
    
    results = analyze_from_experiences(days=args.days)
    print_report(results)


if __name__ == "__main__":
    main()
