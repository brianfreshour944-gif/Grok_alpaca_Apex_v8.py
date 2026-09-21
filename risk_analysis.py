"""
risk_analysis.py — Comprehensive risk management analysis.

Analyzes position sizing, portfolio risk, drawdown, tail risk,
and risk of ruin. Answers whether the bot adjusts risk with volatility.

Usage:
    python risk_analysis.py                     # full risk analysis
    python risk_analysis.py --days 30           # longer window
    python risk_analysis.py --stress-test       # tail risk scenarios
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
    BASE_RISK_PERCENT, MAX_SINGLE_TRADE_USD, MAX_OPEN_POSITIONS,
    MAX_DRAWDOWN_STOP, DAILY_LOSS_LIMIT,
    PROFIT_TARGET_PCT, STOP_LOSS_PCT,
)
from experience_capture import load_experiences
from portfolio import calculate_kelly_multiplier
from regime import calculate_adjusted_risk


# ── Position Sizing Analysis ─────────────────────────────────────────────────

def analyze_position_sizing(events: list[dict]) -> dict:
    """
    Analyze how position sizes are determined.
    Answers: Is risk the same when volatility doubles?
    """
    entries = [e for e in events if e.get("type") == "entry"]
    if not entries:
        return {"error": "No entries"}
    
    # Extract position sizing data
    sizing_data = []
    for ev in entries:
        equity = ev.get("equity", 0)
        atr_pct = ev.get("atr_pct", 0)
        signal = ev.get("signal", 0.5)
        regime = ev.get("regime", "unknown")
        risk_dollars = ev.get("risk_dollars", 0)
        
        if equity > 0 and risk_dollars > 0:
            # Calculate what sizing SHOULD be
            base_risk = equity * BASE_RISK_PERCENT
            
            # Volatility scaling
            baseline_vol = 1.5
            vol_scaler = min(baseline_vol / max(atr_pct, 0.01), 1.0)
            
            # Kelly scaling
            kelly_mult = calculate_kelly_multiplier(signal, PROFIT_TARGET_PCT, STOP_LOSS_PCT)
            
            expected_risk = min(base_risk * vol_scaler * kelly_mult, MAX_SINGLE_TRADE_USD)
            
            sizing_data.append({
                "equity": equity,
                "atr_pct": atr_pct,
                "signal": signal,
                "regime": regime,
                "actual_risk": risk_dollars,
                "expected_risk": expected_risk,
                "risk_pct": (risk_dollars / equity * 100) if equity > 0 else 0,
                "vol_scaler": vol_scaler,
                "kelly_mult": kelly_mult,
            })
    
    if not sizing_data:
        return {"error": "No valid sizing data"}
    
    # Analyze volatility scaling
    low_vol = [d for d in sizing_data if d["atr_pct"] < 2.0]
    high_vol = [d for d in sizing_data if d["atr_pct"] > 4.0]
    
    avg_risk_low_vol = np.mean([d["risk_pct"] for d in low_vol]) if low_vol else 0
    avg_risk_high_vol = np.mean([d["risk_pct"] for d in high_vol]) if high_vol else 0
    vol_scaling_ratio = avg_risk_high_vol / avg_risk_low_vol if avg_risk_low_vol > 0 else 1.0
    
    # Check if scaling is working correctly
    # When volatility doubles, risk should roughly halve
    scaling_effective = vol_scaling_ratio < 0.75  # Allow 25% tolerance
    
    return {
        "n_trades": len(sizing_data),
        "avg_risk_pct": round(float(np.mean([d["risk_pct"] for d in sizing_data])), 4),
        "avg_vol_scaler": round(float(np.mean([d["vol_scaler"] for d in sizing_data])), 4),
        "avg_kelly_mult": round(float(np.mean([d["kelly_mult"] for d in sizing_data])), 4),
        "volatility_scaling": {
            "avg_risk_low_vol": round(float(avg_risk_low_vol), 4),
            "avg_risk_high_vol": round(float(avg_risk_high_vol), 4),
            "scaling_ratio": round(float(vol_scaling_ratio), 4),
            "scaling_effective": scaling_effective,
            "interpretation": "Risk scales DOWN with volatility" if vol_scaling_ratio < 1.0 else "Risk does NOT scale with volatility",
        },
        "sizing_range": {
            "min_risk_pct": round(float(min(d["risk_pct"] for d in sizing_data)), 4),
            "max_risk_pct": round(float(max(d["risk_pct"] for d in sizing_data)), 4),
            "min_dollars": round(float(min(d["actual_risk"] for d in sizing_data)), 2),
            "max_dollars": round(float(max(d["actual_risk"] for d in sizing_data)), 2),
            "at_max_cap": sum(1 for d in sizing_data if d["actual_risk"] >= MAX_SINGLE_TRADE_USD * 0.99),
        },
        "sizing_data": sizing_data[:20],  # First 20 for inspection
    }


# ── Portfolio Risk ───────────────────────────────────────────────────────────

def analyze_portfolio_risk(events: list[dict]) -> dict:
    """
    Analyze portfolio-level risk metrics.
    """
    entries = [e for e in events if e.get("type") == "entry"]
    exits = [e for e in events if e.get("type") == "exit"]
    
    if not entries:
        return {"error": "No entries"}
    
    # Count simultaneous positions
    position_times = []
    for ev in entries:
        entry_time = ev.get("ts", "")
        # Assume average hold of 2 hours if no exit found
        exit_time = None
        for ex in exits:
            if ex.get("symbol") == ev.get("symbol") and ex.get("ts", "") > entry_time:
                exit_time = ex.get("ts", "")
                break
        
        position_times.append({
            "entry": entry_time,
            "exit": exit_time or entry_time,
            "symbol": ev.get("symbol", ""),
        })
    
    # Calculate max simultaneous positions
    # (Simplified: assume each position held for average duration)
    avg_held_hours = np.mean([
        e.get("held_hours", 2) for e in exits
    ]) if exits else 2.0
    
    # Exposure analysis
    equity_values = [e.get("equity", 0) for e in entries if e.get("equity", 0) > 0]
    risk_dollars = [e.get("risk_dollars", 0) for e in entries if e.get("risk_dollars", 0) > 0]
    
    avg_equity = np.mean(equity_values) if equity_values else 0
    avg_risk = np.mean(risk_dollars) if risk_dollars else 0
    max_risk = max(risk_dollars) if risk_dollars else 0
    
    # Portfolio exposure
    total_risk_per_cycle = avg_risk * MAX_OPEN_POSITIONS
    portfolio_risk_pct = (total_risk_per_cycle / avg_equity * 100) if avg_equity > 0 else 0
    
    return {
        "n_positions": len(entries),
        "avg_held_hours": round(float(avg_held_hours), 2),
        "avg_equity": round(float(avg_equity), 2),
        "avg_risk_per_trade": round(float(avg_risk), 2),
        "max_risk_per_trade": round(float(max_risk), 2),
        "portfolio_risk": {
            "total_risk_per_cycle": round(float(total_risk_per_cycle), 2),
            "portfolio_risk_pct": round(float(portfolio_risk_pct), 4),
            "max_positions": MAX_OPEN_POSITIONS,
            "max_portfolio_exposure": round(float(avg_equity * BASE_RISK_PERCENT * MAX_OPEN_POSITIONS), 2),
        },
        "limits": {
            "max_single_trade": MAX_SINGLE_TRADE_USD,
            "max_open_positions": MAX_OPEN_POSITIONS,
            "max_drawdown_stop": MAX_DRAWDOWN_STOP,
            "daily_loss_limit": DAILY_LOSS_LIMIT,
        },
    }


# ── Drawdown Analysis ────────────────────────────────────────────────────────

def analyze_drawdown(events: list[dict]) -> dict:
    """
    Analyze drawdown characteristics from equity curve.
    """
    # Build equity curve from entries
    equity_curve = []
    for ev in events:
        equity = ev.get("equity", 0)
        if equity > 0:
            equity_curve.append({
                "ts": ev.get("ts", ""),
                "equity": equity,
            })
    
    if len(equity_curve) < 2:
        return {"error": "Insufficient equity data"}
    
    equities = np.array([e["equity"] for e in equity_curve])
    
    # Calculate drawdown series
    running_max = np.maximum.accumulate(equities)
    drawdowns = (equities - running_max) / running_max * 100
    
    # Drawdown statistics
    max_dd = float(np.min(drawdowns))
    avg_dd = float(np.mean(drawdowns[drawdowns < 0])) if np.any(drawdowns < 0) else 0
    
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
    
    avg_dd_duration = np.mean(dd_durations) if dd_durations else 0
    max_dd_duration = max(dd_durations) if dd_durations else 0
    
    # Recovery analysis
    recovery_times = []
    peak_idx = 0
    for i, eq in enumerate(equities):
        if eq >= equities[peak_idx]:
            if peak_idx < i:
                recovery_times.append(i - peak_idx)
            peak_idx = i
    
    avg_recovery = np.mean(recovery_times) if recovery_times else 0
    
    # Max consecutive losses
    pnl_series = np.diff(equities) / equities[:-1] * 100
    consecutive_losses = 0
    max_consecutive_losses = 0
    
    for pnl in pnl_series:
        if pnl < 0:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        else:
            consecutive_losses = 0
    
    return {
        "n_equity_points": len(equities),
        "equity_range": {
            "start": round(float(equities[0]), 2),
            "end": round(float(equities[-1]), 2),
            "min": round(float(np.min(equities)), 2),
            "max": round(float(np.max(equities)), 2),
        },
        "drawdown": {
            "max_drawdown_pct": round(float(max_dd), 4),
            "avg_drawdown_pct": round(float(avg_dd), 4),
            "current_drawdown_pct": round(float(drawdowns[-1]), 4),
        },
        "drawdown_duration": {
            "avg_duration": round(float(avg_dd_duration), 1),
            "max_duration": round(float(max_dd_duration), 1),
            "n_drawdowns": len(dd_durations),
        },
        "recovery": {
            "avg_recovery_time": round(float(avg_recovery), 1),
            "max_recovery_time": round(float(max(recovery_times)) if recovery_times else 0, 1),
        },
        "consecutive_losses": {
            "max_consecutive": max_consecutive_losses,
        },
    }


# ── Tail Risk Analysis ──────────────────────────────────────────────────────

def analyze_tail_risk(events: list[dict]) -> dict:
    """
    Analyze tail risk: flash crashes, gaps, extreme moves.
    """
    exits = [e for e in events if e.get("type") == "exit"]
    
    if not exits:
        return {"error": "No exits"}
    
    pnl_pcts = [e.get("pnl_pct", 0) for e in exits]
    pnls = np.array(pnl_pcts)
    
    # Tail risk metrics
    var_95 = float(np.percentile(pnls, 5))  # 5th percentile = 95% VaR
    var_99 = float(np.percentile(pnls, 1))  # 1st percentile = 99% VaR
    
    cvar_95 = float(np.mean(pnls[pnls <= var_95])) if np.any(pnls <= var_95) else var_95
    cvar_99 = float(np.mean(pnls[pnls <= var_99])) if np.any(pnls <= var_99) else var_99
    
    # Kurtosis (fat tails)
    from scipy import stats as scipy_stats
    kurtosis = float(scipy_stats.kurtosis(pnls)) if len(pnls) > 3 else 0
    
    # Skewness
    skewness = float(scipy_stats.skew(pnls)) if len(pnls) > 3 else 0
    
    # Worst-case scenarios
    worst_loss = float(np.min(pnls))
    worst_loss_trade = exits[np.argmin(pnl_pcts)]
    
    # Flash crash detection: losses > 3x average loss
    avg_loss = float(np.mean(pnls[pnls < 0])) if np.any(pnls < 0) else 0
    flash_crash_threshold = avg_loss * 3 if avg_loss < 0 else -5.0
    flash_crashes = [e for e in exits if e.get("pnl_pct", 0) < flash_crash_threshold]
    
    return {
        "n_trades": len(pnls),
        "var_95": round(float(var_95), 4),
        "var_99": round(float(var_99), 4),
        "cvar_95": round(float(cvar_95), 4),
        "cvar_99": round(float(cvar_99), 4),
        "kurtosis": round(float(kurtosis), 4),
        "skewness": round(float(skewness), 4),
        "tail_interpretation": {
            "fat_tails": kurtosis > 3,
            "negative_skew": skewness < 0,
            "interpretation": (
                "Fat tails + negative skew = higher tail risk" if kurtosis > 3 and skewness < 0
                else "Normal tails" if kurtosis <= 3 and abs(skewness) <= 1
                else "Mixed tail characteristics"
            ),
        },
        "worst_case": {
            "worst_loss_pct": round(float(worst_loss), 4),
            "worst_trade": {
                "symbol": worst_loss_trade.get("symbol", ""),
                "pnl_pct": worst_loss_trade.get("pnl_pct", 0),
                "ts": worst_loss_trade.get("ts", ""),
            },
        },
        "flash_crashes": {
            "threshold_pct": round(float(flash_crash_threshold), 4),
            "count": len(flash_crashes),
            "trades": [{"symbol": e.get("symbol"), "pnl_pct": e.get("pnl_pct")} for e in flash_crashes[:5]],
        },
    }


# ── Risk of Ruin ─────────────────────────────────────────────────────────────

def calculate_risk_of_ruin(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    risk_per_trade: float,
    account_size: float,
    max_drawdown_pct: float = 10.0,
    max_consecutive_losses: int = 10,
) -> dict:
    """
    Calculate probability of losing a significant portion of account.
    Uses multiple methods: historical, Monte Carlo, analytical.
    """
    # Avoid division by zero
    if avg_loss == 0:
        avg_loss = 0.01
    if risk_per_trade == 0:
        risk_per_trade = account_size * 0.02
    
    # Expected value per trade
    expected_value = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    # Reward-to-risk ratio
    reward_risk = avg_win / abs(avg_loss) if avg_loss != 0 else 1.0
    
    # Kelly fraction (optimal bet size)
    kelly = win_rate - ((1 - win_rate) / reward_risk) if reward_risk > 0 else 0
    half_kelly = kelly / 2  # Conservative sizing
    
    # Risk of Ruin (analytical approximation)
    # Using the formula: RoR = ((1-edge) / (1+edge))^units
    # where edge = expected value per unit risked, in R-multiples
    # (win_rate*reward_risk - loss_rate) -- NOT raw expected_value/risk_per_trade.
    # FIX: expected_value is computed from avg_win/avg_loss in whatever
    # scale the caller passed them (stress_test.py passes percentage-points,
    # e.g. 2.0 for 2%; this function's other caller, analyze_from_experiences
    # below, passes fractions, e.g. 0.02 -- a real, pre-existing inconsistency
    # between this function's two call sites) while risk_per_trade is always
    # a dollar amount. Dividing one by the other produced a dimensionless
    # number that was off by ~100x whenever percentage-point inputs were
    # used, making even a genuinely strong positive edge evaluate to
    # risk_of_ruin_analytical=1.0 (100%) via the `else` branch below.
    # reward_risk is a ratio, so it cancels whatever scale avg_win/avg_loss
    # were passed in -- correct regardless of which convention a caller uses.
    edge = (win_rate * reward_risk) - (1 - win_rate)
    units_to_ruin = (account_size * max_drawdown_pct / 100) / abs(risk_per_trade) if risk_per_trade != 0 else 100
    
    if edge > 0 and (1 + edge) > 0:
        risk_of_ruin_analytical = ((1 - edge) / (1 + edge)) ** units_to_ruin
    else:
        risk_of_ruin_analytical = 1.0  # 100% if no edge
    
    # Monte Carlo simulation
    # FIX: this used to bet a FIXED dollar amount (risk_per_trade, computed
    # once from the STARTING account_size) every trade regardless of how
    # equity changed across the simulated year. The bot itself never sizes
    # this way -- calculate_adjusted_risk()/calculate_kelly_multiplier()
    # always size as a % of CURRENT equity, recalculated fresh every trade
    # (regime.py, portfolio.py). Fixed-DOLLAR betting can hit literal $0
    # equity; fixed-FRACTIONAL betting (the bot's real behavior) shrinks
    # position size as losses accumulate and mathematically cannot cross
    # zero, so the old simulation materially overstated true ruin
    # probability. risk_fraction recovers the equity-% the caller intended
    # (both current call sites size risk_per_trade as
    # account_size * BASE_RISK_PERCENT) and re-applies it against the
    # CURRENT simulated equity every trade instead of the starting one.
    n_simulations = 10000
    n_trades = 252  # One year of daily trades
    risk_fraction = abs(risk_per_trade) / account_size if account_size > 0 else 0.02

    ruins = 0
    max_drawdowns = []

    for _ in range(n_simulations):
        equity = account_size
        peak = account_size
        max_dd = 0

        for _ in range(n_trades):
            current_risk = equity * risk_fraction
            if np.random.random() < win_rate:
                equity += current_risk * reward_risk
            else:
                equity -= current_risk

            if equity <= 0:
                ruins += 1
                max_dd = 100
                break

            peak = max(peak, equity)
            dd = (peak - equity) / peak * 100
            max_dd = max(max_dd, dd)

        max_drawdowns.append(max_dd)

    risk_of_ruin_mc = ruins / n_simulations

    # Probability a simulated year breaches the account's actual configured
    # drawdown kill-switch (max_drawdown_pct), not just literal $0 equity.
    # FIX: fixed-fractional sizing (above) correctly makes literal ruin
    # near-impossible over a bounded horizon even for a losing edge -- but
    # that alone would make risk_of_ruin_mc misleadingly report "LOW RISK"
    # for a scenario whose OWN drawdown stats (avg/worst/p95_max_drawdown
    # below) show the account getting gutted well before it ever reaches
    # $0. This is the more practically relevant "ruin" for a bot that
    # actually halts trading at max_drawdown_pct, not at $0.
    prob_dd_exceeds_threshold = float(np.mean([dd > max_drawdown_pct for dd in max_drawdowns]))

    # Maximum consecutive losses (probabilistic)
    prob_consec_loss = (1 - win_rate) ** max_consecutive_losses
    
    # Break-even win rate
    break_even_win_rate = abs(avg_loss) / (avg_win + abs(avg_loss)) if (avg_win + abs(avg_loss)) > 0 else 0.5
    
    return {
        "input_params": {
            "win_rate": round(float(win_rate), 4),
            "avg_win_pct": round(float(avg_win), 4),
            "avg_loss_pct": round(float(avg_loss), 4),
            "risk_per_trade_usd": round(float(risk_per_trade), 2),
            "account_size": round(float(account_size), 2),
            "max_drawdown_pct": max_drawdown_pct,
        },
        "expected_value": {
            "per_trade": round(float(expected_value), 6),
            "per_trade_pct": round(float(expected_value / account_size * 100), 4) if account_size > 0 else 0,
            "annual_expected": round(float(expected_value * n_trades), 2),
            "annual_expected_pct": round(float(expected_value * n_trades / account_size * 100), 2) if account_size > 0 else 0,
        },
        "kelly_criterion": {
            "full_kelly": round(float(kelly), 4),
            "half_kelly": round(float(half_kelly), 4),
            "optimal_bet_pct": round(float(half_kelly * 100), 2),
            "interpretation": "Bet {:.1f}% of bankroll per trade".format(half_kelly * 100),
        },
        "risk_of_ruin": {
            "analytical": round(float(risk_of_ruin_analytical), 6),
            "monte_carlo": round(float(risk_of_ruin_mc), 6),
            "prob_consec_losses": round(float(prob_consec_loss), 6),
            "prob_drawdown_exceeds_threshold": round(prob_dd_exceeds_threshold, 6),
            # FIX: interpretation previously looked ONLY at risk_of_ruin_mc
            # (probability of literal $0 equity). Once Monte Carlo sizing
            # was corrected to fixed-fractional (matching how the bot
            # actually sizes trades), risk_of_ruin_mc correctly drops to
            # ~0 for almost any scenario short of a coin-flip-or-worse edge
            # with a huge bet size -- which made this report "LOW RISK"
            # for scenarios whose OWN avg/worst/p95_max_drawdown numbers
            # a few lines down show the account getting gutted 60-90%
            # before ever reaching zero. Weighing
            # prob_drawdown_exceeds_threshold (chance of breaching the
            # account's actual configured kill-switch) alongside literal
            # ruin keeps the headline verdict from contradicting the
            # detailed data sitting right next to it.
            "interpretation": (
                "VERY HIGH RISK" if risk_of_ruin_mc > 0.1 or prob_dd_exceeds_threshold > 0.5
                else "HIGH RISK" if risk_of_ruin_mc > 0.01 or prob_dd_exceeds_threshold > 0.2
                else "MODERATE RISK" if risk_of_ruin_mc > 0.001 or prob_dd_exceeds_threshold > 0.05
                else "LOW RISK"
            ),
        },
        "break_even": {
            "win_rate_needed": round(float(break_even_win_rate), 4),
            "current_edge": round(float(win_rate - break_even_win_rate), 4),
        },
        "monte_carlo": {
            "simulations": n_simulations,
            "trades_per_sim": n_trades,
            "avg_max_drawdown": round(float(np.mean(max_drawdowns)), 4),
            "worst_max_drawdown": round(float(np.max(max_drawdowns)), 4),
            "p95_max_drawdown": round(float(np.percentile(max_drawdowns, 95)), 4),
        },
    }


# ── Main Analysis ────────────────────────────────────────────────────────────

def analyze_from_experiences(days: int = 7) -> dict:
    """Run full risk analysis from experience log."""
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
    position_sizing = analyze_position_sizing(recent)
    portfolio_risk = analyze_portfolio_risk(recent)
    drawdown = analyze_drawdown(recent)
    tail_risk = analyze_tail_risk(recent)
    
    # Calculate risk of ruin from actual data
    exits = [e for e in recent if e.get("type") == "exit"]
    if exits:
        pnls = [e.get("pnl_pct", 0) for e in exits]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        win_rate = len(wins) / len(pnls) if pnls else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Estimate account size from equity data
        equities = [e.get("equity", 0) for e in recent if e.get("equity", 0) > 0]
        account_size = np.mean(equities) if equities else 10000
        
        risk_per_trade = account_size * BASE_RISK_PERCENT
        
        risk_of_ruin = calculate_risk_of_ruin(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            risk_per_trade=risk_per_trade,
            account_size=account_size,
            max_drawdown_pct=abs(MAX_DRAWDOWN_STOP),
        )
    else:
        risk_of_ruin = {"error": "No exits for calculation"}
    
    return {
        "period_days": days,
        "position_sizing": position_sizing,
        "portfolio_risk": portfolio_risk,
        "drawdown": drawdown,
        "tail_risk": tail_risk,
        "risk_of_ruin": risk_of_ruin,
    }


def print_report(results: dict):
    """Print risk analysis report."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RISK MANAGEMENT ANALYSIS")
    print("=" * 80)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    print(f"Period: Last {results['period_days']} days")
    
    # ── Position Sizing ──────────────────────────────────────────────────
    sizing = results.get("position_sizing", {})
    if "error" not in sizing:
        print("\n" + "-" * 80)
        print("POSITION SIZING")
        print("-" * 80)
        print(f"Avg Risk Per Trade: {sizing.get('avg_risk_pct', 0):.2f}% of equity")
        print(f"Avg Volatility Scaler: {sizing.get('avg_vol_scaler', 0):.2f}x")
        print(f"Avg Kelly Multiplier: {sizing.get('avg_kelly_mult', 0):.2f}x")
        
        vol = sizing.get("volatility_scaling", {})
        print(f"\nVolatility Scaling:")
        print(f"  Low Vol Avg Risk: {vol.get('avg_risk_low_vol', 0):.2f}%")
        print(f"  High Vol Avg Risk: {vol.get('avg_risk_high_vol', 0):.2f}%")
        print(f"  Scaling Ratio: {vol.get('scaling_ratio', 0):.2f}x")
        print(f"  Effective: {'YES ✓' if vol.get('scaling_effective') else 'NO ✗'}")
        print(f"  Interpretation: {vol.get('interpretation', 'N/A')}")
        
        r = sizing.get("sizing_range", {})
        print(f"\nSizing Range:")
        print(f"  Min: {r.get('min_risk_pct', 0):.2f}% (${r.get('min_dollars', 0):.2f})")
        print(f"  Max: {r.get('max_risk_pct', 0):.2f}% (${r.get('max_dollars', 0):.2f})")
        print(f"  Trades at Max Cap: {r.get('at_max_cap', 0)}")
    
    # ── Portfolio Risk ───────────────────────────────────────────────────
    portfolio = results.get("portfolio_risk", {})
    if "error" not in portfolio:
        print("\n" + "-" * 80)
        print("PORTFOLIO RISK")
        print("-" * 80)
        pr = portfolio.get("portfolio_risk", {})
        print(f"Max Single Trade: ${portfolio.get('limits', {}).get('max_single_trade', 0):,.2f}")
        print(f"Max Open Positions: {portfolio.get('limits', {}).get('max_open_positions', 0)}")
        print(f"Portfolio Risk Per Cycle: ${pr.get('total_risk_per_cycle', 0):,.2f}")
        print(f"Portfolio Risk %: {pr.get('portfolio_risk_pct', 0):.2f}%")
        print(f"Max Portfolio Exposure: ${pr.get('max_portfolio_exposure', 0):,.2f}")
    
    # ── Drawdown ─────────────────────────────────────────────────────────
    dd = results.get("drawdown", {})
    if "error" not in dd:
        print("\n" + "-" * 80)
        print("DRAWDOWN ANALYSIS")
        print("-" * 80)
        eq = dd.get("equity_range", {})
        print(f"Equity: ${eq.get('start', 0):,.2f} → ${eq.get('end', 0):,.2f}")
        print(f"Min: ${eq.get('min', 0):,.2f} | Max: ${eq.get('max', 0):,.2f}")
        
        drawdown = dd.get("drawdown", {})
        print(f"\nMax Drawdown: {drawdown.get('max_drawdown_pct', 0):.2f}%")
        print(f"Avg Drawdown: {drawdown.get('avg_drawdown_pct', 0):.2f}%")
        print(f"Current Drawdown: {drawdown.get('current_drawdown_pct', 0):.2f}%")
        
        dur = dd.get("drawdown_duration", {})
        print(f"Avg Duration: {dur.get('avg_duration', 0):.1f} periods")
        print(f"Max Duration: {dur.get('max_duration', 0):.1f} periods")
        print(f"Number of Drawdowns: {dur.get('n_drawdowns', 0)}")
        
        rec = dd.get("recovery", {})
        print(f"Avg Recovery Time: {rec.get('avg_recovery_time', 0):.1f} periods")
        print(f"Max Recovery Time: {rec.get('max_recovery_time', 0):.1f} periods")
    
    # ── Tail Risk ────────────────────────────────────────────────────────
    tail = results.get("tail_risk", {})
    if "error" not in tail:
        print("\n" + "-" * 80)
        print("TAIL RISK ANALYSIS")
        print("-" * 80)
        print(f"VaR (95%): {tail.get('var_95', 0):.4f}%")
        print(f"VaR (99%): {tail.get('var_99', 0):.4f}%")
        print(f"CVaR (95%): {tail.get('cvar_95', 0):.4f}%")
        print(f"CVaR (99%): {tail.get('cvar_99', 0):.4f}%")
        print(f"Kurtosis: {tail.get('kurtosis', 0):.4f}")
        print(f"Skewness: {tail.get('skewness', 0):.4f}")
        
        interp = tail.get("tail_interpretation", {})
        print(f"Fat Tails: {'YES' if interp.get('fat_tails') else 'NO'}")
        print(f"Negative Skew: {'YES' if interp.get('negative_skew') else 'NO'}")
        print(f"Interpretation: {interp.get('interpretation', 'N/A')}")
        
        worst = tail.get("worst_case", {})
        print(f"\nWorst Case:")
        print(f"  Worst Loss: {worst.get('worst_loss_pct', 0):.4f}%")
        
        fc = tail.get("flash_crashes", {})
        print(f"\nFlash Crash Analysis:")
        print(f"  Threshold: {fc.get('threshold_pct', 0):.4f}%")
        print(f"  Count: {fc.get('count', 0)}")
    
    # ── Risk of Ruin ─────────────────────────────────────────────────────
    ror = results.get("risk_of_ruin", {})
    if "error" not in ror:
        print("\n" + "-" * 80)
        print("RISK OF RUIN")
        print("-" * 80)
        params = ror.get("input_params", {})
        print(f"Win Rate: {params.get('win_rate', 0):.2%}")
        print(f"Avg Win: {params.get('avg_win_pct', 0):.4f}%")
        print(f"Avg Loss: {params.get('avg_loss_pct', 0):.4f}%")
        print(f"Risk Per Trade: ${params.get('risk_per_trade_usd', 0):,.2f}")
        print(f"Account Size: ${params.get('account_size', 0):,.2f}")
        
        ev = ror.get("expected_value", {})
        print(f"\nExpected Value:")
        print(f"  Per Trade: {ev.get('per_trade', 0):.6f}")
        print(f"  Per Trade %: {ev.get('per_trade_pct', 0):.4f}%")
        print(f"  Annual Expected: ${ev.get('annual_expected', 0):,.2f}")
        print(f"  Annual Expected %: {ev.get('annual_expected_pct', 0):.2f}%")
        
        kelly = ror.get("kelly_criterion", {})
        print(f"\nKelly Criterion:")
        print(f"  Full Kelly: {kelly.get('full_kelly', 0):.4f}")
        print(f"  Half Kelly (recommended): {kelly.get('half_kelly', 0):.4f}")
        print(f"  Optimal Bet: {kelly.get('optimal_bet_pct', 0):.2f}% of bankroll")
        
        risk = ror.get("risk_of_ruin", {})
        print(f"\nRisk of Ruin:")
        print(f"  Analytical: {risk.get('analytical', 0):.6f}")
        print(f"  Monte Carlo: {risk.get('monte_carlo', 0):.6f}")
        print(f"  Interpretation: {risk.get('interpretation', 'N/A')}")
        
        be = ror.get("break_even", {})
        print(f"\nBreak-Even:")
        print(f"  Win Rate Needed: {be.get('win_rate_needed', 0):.2%}")
        print(f"  Current Edge: {be.get('current_edge', 0):.4f}")
        
        mc = ror.get("monte_carlo", {})
        print(f"\nMonte Carlo ({mc.get('simulations', 0):,} simulations, {mc.get('trades_per_sim', 0)} trades):")
        print(f"  Avg Max Drawdown: {mc.get('avg_max_drawdown', 0):.2f}%")
        print(f"  Worst Max Drawdown: {mc.get('worst_max_drawdown', 0):.2f}%")
        print(f"  P95 Max Drawdown: {mc.get('p95_max_drawdown', 0):.2f}%")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive risk analysis")
    parser.add_argument("--days", type=int, default=7, help="Analysis window (days)")
    parser.add_argument("--stress-test", action="store_true", help="Run stress tests")
    args = parser.parse_args()
    
    results = analyze_from_experiences(days=args.days)
    print_report(results)


if __name__ == "__main__":
    main()
