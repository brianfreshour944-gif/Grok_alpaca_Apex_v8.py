"""
drift_monitor.py — Live model performance monitoring and drift detection.

Monitors the champion model's live predictions vs realized outcomes,
detects performance degradation, and triggers alerts/retraining.

Usage:
    python drift_monitor.py                  # show current drift status
    python drift_monitor.py --alert          # test Discord alert
    python drift_monitor.py --days 7         # analyze last 7 days
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from config import logger, DISCORD_WEBHOOK_URL, EXPERIENCE_LOG_PATH
from experience_capture import load_experiences
from feature_engineering import FEATURE_COLS


# ── Config ────────────────────────────────────────────────────────────────────
DRIFT_STATE_PATH = os.getenv("DRIFT_STATE_PATH", "drift_state.json")
ALERT_COOLDOWN_HOURS = int(os.getenv("DRIFT_ALERT_COOLDOWN_HOURS", "6"))

# Thresholds
ACCURACY_FLOOR = float(os.getenv("DRIFT_ACCURACY_FLOOR", "0.45"))
SHARPE_FLOOR = float(os.getenv("DRIFT_SHARPE_FLOOR", "0.5"))
WIN_RATE_FLOOR = float(os.getenv("DRIFT_WIN_RATE_FLOOR", "0.40"))
CONSECUTIVE_LOSSES_LIMIT = int(os.getenv("DRIFT_CONSECUTIVE_LOSSES", "5"))
REGIME_DRIFT_THRESHOLD = float(os.getenv("DRIFT_REGIME_THRESHOLD", "0.3"))


# ── State management ──────────────────────────────────────────────────────────

def load_state() -> dict:
    """Load drift monitoring state."""
    if os.path.exists(DRIFT_STATE_PATH):
        try:
            with open(DRIFT_STATE_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load drift state from {DRIFT_STATE_PATH}: {e}")
    return {
        "last_alert_ts": None,
        "recent_signals": [],
        "recent_outcomes": [],
        "regime_counts": {},
        "drift_score": 0.0,
    }


def save_state(state: dict):
    """Save drift monitoring state."""
    try:
        with open(DRIFT_STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save drift state: {e}")


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_recent_performance(days: int = 7) -> dict:
    """
    Analyze model performance over recent period.
    Returns metrics dict.
    """
    events = load_experiences()
    if not events:
        return {"error": "No experiences found"}

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Collect recent entries and exits
    entries = []
    exits = []
    for ev in events:
        try:
            ts_str = ev.get("ts", "")
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts < cutoff:
                continue
        except (ValueError, TypeError):
            continue

        if ev.get("type") == "entry":
            entries.append(ev)
        elif ev.get("type") == "exit":
            exits.append(ev)

    if not entries:
        return {"error": f"No entries in last {days} days", "entries": 0, "exits": 0}

    # Match entries to exits
    signals = []
    outcomes = []
    regimes = []
    feature_series = {col: [] for col in FEATURE_COLS}

    for entry in entries:
        symbol = entry.get("symbol", "")
        signal = entry.get("signal", 0.5)
        entry_ts = entry.get("ts", "")
        regime = entry.get("regime", "unknown")
        features = entry.get("features", {})

        # Collect feature values
        for col in FEATURE_COLS:
            if col in features:
                feature_series[col].append(features[col])

        # Find matching exit
        for exit_ev in exits:
            if exit_ev.get("symbol") == symbol and exit_ev.get("ts", "") > entry_ts:
                pnl = exit_ev.get("pnl_pct", 0)
                signals.append(signal)
                outcomes.append(1 if pnl > 0 else 0)
                regimes.append(regime)
                break

    if not signals:
        return {"error": "No completed trades", "entries": len(entries), "exits": 0}

    signals = np.array(signals)
    outcomes = np.array(outcomes)

    # Metrics
    accuracy = outcomes.mean()
    win_rate = accuracy
    avg_signal = signals.mean()
    
    # Sharpe-like ratio (mean return / std of returns)
    # Approximate from win rate and avg signal
    if len(outcomes) > 1:
        sharpe = (outcomes.mean() - 0.5) / (outcomes.std() + 1e-8)
    else:
        sharpe = 0.0

    # Consecutive losses
    max_consecutive_losses = 0
    current_streak = 0
    for o in outcomes:
        if o == 0:
            current_streak += 1
            max_consecutive_losses = max(max_consecutive_losses, current_streak)
        else:
            current_streak = 0

    # Regime distribution
    regime_dist = {}
    for r in regimes:
        regime_dist[r] = regime_dist.get(r, 0) + 1

    # Feature degradation analysis
    feature_degradation = {}
    for col in FEATURE_COLS:
        values = feature_series.get(col, [])
        if len(values) > 10:
            arr = np.array(values)
            # Check if feature has become degenerate (low variance)
            variance = arr.var()
            # Check if feature is drifting (mean far from 0 for Z-scored features)
            mean_drift = abs(arr.mean())
            # Check for extreme values
            extreme_pct = ((arr < -3) | (arr > 3)).mean() * 100
            
            feature_degradation[col] = {
                "variance": round(float(variance), 6),
                "mean_drift": round(float(mean_drift), 4),
                "extreme_pct": round(float(extreme_pct), 2),
                "n_samples": len(values),
                "is_degenerate": variance < 0.01,
                "is_drifting": mean_drift > 1.0,
                "has_extremes": extreme_pct > 5.0,
            }

    # Drift score: 0 = no drift, 1 = severe drift
    drift_score = 0.0
    drift_signals = []

    if accuracy < ACCURACY_FLOOR:
        drift_score += 0.3
        drift_signals.append(f"Accuracy {accuracy:.2%} < {ACCURACY_FLOOR:.2%}")

    if win_rate < WIN_RATE_FLOOR:
        drift_score += 0.3
        drift_signals.append(f"Win rate {win_rate:.2%} < {WIN_RATE_FLOOR:.2%}")

    if sharpe < SHARPE_FLOOR:
        drift_score += 0.2
        drift_signals.append(f"Sharpe {sharpe:.2f} < {SHARPE_FLOOR:.2f}")

    if max_consecutive_losses >= CONSECUTIVE_LOSSES_LIMIT:
        drift_score += 0.2
        drift_signals.append(f"Consecutive losses: {max_consecutive_losses}")

    # Check for feature degradation
    degenerate_features = [col for col, stats in feature_degradation.items() if stats.get("is_degenerate")]
    drifting_features = [col for col, stats in feature_degradation.items() if stats.get("is_drifting")]
    
    if degenerate_features:
        drift_score += 0.2
        drift_signals.append(f"Degenerate features: {', '.join(degenerate_features)}")
    if drifting_features:
        drift_score += 0.1
        drift_signals.append(f"Drifting features: {', '.join(drifting_features)}")

    return {
        "period_days": days,
        "entries": len(entries),
        "exits": len(exits),
        "signals_count": len(signals),
        "accuracy": accuracy,
        "win_rate": win_rate,
        "avg_signal": avg_signal,
        "sharpe": sharpe,
        "max_consecutive_losses": max_consecutive_losses,
        "regime_distribution": regime_dist,
        "drift_score": min(drift_score, 1.0),
        "drift_signals": drift_signals,
        "is_drifting": drift_score > 0.5,
        "feature_degradation": feature_degradation,
    }


# ── Alerting ──────────────────────────────────────────────────────────────────

async def send_drift_alert(metrics: dict, state: dict) -> bool:
    """Send Discord alert if drift detected and cooldown expired."""
    if not DISCORD_WEBHOOK_URL:
        logger.info("No Discord webhook configured, skipping alert")
        return False

    # Check cooldown
    last_alert = state.get("last_alert_ts")
    if last_alert:
        try:
            last_dt = datetime.fromisoformat(last_alert.replace("Z", "+00:00"))
            if (datetime.now(timezone.utc) - last_dt).total_seconds() < ALERT_COOLDOWN_HOURS * 3600:
                logger.info("Alert cooldown active, skipping")
                return False
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse last_alert_ts: {e}")

    # Build alert
    drift_signals = metrics.get("drift_signals", [])
    accuracy = metrics.get("accuracy", 0)
    win_rate = metrics.get("win_rate", 0)
    sharpe = metrics.get("sharpe", 0)
    drift_score = metrics.get("drift_score", 0)

    title = "Model Drift Detected" if metrics.get("is_drifting") else "Model Performance Warning"
    color = 0xFF0000 if metrics.get("is_drifting") else 0xFFAA00

    description = (
        f"**Accuracy:** {accuracy:.2%}\n"
        f"**Win Rate:** {win_rate:.2%}\n"
        f"**Sharpe:** {sharpe:.2f}\n"
        f"**Drift Score:** {drift_score:.2f}/1.0\n"
        f"**Period:** {metrics.get('period_days', 7)} days\n"
        f"**Trades:** {metrics.get('signals_count', 0)}\n\n"
        f"**Issues:**\n" + "\n".join(f"• {s}" for s in drift_signals)
    )

    try:
        import httpx
        async with httpx.AsyncClient() as client:
            await client.post(
                DISCORD_WEBHOOK_URL,
                json={
                    "embeds": [{
                        "title": f"Model Drift Detected",
                        "description": description,
                        "color": color,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }]
                },
                timeout=10.0,
            )
        state["last_alert_ts"] = datetime.now(timezone.utc).isoformat()
        save_state(state)
        logger.info("Drift alert sent to Discord")
        return True
    except Exception as e:
        logger.warning(f"Failed to send drift alert: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Model drift monitoring")
    parser.add_argument("--days", type=int, default=7, help="Analysis window (days)")
    parser.add_argument("--alert", action="store_true", help="Test Discord alert")
    args = parser.parse_args()

    state = load_state()

    if args.alert:
        # Test alert
        metrics = {
            "accuracy": 0.42,
            "win_rate": 0.38,
            "sharpe": 0.3,
            "drift_score": 0.8,
            "is_drifting": True,
            "period_days": 7,
            "signals_count": 45,
            "drift_signals": ["Test alert", "Accuracy 42.00% < 45.00%"],
        }
        import asyncio
        asyncio.run(send_drift_alert(metrics, state))
        return

    # Analyze
    metrics = analyze_recent_performance(days=args.days)

    # Update state
    state["drift_score"] = metrics.get("drift_score", 0)
    state["last_analysis"] = datetime.now(timezone.utc).isoformat()
    save_state(state)

    # Print report
    print("\n" + "=" * 60)
    print("MODEL DRIFT MONITOR")
    print("=" * 60)

    if "error" in metrics:
        print(f"Status: {metrics['error']}")
    else:
        print(f"Period: Last {metrics['period_days']} days")
        print(f"Trades analyzed: {metrics['signals_count']}")
        print(f"Entries: {metrics['entries']} | Exits: {metrics['exits']}")
        print()
        print(f"Accuracy:     {metrics['accuracy']:.2%}")
        print(f"Win Rate:     {metrics['win_rate']:.2%}")
        print(f"Avg Signal:   {metrics['avg_signal']:.4f}")
        print(f"Sharpe:       {metrics['sharpe']:.2f}")
        print(f"Max Consec. Losses: {metrics['max_consecutive_losses']}")
        print()
        print(f"Drift Score:  {metrics['drift_score']:.2f}/1.0")
        print(f"Is Drifting:  {'YES' if metrics['is_drifting'] else 'NO'}")

        if metrics.get("drift_signals"):
            print("\nDrift Signals:")
            for s in metrics["drift_signals"]:
                print(f"  • {s}")

        if metrics.get("regime_distribution"):
            print("\nRegime Distribution:")
            for regime, count in metrics["regime_distribution"].items():
                print(f"  {regime}: {count}")

    print("=" * 60)

    # Auto-alert if drifting
    if metrics.get("is_drifting"):
        import asyncio
        asyncio.run(send_drift_alert(metrics, state))


if __name__ == "__main__":
    main()
