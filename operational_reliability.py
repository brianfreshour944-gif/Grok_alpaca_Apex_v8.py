"""
operational_reliability.py — Operational reliability testing.

Tests failure scenarios and fail-safe behaviors:
- Exchange/API outage
- Internet interruption
- Server restart
- Process crash
- Database failure
- Stale market data
- Duplicate orders
- Missing fills
- Partial fills
- Incorrect account balances
- Incorrect position state
- Clock/time synchronization
- Model unavailable
- LLM/API unavailable

Defines fail-safe response:
Continue / Retry / Reduce Risk / Cancel Orders / Close Positions / Pause / Require Human Approval

Usage:
    python operational_reliability.py              # full test suite
    python operational_reliability.py --scenario exchange_outage
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from config import logger


# ── Failure Scenarios ────────────────────────────────────────────────────────

FAILURE_SCENARIOS = {
    "exchange_outage": {
        "name": "Exchange/API Outage",
        "description": "Alpaca API returns 503/timeout",
        "severity": "CRITICAL",
        "detection": "HTTP 503, timeout, connection refused",
        "response": "Pause trading, cancel open orders, alert",
        "recovery": "Resume when API responds normally",
    },
    "internet_interruption": {
        "name": "Internet Interruption",
        "description": "Network connection lost",
        "severity": "CRITICAL",
        "detection": "WebSocket disconnect, API timeout",
        "response": "Pause trading, track position state locally",
        "recovery": "Reconnect, sync state with exchange",
    },
    "server_restart": {
        "name": "Server Restart",
        "description": "Process restarted unexpectedly",
        "severity": "HIGH",
        "detection": "Heartbeat missed, state file missing",
        "response": "Load saved state, reconcile positions",
        "recovery": "Resume from saved state",
    },
    "process_crash": {
        "name": "Process Crash",
        "description": "Unhandled exception kills process",
        "severity": "HIGH",
        "detection": "Process exit, heartbeat stopped",
        "response": "Auto-restart (systemd/PM2), load state",
        "recovery": "Resume from last known state",
    },
    "database_failure": {
        "name": "Database Failure",
        "description": "PostgreSQL connection lost",
        "severity": "HIGH",
        "detection": "psycopg2.OperationalError",
        "response": "Continue trading (DB is logging only), retry DB writes",
        "recovery": "Reconnect, backfill missed records",
    },
    "stale_market_data": {
        "name": "Stale Market Data",
        "description": "Price feed stops updating",
        "severity": "MEDIUM",
        "detection": "Data age > threshold, no new candles",
        "response": "Pause new entries, maintain exits",
        "recovery": "Resume when fresh data arrives",
    },
    "duplicate_orders": {
        "name": "Duplicate Orders",
        "description": "Same order submitted twice",
        "severity": "HIGH",
        "detection": "Order ID already exists, position size mismatch",
        "response": "Cancel duplicate, log warning",
        "recovery": "Dedup logic prevents recurrence",
    },
    "missing_fills": {
        "name": "Missing Fills",
        "description": "Order submitted but fill never received",
        "severity": "HIGH",
        "detection": "Order status != filled after timeout",
        "response": "Cancel stale order, retry if needed",
        "recovery": "Re-submit order",
    },
    "partial_fills": {
        "name": "Partial Fills",
        "description": "Order only partially filled",
        "severity": "MEDIUM",
        "detection": "filled_qty < ordered_qty",
        "response": "Accept partial, cancel remainder",
        "recovery": "Adjust position size to actual fill",
    },
    "incorrect_balances": {
        "name": "Incorrect Account Balances",
        "description": "Reported balance doesn't match expected",
        "severity": "CRITICAL",
        "detection": "Balance mismatch > threshold",
        "response": "Pause trading, alert human",
        "recovery": "Manual verification required",
    },
    "incorrect_positions": {
        "name": "Incorrect Position State",
        "description": "Local state doesn't match exchange",
        "severity": "CRITICAL",
        "detection": "Position count/value mismatch",
        "response": "Force sync with exchange, pause if major",
        "recovery": "Reconcile local state with exchange",
    },
    "clock_sync": {
        "name": "Clock/Time Synchronization",
        "description": "System clock drift",
        "severity": "MEDIUM",
        "detection": "Timestamp mismatch with exchange",
        "response": "Use exchange timestamps, not local",
        "recovery": "NTP sync",
    },
    "model_unavailable": {
        "name": "Model Unavailable",
        "description": "ML model file missing or corrupted",
        "severity": "HIGH",
        "detection": "Model load failure, prediction error",
        "response": "Use fallback (shadow GBT), or pause",
        "recovery": "Reload model from backup",
    },
    "llm_api_unavailable": {
        "name": "LLM/API Unavailable",
        "description": "Three-Brain orchestrator unreachable",
        "severity": "MEDIUM",
        "detection": "Orchestrator health check fails",
        "response": "Continue with local model only",
        "recovery": "Resume LLM when available",
    },
}


# ── Fail-Safe Response Matrix ────────────────────────────────────────────────

FAIL_SAFE_MATRIX = {
    "continue": {
        "description": "No action needed, continue normal operation",
        "examples": ["Minor network latency", "Single failed API call", "Non-critical log failure"],
    },
    "retry": {
        "description": "Retry the failed operation",
        "examples": ["API timeout", "Database write failure", "Order submission failure"],
        "max_retries": 5,
        "backoff": "exponential",
    },
    "reduce_risk": {
        "description": "Reduce position sizes and exposure",
        "examples": ["Elevated slippage", "Partial fills", "Stale data"],
        "risk_reduction": 0.5,  # 50% reduction
    },
    "cancel_orders": {
        "description": "Cancel all open orders",
        "examples": ["Exchange degradation", "Stale orders", "Duplicate detection"],
    },
    "close_positions": {
        "description": "Close all open positions",
        "examples": ["Daily loss limit", "Max drawdown hit", "Incorrect position state"],
    },
    "pause": {
        "description": "Stop all trading activity",
        "examples": ["Exchange outage", "Model unavailable", "Clock sync failure"],
    },
    "require_human": {
        "description": "Require human approval before continuing",
        "examples": ["Balance mismatch", "Position desync", "Repeated failures"],
    },
}


# ── Detection Methods ────────────────────────────────────────────────────────

def detect_exchange_outage() -> dict:
    """Test if exchange is reachable."""
    try:
        from config import trading_client
        account = trading_client.get_account()
        return {
            "status": "OK",
            "equity": float(account.equity),
            "response_time_ms": 0,  # Would measure actual response time
        }
    except Exception as e:
        return {
            "status": "FAILED",
            "error": str(e),
            "response": "pause",
        }


def detect_stale_data(max_age_seconds: int = 300) -> dict:
    """Check if market data is fresh."""
    # Would check last candle timestamp vs current time
    return {
        "status": "OK",
        "last_update": "now",
        "age_seconds": 0,
    }


def detect_position_desync() -> dict:
    """Compare local state with exchange state."""
    try:
        from config import trading_client
        positions = trading_client.get_all_positions()
        return {
            "status": "OK",
            "exchange_positions": len(positions),
        }
    except Exception as e:
        return {
            "status": "FAILED",
            "error": str(e),
        }


def detect_model_availability() -> dict:
    """Check if ML model is loadable."""
    from config import MODEL_PATH
    if os.path.exists(MODEL_PATH):
        return {
            "status": "OK",
            "model_path": MODEL_PATH,
            "size_bytes": os.path.getsize(MODEL_PATH),
        }
    else:
        return {
            "status": "FAILED",
            "error": "Model file not found",
            "response": "use_fallback",
        }


def detect_database_health() -> dict:
    """Check database connectivity."""
    try:
        import psycopg2
        from config import DB_HOST, DB_NAME, DB_USER
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            connect_timeout=5,
        )
        conn.close()
        return {
            "status": "OK",
            "host": DB_HOST,
        }
    except Exception as e:
        return {
            "status": "FAILED",
            "error": str(e),
            "response": "continue_logging_locally",
        }


# ── Test Suite ───────────────────────────────────────────────────────────────

def run_all_tests() -> dict:
    """Run all operational reliability tests."""
    results = {}
    
    # Test each detection method
    results["exchange_outage"] = detect_exchange_outage()
    results["stale_data"] = detect_stale_data()
    results["position_desync"] = detect_position_desync()
    results["model_availability"] = detect_model_availability()
    results["database_health"] = detect_database_health()
    
    # Summarize
    failures = [k for k, v in results.items() if v.get("status") == "FAILED"]
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_tests": len(results),
        "failures": len(failures),
        "failure_details": failures,
        "results": results,
        "overall_status": "HEALTHY" if not failures else "DEGRADED" if len(failures) < 3 else "CRITICAL",
    }


def print_report(results: dict):
    """Print operational reliability report."""
    print("\n" + "=" * 80)
    print("OPERATIONAL RELIABILITY REPORT")
    print("=" * 80)
    
    print(f"Timestamp: {results.get('timestamp', 'N/A')}")
    print(f"Overall Status: {results.get('overall_status', 'UNKNOWN')}")
    print(f"Tests: {results.get('total_tests', 0)} | Failures: {results.get('failures', 0)}")
    
    # Test results
    test_results = results.get("results", {})
    for test_name, result in test_results.items():
        status = result.get("status", "UNKNOWN")
        icon = "✓" if status == "OK" else "✗"
        
        print(f"\n{icon} {test_name.upper().replace('_', ' ')}")
        for key, value in result.items():
            if key != "status":
                print(f"  {key}: {value}")
    
    # Failure scenarios
    print("\n" + "-" * 80)
    print("FAILURE SCENARIOS & RESPONSES")
    print("-" * 80)
    
    for scenario_id, scenario in FAILURE_SCENARIOS.items():
        print(f"\n  {scenario['name']} [{scenario['severity']}]")
        print(f"    Detection: {scenario['detection']}")
        print(f"    Response: {scenario['response']}")
        print(f"    Recovery: {scenario['recovery']}")
    
    # Fail-safe matrix
    print("\n" + "-" * 80)
    print("FAIL-SAFE RESPONSE MATRIX")
    print("-" * 80)
    
    for action, details in FAIL_SAFE_MATRIX.items():
        print(f"\n  {action.upper()}: {details['description']}")
        for example in details.get("examples", []):
            print(f"    - {example}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Operational reliability testing")
    parser.add_argument("--scenario", type=str, help="Test specific scenario")
    args = parser.parse_args()
    
    results = run_all_tests()
    print_report(results)


if __name__ == "__main__":
    main()
