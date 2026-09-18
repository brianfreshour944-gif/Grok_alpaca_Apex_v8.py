"""Tests for performance_dashboard.py — unified performance metrics."""

import pytest
import numpy as np
from performance_dashboard import compute_all_metrics


def _make_trade_events(n=10, win_rate=0.6):
    """Generate synthetic trade events."""
    events = []
    for i in range(n):
        is_win = np.random.random() < win_rate
        pnl = np.random.uniform(0.005, 0.03) if is_win else -np.random.uniform(0.005, 0.02)
        entry_ts = f"2026-01-01T{i:02d}:00:00Z"
        exit_ts = f"2026-01-01T{i:02d}:30:00Z"
        events.append({
            "type": "entry", "symbol": "BTC", "ts": entry_ts,
            "price": 50000, "signal": 0.7, "regime": "normal", "atr_pct": 2.0,
        })
        events.append({
            "type": "exit", "symbol": "BTC", "ts": exit_ts,
            "pnl_pct": pnl, "held_hours": 0.5, "exit_reason": "Trailing Stop",
        })
    return events


class TestComputeAllMetrics:
    def test_empty_events_returns_error(self):
        result = compute_all_metrics([])
        assert "error" in result

    def test_entries_only_returns_error(self):
        events = [{"type": "entry", "symbol": "BTC", "ts": "2026-01-01T00:00:00Z"}]
        result = compute_all_metrics(events)
        assert "error" in result

    def test_exits_only_returns_error(self):
        events = [{"type": "exit", "symbol": "BTC", "ts": "2026-01-01T00:00:00Z", "pnl_pct": 0.01}]
        result = compute_all_metrics(events)
        assert "error" in result

    def test_no_matched_trades(self):
        events = [
            {"type": "entry", "symbol": "BTC", "ts": "2026-01-01T00:00:00Z"},
            {"type": "exit", "symbol": "ETH", "ts": "2026-01-01T01:00:00Z", "pnl_pct": 0.01},
        ]
        result = compute_all_metrics(events)
        assert "error" in result

    def test_valid_trades_returns_metrics(self):
        events = _make_trade_events(10)
        result = compute_all_metrics(events)
        assert "error" not in result
        assert result["n_trades"] == 10

    def test_returns_section(self):
        events = _make_trade_events(10)
        result = compute_all_metrics(events)
        ret = result["returns"]
        assert "total_return" in ret
        assert "avg_trade" in ret
        assert "expectancy" in ret

    def test_risk_section(self):
        events = _make_trade_events(10)
        result = compute_all_metrics(events)
        risk = result["risk"]
        assert "max_drawdown" in risk
        assert "var_95" in risk
        assert "cvar_95" in risk

    def test_risk_adjusted_section(self):
        events = _make_trade_events(10)
        result = compute_all_metrics(events)
        ra = result["risk_adjusted"]
        assert "sharpe" in ra
        assert "sortino" in ra
        assert "calmar" in ra

    def test_trade_quality_section(self):
        events = _make_trade_events(10)
        result = compute_all_metrics(events)
        tq = result["trade_quality"]
        assert 0 <= tq["win_rate"] <= 1
        assert tq["max_consecutive_losses"] >= 0

    def test_trading_efficiency_section(self):
        events = _make_trade_events(10)
        result = compute_all_metrics(events)
        te = result["trading_efficiency"]
        assert te["trades_per_day"] > 0
        assert te["avg_holding_time_hours"] >= 0

    def test_regime_stability(self):
        events = _make_trade_events(10)
        result = compute_all_metrics(events)
        assert "regime_stability" in result

    def test_exit_analysis(self):
        events = _make_trade_events(10)
        result = compute_all_metrics(events)
        assert "exit_analysis" in result
