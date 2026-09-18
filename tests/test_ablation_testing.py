"""Tests for ablation_testing.py — ablation testing for multi-component architecture."""

import pytest
import numpy as np
from ablation_testing import (
    COMPONENTS,
    simulate_system,
    run_ablation,
    analyze_impact,
)


def _make_trade_events(n=20):
    """Generate synthetic trade events with realistic structure."""
    events = []
    for i in range(n):
        pnl = np.random.uniform(-0.02, 0.03)
        entry_ts = f"2026-01-01T{i:02d}:00:00Z"
        exit_ts = f"2026-01-01T{i:02d}:30:00Z"
        events.append({
            "type": "entry",
            "symbol": "BTC",
            "ts": entry_ts,
            "signal": np.random.uniform(0.3, 0.9),
            "price": 50000,
            "regime": np.random.choice(["normal", "wild", "quiet"]),
            "atr_pct": np.random.uniform(1.0, 4.0),
            "features": {f"feat_{j}": np.random.randn() for j in range(5)},
        })
        events.append({
            "type": "exit",
            "symbol": "BTC",
            "ts": exit_ts,
            "pnl_pct": pnl,
            "held_hours": np.random.uniform(0.1, 4.0),
            "exit_reason": np.random.choice(["Trailing Stop", "Stop loss", "Time-Decay", "Signal weak"]),
        })
    return events


class TestSimulateSystem:
    def test_empty_events_returns_error(self):
        result = simulate_system([])
        assert "error" in result

    def test_no_entries_returns_error(self):
        events = [{"type": "exit", "symbol": "BTC", "ts": "2026-01-01T00:00:00Z", "pnl_pct": 0.01}]
        result = simulate_system(events)
        assert "error" in result

    def test_full_system_returns_metrics(self):
        events = _make_trade_events(10)
        result = simulate_system(events, component="full")
        assert "error" not in result
        assert result["n_trades"] > 0
        assert 0 <= result["win_rate"] <= 1

    def test_transformer_ablation(self):
        events = _make_trade_events(20)
        full = simulate_system(events, component="full")
        no_tf = simulate_system(events, component="transformer")
        # Both should return valid results
        assert "error" not in full
        assert "error" not in no_tf

    def test_no_trailing_stop(self):
        events = _make_trade_events(20)
        result = simulate_system(events, use_trailing=False)
        assert "error" not in result

    def test_no_time_decay(self):
        events = _make_trade_events(20)
        result = simulate_system(events, use_time_decay=False)
        assert "error" not in result

    def test_feature_mask(self):
        events = _make_trade_events(10)
        result = simulate_system(events, feature_mask=["feat_0", "feat_1"])
        assert "error" not in result

    def test_metrics_in_valid_range(self):
        events = _make_trade_events(20)
        result = simulate_system(events)
        assert 0 <= result["win_rate"] <= 1
        assert result["max_drawdown"] <= 0


class TestRunAblation:
    def test_returns_all_ablation_results(self):
        events = _make_trade_events(20)
        result = run_ablation(events)
        expected_keys = [
            "full_system", "no_transformer", "no_regime", "no_trailing_stop",
            "no_time_decay", "no_kelly", "no_momentum", "no_volatility",
            "no_microstructure", "no_efficiency",
        ]
        for key in expected_keys:
            assert key in result, f"Missing ablation key: {key}"

    def test_each_has_metrics(self):
        events = _make_trade_events(20)
        result = run_ablation(events)
        for key, entry in result.items():
            assert "metrics" in entry, f"{key} missing metrics"
            assert "component" in entry
            assert "description" in entry


class TestAnalyzeImpact:
    def test_no_baseline_returns_error(self):
        result = analyze_impact({})
        assert "error" in result

    def test_valid_analysis(self):
        events = _make_trade_events(20)
        ablation = run_ablation(events)
        result = analyze_impact(ablation)
        assert "baseline" in result
        assert "impacts" in result
        assert "valuable_components" in result
        assert "removable_components" in result

    def test_baseline_excluded_from_impacts(self):
        events = _make_trade_events(20)
        ablation = run_ablation(events)
        result = analyze_impact(ablation)
        assert "full_system" not in result["impacts"]

    def test_verdicts_are_valid(self):
        events = _make_trade_events(20)
        ablation = run_ablation(events)
        result = analyze_impact(ablation)
        for key, imp in result["impacts"].items():
            assert imp["verdict"] in ("KEEP", "REMOVE")
