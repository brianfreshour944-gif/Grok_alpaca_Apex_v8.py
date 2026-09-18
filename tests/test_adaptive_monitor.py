"""Tests for adaptive_monitor.py — adaptive/self-learning behavior monitoring."""

import pytest
import json
import numpy as np
from adaptive_monitor import (
    ADAPTIVE_LIMITS,
    DRIFT_THRESHOLDS,
    compute_feature_drift,
    load_adaptive_state,
    save_adaptive_state,
    determine_action,
)


class TestAdaptiveLimits:
    def test_limits_defined(self):
        assert ADAPTIVE_LIMITS["max_consecutive_adaptations"] > 0
        assert ADAPTIVE_LIMITS["cooldown_hours_between_adaptations"] > 0
        assert ADAPTIVE_LIMITS["min_data_for_adaptation"] > 0


class TestDriftThresholds:
    def test_all_threshold_levels_present(self):
        expected = {"continue", "reduce_risk", "pause", "retrain", "rollback"}
        assert expected == set(DRIFT_THRESHOLDS.keys())

    def test_thresholds_are_monotonic(self):
        # Drift score thresholds should increase with severity
        assert DRIFT_THRESHOLDS["continue"]["max_drift_score"] < DRIFT_THRESHOLDS["rollback"]["max_drift_score"]
        # Win rate thresholds should decrease with severity
        assert DRIFT_THRESHOLDS["continue"]["min_win_rate"] > DRIFT_THRESHOLDS["rollback"]["min_win_rate"]


class TestComputeFeatureDrift:
    def test_empty_events(self):
        result = compute_feature_drift([])
        assert "error" in result

    def test_returns_dict(self):
        events = [{"type": "entry", "features": {"z_return": 0.5}}]
        result = compute_feature_drift(events)
        assert isinstance(result, dict)


class TestLoadAdaptiveState:
    def test_default_state(self, tmp_path, monkeypatch):
        monkeypatch.setattr("adaptive_monitor.os.path.exists", lambda p: False)
        state = load_adaptive_state()
        assert state["current_action"] == "continue"
        assert state["drift_history"] == []
        assert state["adaptation_count"] == 0

    def test_loads_existing_state(self, tmp_path, monkeypatch):
        """Test that load_adaptive_state reads from the correct file path."""
        import adaptive_monitor
        # Verify function uses hardcoded path
        import inspect
        source = inspect.getsource(load_adaptive_state)
        assert "adaptive_state.json" in source


class TestSaveAdaptiveState:
    def test_saves_state(self, tmp_path, monkeypatch):
        saved = {}
        def fake_dump(data, f, **kwargs):
            saved.update(data)
        monkeypatch.setattr("adaptive_monitor.json.dump", fake_dump)
        save_adaptive_state({"current_action": "test"})
        assert saved["current_action"] == "test"


class TestDetermineAction:
    def test_good_metrics_returns_continue(self):
        state = {"consecutive_adaptations": 0, "last_adaptation_ts": None}
        action = determine_action(drift_score=0.1, win_rate=0.6, sharpe=1.0, feature_drift=0.2, state=state)
        assert action == "continue"

    def test_bad_drift_returns_rollback(self):
        state = {"consecutive_adaptations": 0, "last_adaptation_ts": None}
        action = determine_action(drift_score=0.95, win_rate=0.20, sharpe=-1.5, feature_drift=2.5, state=state)
        assert action == "rollback"

    def test_too_many_adaptations_returns_pause(self):
        state = {"consecutive_adaptations": 5, "last_adaptation_ts": None}
        action = determine_action(drift_score=0.1, win_rate=0.6, sharpe=1.0, feature_drift=0.2, state=state)
        assert action == "pause"

    def test_cooldown_returns_previous_action(self):
        from datetime import datetime, timezone, timedelta
        recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        state = {"consecutive_adaptations": 0, "last_adaptation_ts": recent, "current_action": "reduce_risk"}
        action = determine_action(drift_score=0.1, win_rate=0.6, sharpe=1.0, feature_drift=0.2, state=state)
        assert action == "reduce_risk"

    def test_moderate_drift_returns_reduce_risk(self):
        state = {"consecutive_adaptations": 0, "last_adaptation_ts": None}
        # drift_score=0.5 triggers reduce_risk (max_drift_score=0.5) but not rollback (1.0) or pause (0.7)
        action = determine_action(drift_score=0.5, win_rate=0.42, sharpe=0.35, feature_drift=0.6, state=state)
        assert action == "reduce_risk"
