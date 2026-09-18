"""Tests for operational_reliability.py — failure scenarios and fail-safe."""

import pytest
from operational_reliability import (
    FAILURE_SCENARIOS,
    FAIL_SAFE_MATRIX,
    detect_stale_data,
    detect_model_availability,
    run_all_tests,
)


class TestFailureScenarios:
    def test_all_scenarios_have_required_fields(self):
        for scenario_id, scenario in FAILURE_SCENARIOS.items():
            assert "name" in scenario, f"{scenario_id} missing name"
            assert "severity" in scenario, f"{scenario_id} missing severity"
            assert "response" in scenario, f"{scenario_id} missing response"
            assert "recovery" in scenario, f"{scenario_id} missing recovery"
            assert scenario["severity"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_minimum_scenario_count(self):
        assert len(FAILURE_SCENARIOS) >= 10


class TestFailSafeMatrix:
    def test_all_actions_have_description(self):
        for action, details in FAIL_SAFE_MATRIX.items():
            assert "description" in details
            assert "examples" in details
            assert len(details["examples"]) > 0

    def test_critical_actions_present(self):
        critical = {"pause", "close_positions", "require_human", "cancel_orders"}
        assert critical.issubset(set(FAIL_SAFE_MATRIX.keys()))


class TestDetectStaleData:
    def test_returns_ok(self):
        result = detect_stale_data()
        assert result["status"] == "OK"


class TestDetectModelAvailability:
    def test_missing_model(self, monkeypatch, tmp_path):
        monkeypatch.setattr("operational_reliability.os.path.exists", lambda p: False)
        result = detect_model_availability()
        assert result["status"] == "FAILED"

    def test_existing_model(self, monkeypatch, tmp_path):
        fake_path = tmp_path / "model.pkl"
        fake_path.write_text("fake")
        import config
        monkeypatch.setattr(config, "MODEL_PATH", str(fake_path))
        result = detect_model_availability()
        assert result["status"] == "OK"
        assert result["size_bytes"] > 0


class TestRunAllTests:
    def test_returns_summary(self):
        result = run_all_tests()
        assert "total_tests" in result
        assert "failures" in result
        assert "overall_status" in result
        assert result["overall_status"] in ("HEALTHY", "DEGRADED", "CRITICAL")

    def test_has_all_test_results(self):
        result = run_all_tests()
        assert "exchange_outage" in result["results"]
        assert "stale_data" in result["results"]
        assert "model_availability" in result["results"]
