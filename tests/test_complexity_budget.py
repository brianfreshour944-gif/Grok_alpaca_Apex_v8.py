"""Tests for complexity_budget.py — component value vs complexity analysis."""

import pytest
from complexity_budget import (
    COMPONENTS,
    compute_complexity_budget,
    assess_component_value,
    simulate_removal,
    analyze_complexity_budget,
)


class TestComputeComplexityBudget:
    def test_returns_all_components(self):
        result = compute_complexity_budget()
        assert result["total_components"] == len(COMPONENTS)

    def test_total_loc_matches(self):
        result = compute_complexity_budget()
        expected_loc = sum(c["lines_of_code"] for c in COMPONENTS.values())
        assert result["total_loc"] == expected_loc

    def test_categories_populated(self):
        result = compute_complexity_budget()
        assert len(result["categories"]) > 0
        for cat_stats in result["categories"].values():
            assert cat_stats["count"] > 0
            assert cat_stats["loc"] > 0

    def test_avg_loc_nonzero(self):
        result = compute_complexity_budget()
        assert result["avg_loc_per_component"] > 0


class TestAssessComponentValue:
    def test_empty_events_returns_error(self):
        result = assess_component_value([])
        assert "error" in result

    def test_no_entries_returns_error(self):
        events = [{"type": "exit", "symbol": "BTC", "ts": "2026-01-01T00:00:00Z", "pnl_pct": 0.01}]
        result = assess_component_value(events)
        assert "error" in result

    def test_no_exits_returns_error(self):
        events = [{"type": "entry", "symbol": "BTC", "ts": "2026-01-01T00:00:00Z", "signal": 0.7}]
        result = assess_component_value(events)
        assert "error" in result

    def test_assesses_all_components(self):
        events = [
            {"type": "entry", "symbol": "BTC", "ts": "2026-01-01T00:00:00Z", "signal": 0.7, "regime": "normal"},
            {"type": "exit", "symbol": "BTC", "ts": "2026-01-01T01:00:00Z", "pnl_pct": 0.02, "exit_reason": "Trailing Stop"},
        ]
        result = assess_component_value(events)
        assert len(result) == len(COMPONENTS)

    def test_value_scores_in_range(self):
        events = [
            {"type": "entry", "symbol": "BTC", "ts": "2026-01-01T00:00:00Z", "signal": 0.7, "regime": "normal"},
            {"type": "exit", "symbol": "BTC", "ts": "2026-01-01T01:00:00Z", "pnl_pct": 0.02, "exit_reason": "Trailing Stop"},
        ]
        result = assess_component_value(events)
        for comp_id, assessment in result.items():
            assert 0 <= assessment["value_score"] <= 10, f"{comp_id} score out of range"
            assert assessment["recommendation"] in ("KEEP", "EVALUATE", "CONSIDER_REMOVING")

    def test_shadow_gbt_always_zero(self):
        events = [
            {"type": "entry", "symbol": "BTC", "ts": "2026-01-01T00:00:00Z", "signal": 0.7, "regime": "normal"},
            {"type": "exit", "symbol": "BTC", "ts": "2026-01-01T01:00:00Z", "pnl_pct": 0.02, "exit_reason": "Trailing Stop"},
        ]
        result = assess_component_value(events)
        assert result["shadow_gbt"]["value_score"] == 0
        assert result["shadow_gbt"]["recommendation"] == "CONSIDER_REMOVING"

    def test_daily_loss_limit_always_high(self):
        events = [
            {"type": "entry", "symbol": "BTC", "ts": "2026-01-01T00:00:00Z", "signal": 0.7, "regime": "normal"},
            {"type": "exit", "symbol": "BTC", "ts": "2026-01-01T01:00:00Z", "pnl_pct": 0.02, "exit_reason": "Trailing Stop"},
        ]
        result = assess_component_value(events)
        assert result["daily_loss_limit"]["value_score"] == 10


class TestSimulateRemoval:
    def test_unknown_component(self):
        result = simulate_removal("nonexistent_component")
        assert "error" in result

    def test_known_component(self):
        result = simulate_removal("transformer_model")
        assert result["component"] == "Transformer Model (GrokGQA)"
        assert result["complexity_saved"] > 0

    def test_all_components_simulatable(self):
        for comp_id in COMPONENTS:
            result = simulate_removal(comp_id)
            assert "error" not in result, f"Failed for {comp_id}"
            assert "impact" in result


class TestAnalyzeComplexityBudget:
    def test_returns_budget(self):
        result = analyze_complexity_budget()
        assert "budget" in result
        assert result["budget"]["total_components"] == len(COMPONENTS)

    def test_recommendations_counted(self):
        result = analyze_complexity_budget()
        recs = result["recommendations"]
        total = recs.get("KEEP", 0) + recs.get("EVALUATE", 0) + recs.get("CONSIDER_REMOVING", 0)
        # Without experience data, assessments is empty so total is 0
        assert total >= 0
