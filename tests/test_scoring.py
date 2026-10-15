"""Score formula, weight renormalization, and financial-safety rules."""

from __future__ import annotations

import pytest

from src.recommendations import (
    compute_recommendation_score,
    evidence_label,
    financial_status,
    is_actionable,
    lead_time_change,
    normalize_signed,
    sensitivity_profit_delta,
    weights_from_priority,
    ScenarioAssumptions,
)


def test_lead_time_change_handles_improvement_deterioration_and_zero_baseline():
    days, pct = lead_time_change(100, 80)
    assert days == pytest.approx(20)
    assert pct == pytest.approx(20)
    days, pct = lead_time_change(100, 120)
    assert days == pytest.approx(-20)
    assert pct == pytest.approx(-20)
    days, pct = lead_time_change(0, -5)
    assert days == pytest.approx(5)
    assert pct is None
    assert lead_time_change(None, 10) == (None, None)


def test_signed_normalization_is_neutral_at_zero():
    assert normalize_signed(0, 14) == pytest.approx(0.5)
    assert normalize_signed(14, 14) == pytest.approx(1.0)
    assert normalize_signed(-14, 14) == pytest.approx(0.0)


def test_score_formula_with_profit():
    result = compute_recommendation_score(
        speed_score=0.8,
        profit_score=0.2,
        risk_penalty=0.5,
        speed_weight=0.5,
        profit_weight=0.5,
        risk_weight=0.4,
    )
    assert result["renormalized"] is False
    assert result["score"] == pytest.approx(0.5 * 0.8 + 0.5 * 0.2 - 0.4 * 0.5)
    assert "not financially validated" in result["financial_impact_status"].lower()


def test_missing_profit_is_not_fabricated_and_weights_renormalize():
    result = compute_recommendation_score(
        speed_score=0.5,
        profit_score=None,
        risk_penalty=1.0,
        speed_weight=0.3,
        profit_weight=0.7,
        risk_weight=0.4,
    )
    assert result["renormalized"] is True
    assert result["speed_focused"] is True
    assert result["financial_impact_status"] == "Not quantified"
    assert "profit" not in result["components"]
    assert result["components"]["speed"]["applied_weight"] == pytest.approx(1.0)
    assert result["score"] == pytest.approx(0.5 - 0.4)


def test_sensitivity_profit_is_none_without_cost_inputs():
    assumptions = ScenarioAssumptions()
    assert sensitivity_profit_delta(
        days_saved=3,
        distance_delta_km=100,
        units=2,
        assumptions=assumptions,
        factory="Sugar Shack",
        current_factory="Wicked Choccy's",
    ) is None
    assert financial_status(None) == "Not quantified"


def test_sensitivity_profit_uses_only_entered_assumptions():
    assumptions = ScenarioAssumptions(delay_cost_per_order_day=10.0, freight_cost_per_1000km=50.0)
    delta = sensitivity_profit_delta(
        days_saved=2,
        distance_delta_km=1000,
        units=4,
        assumptions=assumptions,
        factory="Sugar Shack",
        current_factory="Wicked Choccy's",
    )
    # 2 days * $10 avoided, minus 1000 km * $50 / 1000.
    assert delta == pytest.approx(20 - 50)


def test_priority_slider_weights_sum_to_one():
    speed, profit = weights_from_priority(0)
    assert (speed, profit) == (1.0, 0.0)
    speed, profit = weights_from_priority(100)
    assert speed == pytest.approx(0.0)
    assert profit == pytest.approx(1.0)
    speed, profit = weights_from_priority(35)
    assert speed + profit == pytest.approx(1.0)


def test_unlearned_factory_claim_cannot_be_higher_evidence_or_actionable():
    evidence = evidence_label(
        sample_size=500,
        eligibility="eligible",
        factory_effect_learned=False,
        uses_unvalidated_assumption=False,
        extrapolation=False,
        model_disagreement=False,
    )
    assert evidence == "Requires validation"
    assert is_actionable(
        evidence=evidence,
        eligibility="eligible",
        days_saved=4,
        factory_effect_learned=False,
        uses_unvalidated_assumption=False,
    ) is False


def test_actionable_requires_learned_effect_confirmed_eligibility_and_gain():
    assert is_actionable(
        evidence="Higher evidence",
        eligibility="eligible",
        days_saved=2,
        factory_effect_learned=True,
        uses_unvalidated_assumption=False,
    )
    assert not is_actionable(
        evidence="Higher evidence",
        eligibility="requires_validation",
        days_saved=2,
        factory_effect_learned=True,
        uses_unvalidated_assumption=False,
    )
