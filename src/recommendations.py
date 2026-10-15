"""Transparent recommendation scores, evidence labels, and risk flags."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    ADVISORY_TEXT,
    DEFAULT_RISK_WEIGHT,
    DEFAULT_REFERENCE_DOW,
    DEFAULT_REFERENCE_MONTH,
    MIN_SAMPLE_STABLE,
    MIN_SAMPLE_WARNING,
    MODEL_VERSION,
    PROFIT_CAP_DOLLARS,
    RAW_DISAGREEMENT_DAYS,
    REASON_TEXT,
    SPEED_CAP_DAYS,
)


@dataclass
class ScenarioAssumptions:
    """User-controlled weights and labeled what-if assumptions.

    Distance and cost fields are assumptions. They are never described as
    historically validated freight or factory effects.
    """

    speed_weight: float = 0.70
    profit_weight: float = 0.30
    risk_weight: float = DEFAULT_RISK_WEIGHT
    distance_enabled: bool = False
    days_per_1000km: float = 1.0
    coordinate_source: str = "none"  # none, manual, state_centroid, uploaded
    centroid_acknowledged: bool = False
    delay_cost_per_order_day: float | None = None
    freight_cost_per_1000km: float | None = None
    factory_handling_days: dict[str, float] = field(default_factory=dict)
    factory_cost_delta_per_unit: dict[str, float] = field(default_factory=dict)
    use_factory_cost: bool = False
    reference_month: int = DEFAULT_REFERENCE_MONTH
    reference_dow: int = DEFAULT_REFERENCE_DOW
    speed_cap_days: float = SPEED_CAP_DAYS
    profit_cap_dollars: float = PROFIT_CAP_DOLLARS
    rank_by_scenario: bool = False
    factory_effect_learned: bool = False
    target_quality_penalty: float = 0.85
    model_name: str = "unavailable"
    model_version: str = MODEL_VERSION
    disagreement_threshold_days: float = RAW_DISAGREEMENT_DAYS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def weights_from_priority(profit_priority: float) -> tuple[float, float]:
    """Map a 0–100 profit-protection priority to weights that sum to 1."""
    profit_weight = float(np.clip(profit_priority, 0.0, 100.0)) / 100.0
    return 1.0 - profit_weight, profit_weight


def normalize_signed(value: float, cap: float) -> float:
    if cap <= 0:
        raise ValueError("Normalization cap must be positive.")
    return float(np.clip((value + cap) / (2.0 * cap), 0.0, 1.0))


def lead_time_change(current_days: float | None, alternative_days: float | None) -> tuple[float | None, float | None]:
    """Return days saved and percent change. Percent is undefined when current days is not positive."""
    if current_days is None or alternative_days is None:
        return None, None
    if isinstance(current_days, float) and np.isnan(current_days):
        return None, None
    if isinstance(alternative_days, float) and np.isnan(alternative_days):
        return None, None
    days = float(current_days) - float(alternative_days)
    if float(current_days) <= 0:
        return days, None
    return days, days / float(current_days) * 100.0


def scenario_adjustments_active(assumptions: ScenarioAssumptions) -> bool:
    handling = assumptions.factory_handling_days or {}
    has_handling = any(abs(float(value)) > 1e-9 for value in handling.values())
    return bool(assumptions.distance_enabled or has_handling)


def profit_inputs_present(assumptions: ScenarioAssumptions) -> bool:
    if assumptions.delay_cost_per_order_day is not None:
        return True
    if assumptions.freight_cost_per_1000km is not None:
        return True
    if assumptions.use_factory_cost and any(abs(float(v)) > 1e-9 for v in (assumptions.factory_cost_delta_per_unit or {}).values()):
        return True
    return False


def sensitivity_profit_delta(
    *,
    days_saved: float | None,
    distance_delta_km: float | None,
    units: float,
    assumptions: ScenarioAssumptions,
    factory: str,
    current_factory: str,
) -> float | None:
    """Dollar sensitivity from user assumptions only. None means not quantified."""
    if not profit_inputs_present(assumptions):
        return None
    total = 0.0
    used = False
    if assumptions.delay_cost_per_order_day is not None and days_saved is not None:
        total += float(days_saved) * float(assumptions.delay_cost_per_order_day)
        used = True
    if assumptions.freight_cost_per_1000km is not None and distance_delta_km is not None and np.isfinite(distance_delta_km):
        total += -(float(distance_delta_km) / 1000.0) * float(assumptions.freight_cost_per_1000km)
        used = True
    if assumptions.use_factory_cost:
        deltas = assumptions.factory_cost_delta_per_unit or {}
        delta = float(deltas.get(factory, 0.0)) - float(deltas.get(current_factory, 0.0))
        total += -delta * float(units)
        used = True
    if not used:
        return None
    return float(total)


def financial_status(profit_delta: float | None) -> str:
    if profit_delta is None:
        return "Not quantified"
    return "User-assumption sensitivity — not financially validated"


def compute_recommendation_score(
    speed_score: float | None,
    profit_score: float | None,
    risk_penalty: float,
    speed_weight: float,
    profit_weight: float,
    risk_weight: float,
) -> dict[str, Any]:
    """Score = available weighted components minus risk penalty.

    If profit impact is missing, the profit weight is dropped and the remaining
    positive weight is renormalized. A profit score is never fabricated.
    """
    components: dict[str, Any] = {}
    available: list[tuple[str, float, float]] = []
    if speed_score is not None:
        available.append(("speed", max(float(speed_weight), 0.0), float(speed_score)))
    if profit_score is not None:
        available.append(("profit", max(float(profit_weight), 0.0), float(profit_score)))
    weight_sum = sum(weight for _, weight, _ in available)
    profit_missing = profit_score is None and float(profit_weight) > 0
    positive = 0.0
    if weight_sum > 0:
        for name, weight, score in available:
            applied = weight / weight_sum if weight_sum else 0.0
            contribution = applied * score
            components[name] = {
                "requested_weight": weight,
                "applied_weight": applied,
                "score": score,
                "contribution": contribution,
            }
            positive += contribution
    elif speed_score is not None:
        components["speed"] = {
            "requested_weight": float(speed_weight),
            "applied_weight": 0.0,
            "score": float(speed_score),
            "contribution": 0.0,
        }
    risk_contribution = -float(risk_weight) * float(risk_penalty)
    return {
        "score": positive + risk_contribution,
        "renormalized": bool(profit_missing),
        "speed_focused": profit_score is None,
        "components": components,
        "risk_penalty": float(risk_penalty),
        "risk_weight": float(risk_weight),
        "risk_contribution": risk_contribution,
        "financial_impact_status": financial_status(0.0 if profit_score is not None else None),
    }


def risk_components(
    *,
    sample_size: int,
    eligibility: str,
    target_quality_penalty: float,
    uses_unvalidated_assumption: bool,
    extrapolation: bool,
    model_disagreement: bool,
    profit_weight: float,
    profit_available: bool,
    missing_coordinates: bool = False,
) -> tuple[float, list[str]]:
    flags: list[str] = []
    if sample_size < MIN_SAMPLE_WARNING:
        sample_penalty = 1.0
        flags.append("LOW_SAMPLE")
    elif sample_size < MIN_SAMPLE_STABLE:
        sample_penalty = 0.55
        flags.append("LIMITED_SAMPLE")
    elif sample_size < 300:
        sample_penalty = 0.20
    else:
        sample_penalty = 0.0

    if eligibility == "eligible" or eligibility == "confirmed_current":
        eligibility_penalty = 0.0
    elif eligibility == "requires_validation":
        eligibility_penalty = 1.0
        flags.append("ELIGIBILITY_UNVERIFIED")
    elif eligibility == "ineligible":
        eligibility_penalty = 1.0
        flags.append("INELIGIBLE")
    else:
        eligibility_penalty = 1.0
        flags.append("ELIGIBILITY_UNVERIFIED")

    assumption_penalty = 0.75 if uses_unvalidated_assumption else 0.0
    if uses_unvalidated_assumption:
        flags.append("UNVALIDATED_ASSUMPTION")
    extra = 0.0
    if extrapolation:
        extra += 0.35
        flags.append("EXTRAPOLATION")
    if model_disagreement:
        extra += 0.25
        flags.append("MODEL_DISAGREEMENT")
    if profit_weight > 0 and not profit_available:
        extra += 0.15
        flags.append("PROFIT_NOT_QUANTIFIED")
    if missing_coordinates and uses_unvalidated_assumption:
        extra += 0.15
        flags.append("MISSING_COORDINATES")
    if target_quality_penalty >= 0.5:
        flags.append("DATE_ANOMALY_DOMINATES_TARGET")
    penalty = (
        0.25 * sample_penalty
        + 0.30 * eligibility_penalty
        + 0.25 * float(target_quality_penalty)
        + 0.20 * assumption_penalty
        + extra
    )
    return float(min(1.0, penalty)), flags


def evidence_label(
    *,
    sample_size: int,
    eligibility: str,
    factory_effect_learned: bool,
    uses_unvalidated_assumption: bool,
    extrapolation: bool,
    model_disagreement: bool,
) -> str:
    """Evidence for a factory-reassignment claim.

    Higher evidence is reserved for a learned factory effect with confirmed
    eligibility and a stable sample. Swapping an unlearned factory label never
    qualifies.
    """
    if eligibility in {"requires_validation", "ineligible", "unknown"}:
        return "Requires validation"
    if uses_unvalidated_assumption or not factory_effect_learned:
        return "Requires validation"
    if sample_size < MIN_SAMPLE_STABLE or extrapolation or model_disagreement:
        return "Limited evidence"
    return "Higher evidence"


def is_actionable(
    *,
    evidence: str,
    eligibility: str,
    days_saved: float | None,
    factory_effect_learned: bool,
    uses_unvalidated_assumption: bool,
) -> bool:
    return bool(
        factory_effect_learned
        and eligibility == "eligible"
        and evidence == "Higher evidence"
        and days_saved is not None
        and days_saved > 0
        and not uses_unvalidated_assumption
    )


def explain_recommendation(record: dict[str, Any]) -> str:
    current = record.get("current_factory")
    candidate = record.get("candidate_factory")
    product = record.get("product_name")
    region = record.get("region")
    ship_mode = record.get("ship_mode")
    predicted = record.get("predicted_current_lead_time_days")
    sentences = [
        f"{candidate} is an alternative to {current} for {product} going to {region} by {ship_mode}."
    ]
    if not record.get("factory_effect_learned"):
        pred_text = "unavailable" if predicted is None else f"{float(predicted):.1f} days"
        sentences.append(
            "Origin factory is not in the historical orders, so the model assigns "
            f"{pred_text} to every factory in this context. That is not evidence of a faster factory."
        )
    days = record.get("days_saved")
    if days is not None and days <= 0 and not record.get("scenario_applied"):
        sentences.append("Model-supported lead-time change is 0 days. Do not reassign on this evidence.")
    scenario_days = record.get("scenario_days_saved")
    if record.get("scenario_applied") and scenario_days is not None:
        if scenario_days > 0.05:
            direction = "faster"
        elif scenario_days < -0.05:
            direction = "slower"
        else:
            direction = "unchanged"
        sentences.append(
            f"Under the stated assumptions the scenario is {abs(float(scenario_days)):.1f} days {direction}. "
            "That figure is not a validated model result."
        )
    if record.get("financial_impact_status") == "Not quantified":
        sentences.append("Profit impact is not quantified because freight and factory costs were not supplied.")
    else:
        sentences.append("Any dollar figure is a user-assumption sensitivity, not a financially validated profit improvement.")
    sentences.append(ADVISORY_TEXT)
    return " ".join(sentences)


def reason_glossary() -> pd.DataFrame:
    return pd.DataFrame([{"code": code, "meaning": text} for code, text in REASON_TEXT.items()])
