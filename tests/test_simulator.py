"""Scenario fairness, eligibility, and the ban on invented factory effects."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import FACTORIES
from src.data_loader import default_eligibility, normalize_product_name, reference_mapping
from src.geo import haversine_km
from src.recommendations import ScenarioAssumptions
from src.simulator import build_recommendations, simulate_scenarios
from tests.conftest import frame_from_rows, raw_order


def _predict_constant(frame: pd.DataFrame) -> np.ndarray:
    assert "origin_factory" not in frame.columns
    assert frame["ship_mode"].nunique() == 1
    assert frame["region"].nunique() == 1
    return np.full(len(frame), 1274.0)


def _predict_learned(frame: pd.DataFrame) -> np.ndarray:
    return np.where(frame["origin_factory"].eq("Sugar Shack"), 10.0, 12.0).astype(float)


def test_priority_slider_reorders_conflicting_cost_assumptions():
    from src.config import EXAMPLE_COST_DELTA, EXAMPLE_DELAY_COST, EXAMPLE_HANDLING_DAYS

    def ranked(speed: float, profit: float) -> list[str]:
        assumptions = ScenarioAssumptions(
            speed_weight=speed,
            profit_weight=profit,
            delay_cost_per_order_day=EXAMPLE_DELAY_COST,
            use_factory_cost=True,
            factory_handling_days=EXAMPLE_HANDLING_DAYS,
            factory_cost_delta_per_unit=EXAMPLE_COST_DELTA,
            rank_by_scenario=True,
            factory_effect_learned=False,
            target_quality_penalty=0.85,
        )
        result = simulate_scenarios(
            product_name="Wonka Bar - Milk Chocolate",
            region="Interior",
            ship_mode="Standard Class",
            units=4,
            division="Chocolate",
            state="Texas",
            country="United States",
            mapping=reference_mapping(),
            eligibility=default_eligibility(),
            factories=pd.DataFrame(FACTORIES),
            assumptions=assumptions,
            predict_fn=_predict_constant,
            historical_n=80,
        )
        alts = result["rows"].loc[result["rows"]["role"] == "alternative"]
        return alts.sort_values("recommendation_score", ascending=False)["candidate_factory"].tolist()

    speed_first = ranked(1.0, 0.0)
    profit_first = ranked(0.0, 1.0)
    assert speed_first[0] != profit_first[0]
    assert speed_first[0] == "Lot's O' Nuts"
    assert profit_first[0] == "Sugar Shack"


def test_haversine_is_symmetric_and_positive():
    forward = haversine_km(32.881893, -111.768036, 32.076176, -81.088371)
    backward = haversine_km(32.076176, -81.088371, 32.881893, -111.768036)
    assert forward == pytest.approx(backward)
    assert 2500 < forward < 3200
    one_degree = haversine_km(0, 0, 0, 1)
    assert one_degree == pytest.approx(111.19, abs=0.2)


def test_factory_swap_does_not_change_unlearned_prediction():
    assumptions = ScenarioAssumptions(factory_effect_learned=False, model_name="constant")
    result = simulate_scenarios(
        product_name="Wonka Bar - Milk Chocolate",
        region="Interior",
        ship_mode="Standard Class",
        units=3,
        division="Chocolate",
        state="Texas",
        country="United States",
        mapping=reference_mapping(),
        eligibility=default_eligibility(),
        factories=pd.DataFrame(FACTORIES),
        assumptions=assumptions,
        predict_fn=_predict_constant,
        historical_n=120,
        historical_median=1274,
        historical_gross_profit=500,
    )
    rows = result["rows"]
    assert result["meta"]["fairness"]["model_predictions_factory_invariant"]
    assert rows["predicted_alternative_lead_time_days"].nunique() == 1
    alternatives = rows.loc[rows["role"] == "alternative"]
    assert (alternatives["days_saved"] == 0).all()
    assert (alternatives["lead_time_change_pct"] == 0).all()
    assert "Wicked Choccy's" not in set(alternatives["candidate_factory"])
    assert set(alternatives["eligibility"]) == {"requires_validation"}
    assert (alternatives["financial_impact_status"] == "Not quantified").all()
    assert (alternatives["actionable"] == False).all()  # noqa: E712
    assert result["meta"]["decision"] == "retain_current"
    assert "customer_id" not in rows.columns


def test_ineligible_factory_is_excluded_and_negative_scenario_is_kept():
    eligibility = default_eligibility()
    key = normalize_product_name("Wonka Bar - Milk Chocolate")
    eligibility.loc[
        (eligibility["product_name_key"] == key) & (eligibility["factory"] == "Sugar Shack"),
        "status",
    ] = "ineligible"
    eligibility.loc[
        (eligibility["product_name_key"] == key) & (eligibility["factory"] == "Secret Factory"),
        "status",
    ] = "eligible"
    assumptions = ScenarioAssumptions(
        factory_effect_learned=False,
        factory_handling_days={"Secret Factory": 3.0, "Wicked Choccy's": 0.0},
        rank_by_scenario=True,
        delay_cost_per_order_day=8.0,
    )
    result = simulate_scenarios(
        product_name="Wonka Bar - Milk Chocolate",
        region="Atlantic",
        ship_mode="Second Class",
        units=4,
        division="Chocolate",
        state="New York",
        country="United States",
        mapping=reference_mapping(),
        eligibility=eligibility,
        factories=pd.DataFrame(FACTORIES),
        assumptions=assumptions,
        predict_fn=_predict_constant,
        historical_n=40,
    )
    rows = result["rows"]
    assert "Sugar Shack" not in set(rows["candidate_factory"])
    secret = rows.loc[rows["candidate_factory"] == "Secret Factory"].iloc[0]
    assert secret["scenario_days_saved"] == pytest.approx(-3.0)
    assert secret["scenario_change_pct"] < 0
    assert "PREDICTED_DETERIORATION" in secret["reason_codes"]
    assert secret["financial_impact_status"].startswith("User-assumption")
    assert secret["sensitivity_profit_delta"] == pytest.approx(-24.0)
    assert secret["actionable"] == False  # noqa: E712


def test_learned_factory_can_change_prediction():
    assumptions = ScenarioAssumptions(factory_effect_learned=True, model_name="factory-aware")
    result = simulate_scenarios(
        product_name="Wonka Bar - Milk Chocolate",
        region="Gulf",
        ship_mode="First Class",
        units=2,
        division="Chocolate",
        state="Florida",
        country="United States",
        mapping=reference_mapping(),
        eligibility=default_eligibility(),
        factories=pd.DataFrame(FACTORIES),
        assumptions=assumptions,
        predict_fn=_predict_learned,
        feature_columns=[
            "ship_mode",
            "region",
            "division",
            "product_name",
            "state",
            "country",
            "units",
            "order_month",
            "order_dow",
            "order_quarter",
            "origin_factory",
        ],
        historical_n=80,
    )
    rows = result["rows"]
    current = rows.loc[rows["role"] == "current"].iloc[0]
    sugar = rows.loc[rows["candidate_factory"] == "Sugar Shack"].iloc[0]
    assert current["predicted_alternative_lead_time_days"] == pytest.approx(12)
    assert sugar["predicted_alternative_lead_time_days"] == pytest.approx(10)
    assert sugar["days_saved"] == pytest.approx(2)
    assert sugar["lead_time_change_pct"] == pytest.approx(2 / 12 * 100)


def test_distance_scenario_is_relative_to_current_and_labeled(monkeypatch):
    assumptions = ScenarioAssumptions(
        distance_enabled=True,
        coordinate_source="manual",
        days_per_1000km=1.0,
        rank_by_scenario=True,
        centroid_acknowledged=False,
    )
    result = simulate_scenarios(
        product_name="Wonka Bar - Milk Chocolate",
        region="Interior",
        ship_mode="Standard Class",
        units=2,
        division="Chocolate",
        state="Texas",
        country="United States",
        mapping=reference_mapping(),
        eligibility=default_eligibility(),
        factories=pd.DataFrame(FACTORIES),
        assumptions=assumptions,
        predict_fn=_predict_constant,
        destination_lat=31.0,
        destination_lon=-99.0,
        coordinate_label="manual coordinate, straight-line",
        historical_n=20,
    )
    rows = result["rows"]
    current = rows.loc[rows["role"] == "current"].iloc[0]
    assert current["distance_delta_km"] == pytest.approx(0.0)
    assert current["scenario_days_saved"] == pytest.approx(0.0)
    assert (rows["coordinate_basis"] == "manual coordinate, straight-line").all()
    assert rows["distance_km"].notna().all()
    # A farther factory must not be hidden if the scenario is slower.
    slower = rows.loc[rows["scenario_days_saved"] < 0]
    assert not slower.empty


def test_recommendations_exclude_customer_identifiers_and_do_not_invent_profit():
    rows = []
    for idx, region in enumerate(["Interior", "Atlantic"]):
        for ship_mode in ["Standard Class", "Same Day"]:
            rows.append(
                raw_order(
                    **{
                        "Row ID": idx * 10 + len(ship_mode),
                        "Order ID": f"US-2021-{idx}-{ship_mode[:3]}-CHO-MIL-31000",
                        "Region": region,
                        "Ship Mode": ship_mode,
                        "Product Name": "Wonka Bar - Milk Chocolate",
                        "Customer ID": f"SECRET-{idx}",
                    }
                )
            )
            rows.append(
                raw_order(
                    **{
                        "Row ID": 100 + idx * 10 + len(ship_mode),
                        "Order ID": f"US-2022-{idx}-{ship_mode[:3]}-CHO-SCR-58000",
                        "Region": region,
                        "Ship Mode": ship_mode,
                        "Product Name": "Wonka Bar -Scrumdiddlyumptious",
                        "Division": "Chocolate",
                        "Product ID": "CHO-SCR-58000",
                        "Customer ID": f"SECRET-{idx}",
                    }
                )
            )
    orders = frame_from_rows(rows)
    assumptions = ScenarioAssumptions(factory_effect_learned=False, profit_weight=0.8, speed_weight=0.2)
    table = build_recommendations(
        orders,
        reference_mapping(),
        default_eligibility(),
        pd.DataFrame(FACTORIES),
        assumptions,
        _predict_constant,
    )
    assert not table.empty
    assert "customer_id" not in table.columns
    assert "postal_code" not in table.columns
    assert (table["days_saved"] == 0).all()
    assert (table["financial_impact_status"] == "Not quantified").all()
    assert table["score_renormalized"].all()
    assert (table["evidence_level"] == "Requires validation").all()
    assert (table["actionable"] == False).all()  # noqa: E712
    assert "SECRET" not in table.to_csv(index=False)
