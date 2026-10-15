"""Factory scenario comparisons. Factory labels change a prediction only if the model learned them."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import ADVISORY_TEXT, BASE_FEATURES, REASON_TEXT
from .data_loader import normalize_product_name
from .geo import haversine_km, valid_lat_lon
from .recommendations import (
    ScenarioAssumptions,
    compute_recommendation_score,
    evidence_label,
    explain_recommendation,
    financial_status,
    is_actionable,
    lead_time_change,
    normalize_signed,
    profit_inputs_present,
    risk_components,
    scenario_adjustments_active,
    sensitivity_profit_delta,
)

PredictFn = Callable[[pd.DataFrame], np.ndarray]


def eligibility_status(eligibility: pd.DataFrame, product_name: str, factory: str, current_factory: str | None) -> str:
    key = normalize_product_name(product_name)
    if eligibility is None or eligibility.empty:
        return "confirmed_current" if factory == current_factory else "requires_validation"
    hit = eligibility.loc[
        (eligibility["product_name_key"] == key) & (eligibility["factory"] == factory)
    ]
    if hit.empty:
        return "confirmed_current" if factory == current_factory else "requires_validation"
    return str(hit.iloc[0]["status"])


def current_factory_for(mapping: pd.DataFrame, product_name: str) -> str | None:
    key = normalize_product_name(product_name)
    hit = mapping.loc[mapping["product_name_key"] == key]
    if hit.empty:
        return None
    return str(hit.iloc[0]["current_factory"])


def _prediction_row(
    *,
    product_name: str,
    region: str,
    ship_mode: str,
    division: str,
    state: str,
    country: str,
    units: float,
    order_month: int,
    order_dow: int,
    origin_factory: str | None,
    factory_effect_learned: bool,
) -> dict[str, Any]:
    quarter = int((int(order_month) - 1) // 3 + 1)
    row = {
        "ship_mode": ship_mode,
        "region": region,
        "division": division,
        "product_name": product_name,
        "state": state or "Unknown",
        "country": country or "Unknown",
        "units": float(units),
        "order_month": int(order_month),
        "order_dow": int(order_dow),
        "order_quarter": quarter,
    }
    if factory_effect_learned:
        row["origin_factory"] = origin_factory or "Unknown"
    return row


def _handling(assumptions: ScenarioAssumptions, factory: str) -> float:
    return float((assumptions.factory_handling_days or {}).get(factory, 0.0))


def simulate_scenarios(
    *,
    product_name: str,
    region: str,
    ship_mode: str,
    units: float,
    division: str,
    state: str | None,
    country: str | None,
    mapping: pd.DataFrame,
    eligibility: pd.DataFrame,
    factories: pd.DataFrame,
    assumptions: ScenarioAssumptions,
    predict_fn: PredictFn,
    alternate_predict_fn: PredictFn | None = None,
    feature_columns: list[str] | None = None,
    destination_lat: float | None = None,
    destination_lon: float | None = None,
    coordinate_label: str = "none",
    historical_n: int = 0,
    historical_median: float | None = None,
    historical_sales: float = 0.0,
    historical_gross_profit: float = 0.0,
    historical_margin: float | None = None,
    include_ineligible: bool = False,
    order_month: int | None = None,
    order_dow: int | None = None,
) -> dict[str, Any]:
    """Compare the current factory with alternatives using the same decision context."""
    current = current_factory_for(mapping, product_name)
    if current is None:
        return {
            "rows": pd.DataFrame(),
            "meta": {
                "error": f"No supplied factory mapping for {product_name}.",
                "decision": "blocked",
                "headline": "Cannot simulate this product.",
                "detail": "The product is not in the supplied factory mapping.",
            },
        }

    month = int(order_month if order_month is not None else assumptions.reference_month)
    dow = int(order_dow if order_dow is not None else assumptions.reference_dow)
    learned = bool(assumptions.factory_effect_learned)
    columns = list(feature_columns or BASE_FEATURES)
    if learned and "origin_factory" not in columns:
        columns.append("origin_factory")

    factory_rows = []
    for factory in factories["factory"].tolist():
        status = eligibility_status(eligibility, product_name, factory, current)
        role = "current" if factory == current else "alternative"
        if role == "alternative" and status == "ineligible" and not include_ineligible:
            continue
        factory_rows.append({"factory": factory, "eligibility": status, "role": role})
    if not factory_rows:
        return {
            "rows": pd.DataFrame(),
            "meta": {"error": "No factories available to simulate.", "decision": "blocked"},
        }

    predict_input = pd.DataFrame(
        [
            _prediction_row(
                product_name=product_name,
                region=region,
                ship_mode=ship_mode,
                division=division,
                state=state or "Unknown",
                country=country or "Unknown",
                units=units,
                order_month=month,
                order_dow=dow,
                origin_factory=item["factory"] if learned else None,
                factory_effect_learned=learned,
            )
            for item in factory_rows
        ]
    )
    # Fairness: non-factory inputs are identical. Factory is passed only when learned.
    if not learned:
        shared = predict_input.drop_duplicates()
        if len(shared) != 1:
            raise RuntimeError("Scenario context was not held constant across factories.")
        shared_pred = float(np.asarray(predict_fn(shared), dtype=float)[0])
        predictions = np.full(len(predict_input), shared_pred)
        if alternate_predict_fn is not None:
            alt_shared = float(np.asarray(alternate_predict_fn(shared), dtype=float)[0])
            alternate_predictions = np.full(len(predict_input), alt_shared)
        else:
            alternate_predictions = None
    else:
        predictions = np.asarray(predict_fn(predict_input), dtype=float)
        alternate_predictions = (
            np.asarray(alternate_predict_fn(predict_input), dtype=float)
            if alternate_predict_fn is not None
            else None
        )

    coords_ok = valid_lat_lon(destination_lat, destination_lon)
    factory_lookup = factories.set_index("factory")
    current_coords = factory_lookup.loc[current] if current in factory_lookup.index else None
    current_distance = None
    if coords_ok and current_coords is not None and pd.notna(current_coords["latitude"]):
        current_distance = haversine_km(
            float(destination_lat),
            float(destination_lon),
            float(current_coords["latitude"]),
            float(current_coords["longitude"]),
        )

    distance_requested = bool(assumptions.distance_enabled and assumptions.coordinate_source != "none")
    if assumptions.coordinate_source == "state_centroid" and not assumptions.centroid_acknowledged:
        distance_requested = False
    handling_active = any(abs(_handling(assumptions, item["factory"])) > 1e-9 for item in factory_rows)
    scenario_applied = bool((distance_requested and coords_ok) or handling_active)
    uses_assumption = bool(scenario_applied and (distance_requested or handling_active))
    # Ranking may use the scenario only when the user asks and a scenario exists.
    rank_on_scenario = bool(assumptions.rank_by_scenario and scenario_applied)

    current_pred = float(predictions[next(i for i, item in enumerate(factory_rows) if item["role"] == "current")])
    current_handling = _handling(assumptions, current)
    current_scenario = current_pred + 0.0  # relative distance adjustment is zero at the current factory
    if scenario_applied:
        current_scenario = current_pred + 0.0

    records = []
    for idx, item in enumerate(factory_rows):
        factory = item["factory"]
        pred = float(predictions[idx])
        distance_km = None
        distance_delta = None
        distance_days = 0.0
        missing_coordinates = False
        if distance_requested:
            if not coords_ok or factory not in factory_lookup.index or pd.isna(factory_lookup.loc[factory, "latitude"]):
                missing_coordinates = True
            else:
                distance_km = haversine_km(
                    float(destination_lat),
                    float(destination_lon),
                    float(factory_lookup.loc[factory, "latitude"]),
                    float(factory_lookup.loc[factory, "longitude"]),
                )
                if current_distance is not None:
                    distance_delta = distance_km - current_distance
                    distance_days = distance_delta / 1000.0 * float(assumptions.days_per_1000km)
        handling_delta = _handling(assumptions, factory) - current_handling
        if scenario_applied and not missing_coordinates:
            scenario_lead = pred + distance_days + handling_delta
        elif scenario_applied and missing_coordinates and handling_active:
            scenario_lead = pred + handling_delta
        else:
            scenario_lead = None

        model_days, model_pct = lead_time_change(current_pred, pred)
        if scenario_lead is None:
            scenario_days, scenario_pct = None, None
            scenario_current = None
        else:
            scenario_current = current_scenario if not handling_active and not distance_requested else current_pred
            # Current scenario lead time includes zero relative adjustment.
            scenario_current = current_pred + 0.0
            scenario_days, scenario_pct = lead_time_change(scenario_current, scenario_lead)

        disagreement = False
        if alternate_predictions is not None:
            disagreement = abs(pred - float(alternate_predictions[idx])) > float(assumptions.disagreement_threshold_days)

        extrapolation = historical_n < 1
        profit_days = scenario_days if rank_on_scenario and scenario_days is not None else model_days
        profit_delta = sensitivity_profit_delta(
            days_saved=profit_days,
            distance_delta_km=distance_delta,
            units=units,
            assumptions=assumptions,
            factory=factory,
            current_factory=current,
        )
        speed_basis = scenario_days if rank_on_scenario and scenario_days is not None else model_days
        speed_score = None if speed_basis is None else normalize_signed(speed_basis, assumptions.speed_cap_days)
        profit_score = None if profit_delta is None else normalize_signed(profit_delta, assumptions.profit_cap_dollars)
        penalty, flags = risk_components(
            sample_size=int(historical_n),
            eligibility=item["eligibility"] if item["role"] == "alternative" else "confirmed_current",
            target_quality_penalty=assumptions.target_quality_penalty,
            uses_unvalidated_assumption=bool(uses_assumption and item["role"] == "alternative" and rank_on_scenario),
            extrapolation=extrapolation,
            model_disagreement=disagreement,
            profit_weight=assumptions.profit_weight,
            profit_available=profit_inputs_present(assumptions),
            missing_coordinates=missing_coordinates and distance_requested,
        )
        if not learned:
            flags.append("FACTORY_EFFECT_NOT_OBSERVED")
        if model_days is not None and model_days <= 0:
            flags.append("NO_LEAD_TIME_GAIN")
        if scenario_days is not None and scenario_days < -0.05:
            flags.append("PREDICTED_DETERIORATION")
        if distance_requested and coords_ok:
            flags.append("DISTANCE_PROXY_ASSUMPTION")
        flags.append("ADVISORY_ONLY")
        flags.append("OBSERVED_LEVER_IS_SHIP_MODE")
        # Preserve order while dropping duplicates.
        seen: set[str] = set()
        reason_codes = []
        for code in flags:
            if code not in seen:
                seen.add(code)
                reason_codes.append(code)

        evidence = evidence_label(
            sample_size=int(historical_n),
            eligibility=item["eligibility"] if item["role"] == "alternative" else "eligible",
            factory_effect_learned=learned,
            uses_unvalidated_assumption=bool(uses_assumption and rank_on_scenario and item["role"] == "alternative"),
            extrapolation=extrapolation,
            model_disagreement=disagreement,
        )
        # Current factory is the baseline, not a reassignment claim.
        if item["role"] == "current":
            evidence = "Limited evidence" if assumptions.target_quality_penalty >= 0.5 else evidence
            if not learned:
                evidence = "Limited evidence"

        score = compute_recommendation_score(
            speed_score=speed_score,
            profit_score=profit_score,
            risk_penalty=penalty,
            speed_weight=assumptions.speed_weight,
            profit_weight=assumptions.profit_weight,
            risk_weight=assumptions.risk_weight,
        )
        actionable = is_actionable(
            evidence=evidence,
            eligibility=item["eligibility"],
            days_saved=model_days,
            factory_effect_learned=learned,
            uses_unvalidated_assumption=uses_assumption,
        )
        record = {
            "product_name": product_name,
            "division": division,
            "region": region,
            "ship_mode": ship_mode,
            "state_context": state or "Unknown",
            "country_context": country or "Unknown",
            "units": float(units),
            "order_month": month,
            "order_dow": dow,
            "current_factory": current,
            "candidate_factory": factory,
            "role": item["role"],
            "eligibility": item["eligibility"],
            "sample_size": int(historical_n),
            "predicted_current_lead_time_days": current_pred,
            "predicted_alternative_lead_time_days": pred,
            "days_saved": model_days,
            "lead_time_change_pct": model_pct,
            "scenario_applied": scenario_applied,
            "scenario_current_lead_time_days": scenario_current if scenario_applied else None,
            "scenario_alternative_lead_time_days": scenario_lead,
            "scenario_days_saved": scenario_days,
            "scenario_change_pct": scenario_pct,
            "ranking_basis": "assumption_scenario" if rank_on_scenario else "model",
            "speed_score": speed_score,
            "profit_score": profit_score,
            "risk_penalty": penalty,
            "recommendation_score": score["score"],
            "score_renormalized": score["renormalized"],
            "applied_speed_weight": score["components"].get("speed", {}).get("applied_weight"),
            "applied_profit_weight": score["components"].get("profit", {}).get("applied_weight"),
            "requested_speed_weight": assumptions.speed_weight,
            "requested_profit_weight": assumptions.profit_weight,
            "risk_weight": assumptions.risk_weight,
            "evidence_level": evidence,
            "risk_flags": "; ".join(reason_codes),
            "reason_codes": reason_codes,
            "financial_impact_status": financial_status(profit_delta),
            "sensitivity_profit_delta": profit_delta,
            "distance_km": distance_km,
            "distance_delta_km": distance_delta,
            "coordinate_basis": coordinate_label if coords_ok else "none",
            "historical_median_lead_time_days": historical_median,
            "historical_sales": historical_sales,
            "historical_gross_profit": historical_gross_profit,
            "historical_margin": historical_margin,
            "model_name": assumptions.model_name,
            "model_version": assumptions.model_version,
            "factory_effect_learned": learned,
            "model_disagreement": disagreement,
            "actionable": actionable,
            "advisory": ADVISORY_TEXT,
        }
        record["explanation"] = explain_recommendation(record)
        records.append(record)

    rows = pd.DataFrame(records)
    fairness = {
        "same_product": int(rows["product_name"].nunique()) == 1,
        "same_region": int(rows["region"].nunique()) == 1,
        "same_ship_mode": int(rows["ship_mode"].nunique()) == 1,
        "same_units": int(rows["units"].nunique()) == 1,
        "factory_effect_learned": learned,
        "model_predictions_factory_invariant": bool(rows["predicted_alternative_lead_time_days"].nunique(dropna=False) == 1)
        if not learned
        else False,
    }
    if not learned and not fairness["model_predictions_factory_invariant"]:
        raise RuntimeError("Unlearned factory labels changed the model prediction.")

    alts = rows.loc[rows["role"] == "alternative"]
    best_model = float(alts["days_saved"].max()) if not alts.empty else 0.0
    best_scenario = alts["scenario_days_saved"].dropna()
    best_scenario_days = float(best_scenario.max()) if not best_scenario.empty else None
    if learned and bool(alts["actionable"].any()):
        decision = "actionable_candidate"
        headline = "An alternative meets the actionable-evidence rule."
        detail = "Review the evidence, eligibility, and financial status before any assignment change."
    elif rank_on_scenario and best_scenario_days is not None and best_scenario_days > 0:
        decision = "exploratory_scenario"
        headline = "Exploratory scenario only. Do not treat this as a decision."
        detail = (
            "An alternative looks faster only under the distance or handling assumptions you entered. "
            "Those assumptions are not a historical factory effect."
        )
    else:
        decision = "retain_current"
        headline = f"Retain {current}."
        detail = (
            "No model-supported lead-time gain is available for an alternative factory. "
            f"Best model-supported change: {best_model:.1f} days."
        )
    meta = {
        "decision": decision,
        "headline": headline,
        "detail": detail,
        "fairness": fairness,
        "current_factory": current,
        "scenario_applied": scenario_applied,
        "coordinate_label": coordinate_label if coords_ok else "none",
        "feature_columns": columns,
        "context_held_constant": ["product_name", "region", "ship_mode", "units", "state", "country", "order_month", "order_dow"],
        "reason_text": REASON_TEXT,
    }
    return {"rows": rows, "meta": meta}


def segment_history(orders: pd.DataFrame) -> pd.DataFrame:
    work = orders.dropna(subset=["lead_time_days"]).copy()
    grouped = (
        work.groupby(["product_name", "region", "ship_mode"], dropna=False)
        .agg(
            sample_size=("lead_time_days", "size"),
            historical_median=("lead_time_days", "median"),
            historical_sales=("sales", "sum"),
            historical_gross_profit=("gross_profit", "sum"),
            units=("units", "median"),
            division=("division", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
            state=("state", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
            country=("country", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
        )
        .reset_index()
    )
    grouped["historical_margin"] = np.where(
        grouped["historical_sales"] > 0,
        grouped["historical_gross_profit"] / grouped["historical_sales"],
        np.nan,
    )
    return grouped


def build_recommendations(
    orders: pd.DataFrame,
    mapping: pd.DataFrame,
    eligibility: pd.DataFrame,
    factories: pd.DataFrame,
    assumptions: ScenarioAssumptions,
    predict_fn: PredictFn,
    alternate_predict_fn: PredictFn | None = None,
    feature_columns: list[str] | None = None,
    destination_lookup: dict[tuple[str, str], tuple[float, float, str]] | None = None,
) -> pd.DataFrame:
    """Rank observed product-region-ship-mode alternatives.

    Unobserved combinations are left to the simulator so the ranked table does
    not invent a history. Ineligible factories are excluded.
    """
    history = segment_history(orders)
    frames = []
    for row in history.itertuples(index=False):
        destination = None
        label = "none"
        if destination_lookup:
            destination = destination_lookup.get((row.region, row.state)) or destination_lookup.get((row.region, "*"))
            if destination is not None:
                lat, lon, label = destination
            else:
                lat = lon = None
        else:
            lat = lon = None
        result = simulate_scenarios(
            product_name=row.product_name,
            region=row.region,
            ship_mode=row.ship_mode,
            units=float(row.units) if pd.notna(row.units) else 1.0,
            division=str(row.division),
            state=str(row.state),
            country=str(row.country),
            mapping=mapping,
            eligibility=eligibility,
            factories=factories,
            assumptions=assumptions,
            predict_fn=predict_fn,
            alternate_predict_fn=alternate_predict_fn,
            feature_columns=feature_columns,
            destination_lat=lat,
            destination_lon=lon,
            coordinate_label=label,
            historical_n=int(row.sample_size),
            historical_median=float(row.historical_median) if pd.notna(row.historical_median) else None,
            historical_sales=float(row.historical_sales),
            historical_gross_profit=float(row.historical_gross_profit),
            historical_margin=float(row.historical_margin) if pd.notna(row.historical_margin) else None,
            include_ineligible=False,
        )
        if result["rows"].empty:
            continue
        frames.append(result["rows"])
    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True)
    table = table.loc[table["role"] == "alternative"].copy()
    table = table.sort_values(
        ["recommendation_score", "historical_gross_profit", "product_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    return table
