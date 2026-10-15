"""Module C — side-by-side what-if comparison."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from ..simulator import simulate_scenarios
from ..ui import (
    assumptions_from_state,
    callout,
    centroid_destination,
    effective_eligibility,
    manual_destination,
    page_intro,
    predictors,
    render_assumption_controls,
    ship_mode_options,
    show_table,
    weight_note,
)
from ..visualizations import whatif_bars
from .simulator_page import _mode


def _outcome(days) -> str:
    if days is None or (isinstance(days, float) and pd.isna(days)):
        return "No model difference"
    if days > 0.05:
        return "Faster"
    if days < -0.05:
        return "Slower"
    return "No change"


def render_whatif(ctx) -> None:
    page_intro(
        "Module C",
        "What-if analysis",
        "Hold the destination and ship mode fixed, then read the current factory beside one or more alternatives. Slower outcomes stay on the chart.",
    )
    if ctx.bundle is None:
        callout("warning", "Predictions are unavailable", ctx.model_error or "Train a model first.")
        return

    orders = ctx.orders
    products = sorted(orders["product_name"].dropna().unique())
    regions = sorted(orders["region"].dropna().unique())
    controls = st.columns(4)
    product = controls[0].selectbox("Product", products, key="sim_product")
    region = controls[1].selectbox("Destination region", regions, key="sim_region")
    ship_mode = controls[2].selectbox("Ship mode", ship_mode_options(orders), key="sim_ship_mode")
    product_rows = orders.loc[orders["product_name"] == product]
    region_rows = orders.loc[orders["region"] == region]
    if st.session_state.get("_units_for") != product:
        st.session_state["sim_units"] = int(max(1, round(float(product_rows["units"].median()))))
        st.session_state["_units_for"] = product
    units = controls[3].number_input("Units", min_value=1, max_value=50, step=1, key="sim_units")
    states = ["Most common in this region"] + sorted(region_rows["state"].dropna().unique())
    state_choice = st.selectbox("State context", states, key="sim_state")
    st.checkbox("Enter a destination coordinate", key="use_manual_coord")
    if st.session_state["use_manual_coord"]:
        cols = st.columns(2)
        cols[0].number_input("Destination latitude", min_value=-90.0, max_value=90.0, format="%.5f", key="sim_lat")
        cols[1].number_input("Destination longitude", min_value=-180.0, max_value=180.0, format="%.5f", key="sim_lon")

    render_assumption_controls(ctx.factories)
    assumptions = assumptions_from_state(ctx.bundle, ctx.factories)
    state = _mode(region_rows["state"]) if state_choice == "Most common in this region" else state_choice
    country = _mode(region_rows.loc[region_rows["state"] == state, "country"], _mode(region_rows["country"]))
    division = _mode(product_rows["division"])
    cell = orders.loc[
        (orders["product_name"] == product) & (orders["region"] == region) & (orders["ship_mode"] == ship_mode)
    ]
    lat, lon, coord_label = None, None, "none"
    if assumptions.distance_enabled and assumptions.coordinate_source == "manual":
        lat, lon, coord_label = manual_destination()
    elif assumptions.distance_enabled and assumptions.coordinate_source == "state_centroid" and assumptions.centroid_acknowledged:
        lat, lon, coord_label = centroid_destination(ctx.centroids, state)

    predict, alternate = predictors(ctx.bundle)
    result = simulate_scenarios(
        product_name=product,
        region=region,
        ship_mode=ship_mode,
        units=float(units),
        division=division,
        state=state,
        country=country,
        mapping=ctx.mapping,
        eligibility=effective_eligibility(ctx.eligibility),
        factories=ctx.factories,
        assumptions=assumptions,
        predict_fn=predict,
        alternate_predict_fn=alternate,
        feature_columns=ctx.bundle["feature_columns"],
        destination_lat=lat,
        destination_lon=lon,
        coordinate_label=coord_label,
        historical_n=int(len(cell)),
        historical_median=float(cell["lead_time_days"].median()) if not cell.empty else None,
        historical_sales=float(cell["sales"].sum()) if not cell.empty else 0.0,
        historical_gross_profit=float(cell["gross_profit"].sum()) if not cell.empty else 0.0,
        order_month=assumptions.reference_month,
        order_dow=assumptions.reference_dow,
    )
    rows = result["rows"]
    if rows.empty:
        callout("critical", "No comparison", result["meta"].get("detail") or result["meta"].get("error", "Nothing to compare."))
        return

    current = rows.loc[rows["role"] == "current"].iloc[0]
    alternatives = rows.loc[rows["role"] == "alternative", "candidate_factory"].tolist()
    chosen = st.multiselect("Alternatives to compare", alternatives, default=alternatives[:2], key="whatif_alts")
    selected = rows.loc[rows["candidate_factory"].isin([current["candidate_factory"], *chosen])].copy()
    if selected.empty:
        selected = rows.copy()

    kind = "warning" if result["meta"].get("decision") != "actionable_candidate" else "info"
    callout(kind, result["meta"].get("headline", ""), result["meta"].get("detail", ""))
    st.caption(weight_note(assumptions))

    left, right = st.columns(2)
    with left:
        st.markdown(f"**Baseline · {current['current_factory']}**")
        st.metric("Model lead time", f"{current['predicted_current_lead_time_days']:.1f} days")
        st.metric("Eligibility", current["eligibility"].replace("_", " "))
        st.caption(f"Evidence for the baseline context: {current['evidence_level']}.")
    with right:
        if chosen:
            focus = rows.loc[rows["candidate_factory"] == chosen[0]].iloc[0]
            st.markdown(f"**Alternative · {focus['candidate_factory']}**")
            st.metric("Model lead time", f"{focus['predicted_alternative_lead_time_days']:.1f} days", delta=f"{focus['days_saved']:.1f} days saved")
            scenario_delta = focus["scenario_days_saved"]
            scenario_text = "not applied" if pd.isna(scenario_delta) else f"{scenario_delta:.1f} days saved"
            st.metric("Scenario versus current", scenario_text)
            st.caption(f"Outcome on the model: {_outcome(focus['days_saved'])}. Scenario outcome: {_outcome(focus['scenario_days_saved'])}.")
        else:
            st.caption("Select at least one alternative. The current factory stays in the chart either way.")

    st.plotly_chart(whatif_bars(selected), width="stretch", config={"displayModeBar": False})
    table = selected.copy()
    table["model_outcome"] = table["days_saved"].map(_outcome)
    table["scenario_outcome"] = table["scenario_days_saved"].map(_outcome)
    show_table(
        table[
            [
                "role",
                "candidate_factory",
                "eligibility",
                "predicted_alternative_lead_time_days",
                "days_saved",
                "model_outcome",
                "scenario_alternative_lead_time_days",
                "scenario_days_saved",
                "scenario_outcome",
                "financial_impact_status",
                "evidence_level",
            ]
        ].rename(
            columns={
                "role": "Role",
                "candidate_factory": "Factory",
                "eligibility": "Eligibility",
                "predicted_alternative_lead_time_days": "Model lead time (days)",
                "days_saved": "Days saved (model)",
                "model_outcome": "Model outcome",
                "scenario_alternative_lead_time_days": "Scenario lead time (days)",
                "scenario_days_saved": "Days saved (scenario)",
                "scenario_outcome": "Scenario outcome",
                "financial_impact_status": "Financial impact",
                "evidence_level": "Evidence",
            }
        )
    )
    st.caption("Faster, slower, and no change are written out. Color is not the only signal. A negative days-saved value is a deterioration.")

    assumptions_view = {
        "product": product,
        "region": region,
        "ship_mode": ship_mode,
        "units": float(units),
        "state_context": state,
        "country_context": country,
        "reference_month": assumptions.reference_month,
        "reference_weekday": assumptions.reference_dow,
        "model": assumptions.model_name,
        "model_version": assumptions.model_version,
        "factory_effect_learned": assumptions.factory_effect_learned,
        "speed_weight_requested": assumptions.speed_weight,
        "profit_weight_requested": assumptions.profit_weight,
        "risk_weight": assumptions.risk_weight,
        "distance_enabled": assumptions.distance_enabled,
        "coordinate_source": assumptions.coordinate_source,
        "coordinate_basis": coord_label,
        "days_per_1000km": assumptions.days_per_1000km,
        "delay_cost_per_order_day": assumptions.delay_cost_per_order_day,
        "freight_cost_per_1000km": assumptions.freight_cost_per_1000km,
        "rank_by_scenario": assumptions.rank_by_scenario,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "fairness": result["meta"].get("fairness"),
    }
    with st.expander("Assumptions used for this comparison", expanded=False):
        st.json(assumptions_view)
    export = table.drop(columns=["reason_codes"], errors="ignore")
    st.download_button(
        "Download comparison",
        export.to_csv(index=False).encode("utf-8"),
        file_name="what_if_comparison.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download assumptions",
        json.dumps(assumptions_view, indent=2, default=str).encode("utf-8"),
        file_name="what_if_assumptions.json",
        mime="application/json",
    )
