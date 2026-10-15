"""Module B — Factory optimization simulator."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ..simulator import simulate_scenarios
from ..ui import (
    assumptions_from_state,
    callout,
    centroid_destination,
    effective_eligibility,
    manual_destination,
    model_label,
    page_intro,
    predictors,
    render_assumption_controls,
    ship_mode_options,
    show_table,
    weight_note,
)
from ..visualizations import simulator_bars


def _mode(series: pd.Series, fallback: str = "Unknown") -> str:
    if series.empty or series.dropna().empty:
        return fallback
    modes = series.dropna().mode()
    return str(modes.iloc[0]) if not modes.empty else fallback


def render_simulator(ctx) -> None:
    page_intro(
        "Module B",
        "Factory optimization simulator",
        "Compare the assumed current factory with alternatives under the same product, region, ship mode, and order size. The historical model changes a factory result only if it learned an origin factory.",
    )
    if ctx.bundle is None:
        callout("warning", "Predictions are unavailable", ctx.model_error or "Train a model on the Model and data quality page.")
        return

    orders = ctx.orders
    products = sorted(orders["product_name"].dropna().unique())
    regions = sorted(orders["region"].dropna().unique())
    modes = ship_mode_options(orders)
    controls = st.columns(4)
    product = controls[0].selectbox("Product", products, key="sim_product")
    region = controls[1].selectbox("Destination region", regions, key="sim_region")
    ship_mode = controls[2].selectbox("Ship mode", modes, key="sim_ship_mode")
    product_rows = orders.loc[orders["product_name"] == product]
    region_rows = orders.loc[orders["region"] == region]
    if st.session_state.get("_units_for") != product:
        st.session_state["sim_units"] = int(max(1, round(float(product_rows["units"].median()))))
        st.session_state["_units_for"] = product
    units = controls[3].number_input("Units", min_value=1, max_value=50, step=1, key="sim_units")

    states = ["Most common in this region"] + sorted(region_rows["state"].dropna().unique())
    extra = st.columns(3)
    state_choice = extra[0].selectbox("State context", states, key="sim_state")
    extra[1].checkbox("Enter a destination coordinate", key="use_manual_coord")
    if st.session_state["use_manual_coord"]:
        coord_cols = st.columns(2)
        coord_cols[0].number_input("Destination latitude", min_value=-90.0, max_value=90.0, format="%.5f", key="sim_lat")
        coord_cols[1].number_input("Destination longitude", min_value=-180.0, max_value=180.0, format="%.5f", key="sim_lon")
        st.caption("This coordinate is used only for the straight-line sensitivity. It is not stored as a customer record.")

    render_assumption_controls(ctx.factories)
    assumptions = assumptions_from_state(ctx.bundle, ctx.factories)
    state = _mode(region_rows["state"]) if state_choice == "Most common in this region" else state_choice
    country = _mode(region_rows.loc[region_rows["state"] == state, "country"], _mode(region_rows["country"]))
    division = _mode(product_rows["division"])
    mapping_hit = ctx.mapping.loc[ctx.mapping["product_name"] == product]
    if mapping_hit.empty:
        callout("critical", "Unmapped product", "This product is not in the supplied factory mapping, so there is no current-factory baseline.")
        return
    current = str(mapping_hit.iloc[0]["current_factory"])
    mapped_division = str(mapping_hit.iloc[0]["mapping_division"])
    if mapped_division != division:
        st.caption(f"Extract division is {division}; the mapping table says {mapped_division}. The model uses the extract division.")

    cell = orders.loc[
        (orders["product_name"] == product) & (orders["region"] == region) & (orders["ship_mode"] == ship_mode)
    ]
    lat, lon, coord_label = None, None, "none"
    if assumptions.distance_enabled and assumptions.coordinate_source == "manual":
        lat, lon, coord_label = manual_destination()
        if lat is None:
            st.caption("Distance sensitivity is on, but no destination coordinate is entered yet. Model results still show.")
    elif assumptions.distance_enabled and assumptions.coordinate_source == "state_centroid":
        if assumptions.centroid_acknowledged:
            lat, lon, coord_label = centroid_destination(ctx.centroids, state)
        else:
            st.caption("Acknowledge the state-centroid limitation in Scenario assumptions before distance is applied.")
    elif assumptions.distance_enabled and assumptions.coordinate_source == "uploaded":
        geocodes = st.session_state.get("geocodes")
        if geocodes is None or getattr(geocodes, "empty", True):
            st.caption("Upload a geocode file in the sidebar to use this coordinate source.")
        else:
            city = _mode(cell["city"]) if not cell.empty else None
            hit = geocodes.loc[
                (geocodes["state_key"] == str(state).strip().lower())
                & (geocodes["city_key"] == str(city or "").strip().lower())
            ]
            if not hit.empty:
                lat = float(hit.iloc[0]["latitude"])
                lon = float(hit.iloc[0]["longitude"])
                coord_label = f"Uploaded coordinate for {city}, {state}. Straight-line, not road distance."

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
        historical_margin=float(cell["gross_profit"].sum() / cell["sales"].sum()) if not cell.empty and cell["sales"].sum() else None,
        order_month=assumptions.reference_month,
        order_dow=assumptions.reference_dow,
    )
    st.session_state["last_scenario"] = {
        "product": product,
        "region": region,
        "ship_mode": ship_mode,
        "units": float(units),
        "state": state,
    }
    meta = result["meta"]
    kind = "warning" if meta.get("decision") in {"retain_current", "exploratory_scenario"} else "info"
    if meta.get("decision") == "blocked":
        kind = "critical"
    callout(kind, meta.get("headline", "No result"), meta.get("detail", ""))
    st.caption(weight_note(assumptions) + f" Selected model: {model_label(assumptions.model_name)}.")

    rows = result["rows"]
    if rows.empty:
        return
    current_row = rows.loc[rows["role"] == "current"].iloc[0]
    best_alt = rows.loc[rows["role"] == "alternative", "days_saved"].max()
    metrics = st.columns(4)
    metrics[0].metric("Current factory", current)
    metrics[1].metric("Model lead time", f"{current_row['predicted_current_lead_time_days']:.1f} days")
    metrics[2].metric("Best model days saved", f"{best_alt:.1f}")
    metrics[3].metric("Historical orders in this cell", f"{int(len(cell)):,}")
    band = ctx.bundle["targets"]["lead_time_days"]
    st.caption(
        "Illustrative test-residual band, not a calibrated prediction interval: "
        f"{current_row['predicted_current_lead_time_days'] + band['test_residual_p10']:.0f} to "
        f"{current_row['predicted_current_lead_time_days'] + band['test_residual_p90']:.0f} days. "
        f"Reference timing is month {assumptions.reference_month}, weekday {assumptions.reference_dow}, state {state}. "
        "The same inputs are used for every factory."
    )
    if not cell.empty and "descriptive_residual_days" in cell:
        st.caption(
            f"Descriptive within-cohort residual for this cell: median {cell['descriptive_residual_days'].median():.1f} days "
            "versus the Order ID year median. That residual is not a factory effect."
        )

    st.plotly_chart(simulator_bars(rows), width="stretch", config={"displayModeBar": False})
    if rows["predicted_alternative_lead_time_days"].nunique() == 1:
        st.caption("The model bars are equal on purpose. Swapping a factory label the model never learned does not create a faster prediction.")
    display_cols = [
        "role",
        "candidate_factory",
        "eligibility",
        "predicted_alternative_lead_time_days",
        "days_saved",
        "lead_time_change_pct",
        "scenario_alternative_lead_time_days",
        "scenario_days_saved",
        "scenario_change_pct",
        "distance_km",
        "evidence_level",
        "financial_impact_status",
        "sensitivity_profit_delta",
        "risk_flags",
        "recommendation_score",
    ]
    show_table(
        rows.loc[:, [col for col in display_cols if col in rows.columns]].rename(
            columns={
                "role": "Role",
                "candidate_factory": "Factory",
                "eligibility": "Eligibility",
                "predicted_alternative_lead_time_days": "Model lead time (days)",
                "days_saved": "Days saved (model)",
                "lead_time_change_pct": "Change % (model)",
                "scenario_alternative_lead_time_days": "Scenario lead time (days)",
                "scenario_days_saved": "Days saved (scenario)",
                "scenario_change_pct": "Change % (scenario)",
                "distance_km": "Great-circle km",
                "evidence_level": "Evidence",
                "financial_impact_status": "Financial impact",
                "sensitivity_profit_delta": "Sensitivity $",
                "risk_flags": "Risk flags",
                "recommendation_score": "Score",
            }
        )
    )
    with st.expander("Why this comparison looks like this", expanded=True):
        st.write(rows.loc[rows["role"] == "current", "explanation"].iloc[0])
        st.write(
            "Negative scenario results are kept. A factory that looks slower under your assumptions is shown, not filtered out. "
            "Eligibility of alternatives is unverified unless you mark them eligible on the Risk and impact page. "
            "Marking a factory eligible removes only the eligibility warning. It does not create historical evidence."
        )
    csv = rows.drop(columns=["reason_codes"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button("Download this scenario", csv, file_name="factory_scenario.csv", mime="text/csv")
