"""Shared Streamlit controls. Core calculations stay outside this module."""

from __future__ import annotations

from html import escape
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from .config import (
    ADVISORY_TEXT,
    EXAMPLE_COST_DELTA,
    EXAMPLE_DELAY_COST,
    EXAMPLE_HANDLING_DAYS,
    MODEL_VERSION,
    REASON_TEXT,
    SHIP_MODE_ORDER,
)
from .data_loader import normalize_product_name
from .recommendations import ScenarioAssumptions, profit_inputs_present, weights_from_priority
from .train import predict_frame

PAGES = [
    "Overview",
    "Factory simulator",
    "What-if analysis",
    "Recommendations",
    "Risk and impact",
    "Model and data quality",
]

SOURCE_LABELS = {
    "Manual latitude/longitude": "manual",
    "Approximate state centroids": "state_centroid",
    "Uploaded geocodes": "uploaded",
}
MODEL_LABELS = {
    "group_median_baseline": "Ship-mode median baseline",
    "linear_regression": "Linear regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "global_median": "Global median",
}
HIDDEN_ORDER_COLUMNS = {
    "customer_id",
    "postal_code",
    "order_date_mdy",
    "ship_date_mdy",
    "gp_gap",
    "descriptive_cohort_median",
}


def init_state() -> None:
    defaults = {
        "profit_priority": 30,
        "risk_weight": 0.40,
        "distance_enabled": False,
        "days_per_1000km": 1.0,
        "coordinate_source": "Manual latitude/longitude",
        "centroid_ack": False,
        "use_delay": False,
        "delay_cost": 5.0,
        "use_freight": False,
        "freight_cost": 40.0,
        "use_factory_cost": False,
        "rank_by_scenario": False,
        "reference_month": 6,
        "reference_dow": 2,
        "use_manual_coord": False,
        "sim_lat": 0.0,
        "sim_lon": 0.0,
        "page": "Overview",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def model_label(name: str | None) -> str:
    if not name:
        return "Not loaded"
    return MODEL_LABELS.get(name, name.replace("_", " "))


def render_header() -> None:
    st.markdown(
        f"""
        <div class="app-header">
          <div class="brand-mark">NC</div>
          <div>
            <div class="eyebrow">Nassau Candy Distributor · Decision support</div>
            <div class="app-title">Factory Reallocation &amp; Shipping Optimization</div>
            <div class="app-sub">{escape(ADVISORY_TEXT)} A lower predicted lead time is a reason to investigate, not proof that a transfer will improve delivery or profit.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_intro(kicker: str, title: str, lede: str) -> None:
    st.markdown(
        f"""
        <div class="page-intro">
          <div class="kicker">{escape(kicker)}</div>
          <h1>{escape(title)}</h1>
          <p class="lede">{escape(lede)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def callout(kind: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="callout callout-{escape(kind)}">
          <div class="callout-kicker">{escape(kind)}</div>
          <div class="callout-title">{escape(title)}</div>
          <div class="callout-body">{escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer(version: str = MODEL_VERSION) -> None:
    st.markdown(
        f'<div class="foot">Model {escape(version)} · recommendations are advisory and require human review · '
        "straight-line distance is never road distance · profit is not quantified unless you enter a labeled assumption.</div>",
        unsafe_allow_html=True,
    )


def show_table(frame: pd.DataFrame, height: int = 440) -> None:
    st.dataframe(frame, hide_index=True, width="stretch", height=height)


def preview_orders(orders: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    columns = [col for col in orders.columns if col not in HIDDEN_ORDER_COLUMNS]
    return orders.loc[:, columns].head(n)


def assumptions_from_state(bundle: dict[str, Any] | None, factories: pd.DataFrame) -> ScenarioAssumptions:
    speed, profit = weights_from_priority(float(st.session_state["profit_priority"]))
    handling: dict[str, float] = {}
    costs: dict[str, float] = {}
    table = st.session_state.get("handling_live")
    if not isinstance(table, pd.DataFrame) or table.empty:
        table = st.session_state.get("handling_table")
    if not isinstance(table, pd.DataFrame) or table.empty:
        table = pd.DataFrame(
            {"factory": factories["factory"], "handling_days": 0.0, "cost_delta_per_unit": 0.0}
        )
    for row in table.itertuples(index=False):
        handling[str(row.factory)] = float(row.handling_days or 0.0)
        costs[str(row.factory)] = float(row.cost_delta_per_unit or 0.0)
    source = SOURCE_LABELS.get(st.session_state["coordinate_source"], "none")
    learned = bool(bundle and bundle.get("factory_effect_learned"))
    model_name = bundle["targets"]["lead_time_days"]["selected_model"] if bundle else "unavailable"
    penalty = float(bundle.get("target_quality_penalty") or 0.85) if bundle else 0.85
    version = bundle.get("model_version", MODEL_VERSION) if bundle else MODEL_VERSION
    return ScenarioAssumptions(
        speed_weight=speed,
        profit_weight=profit,
        risk_weight=float(st.session_state["risk_weight"]),
        distance_enabled=bool(st.session_state["distance_enabled"]),
        days_per_1000km=float(st.session_state["days_per_1000km"]),
        coordinate_source=source if st.session_state["distance_enabled"] else "none",
        centroid_acknowledged=bool(st.session_state["centroid_ack"]),
        delay_cost_per_order_day=float(st.session_state["delay_cost"]) if st.session_state["use_delay"] else None,
        freight_cost_per_1000km=float(st.session_state["freight_cost"]) if st.session_state["use_freight"] else None,
        factory_handling_days=handling,
        factory_cost_delta_per_unit=costs,
        use_factory_cost=bool(st.session_state["use_factory_cost"]),
        reference_month=int(st.session_state["reference_month"]),
        reference_dow=int(st.session_state["reference_dow"]),
        rank_by_scenario=bool(st.session_state["rank_by_scenario"]),
        factory_effect_learned=learned,
        target_quality_penalty=penalty,
        model_name=model_name,
        model_version=version,
    )


def _example_handling(factories: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "factory": factories["factory"].tolist(),
            "handling_days": [EXAMPLE_HANDLING_DAYS.get(name, 0.0) for name in factories["factory"]],
            "cost_delta_per_unit": [EXAMPLE_COST_DELTA.get(name, 0.0) for name in factories["factory"]],
        }
    )


def render_assumption_controls(factories: pd.DataFrame) -> None:
    """Priority slider plus labeled cost and distance assumptions."""
    st.slider(
        "Optimization priority — speed to profit protection",
        min_value=0,
        max_value=100,
        key="profit_priority",
        help="0 puts all positive weight on speed. 100 puts it on profit protection. Profit weight is not applied until a cost assumption is entered.",
    )
    speed, profit = weights_from_priority(float(st.session_state["profit_priority"]))
    profit_live = st.session_state["use_delay"] or st.session_state["use_freight"] or st.session_state["use_factory_cost"]
    applied_profit = profit if profit_live else 0.0
    applied_speed = 1.0 if not profit_live else speed
    if profit_live and (speed + profit) > 0:
        applied_speed = speed
        applied_profit = profit
    st.markdown(
        f"""
        <div class="weight-row">
          <span class="weight-chip">Requested speed {speed:.0%}</span>
          <span class="weight-chip">Requested profit {profit:.0%}</span>
          <span class="weight-chip">Applied speed {applied_speed:.0%}</span>
          <span class="weight-chip">Applied profit {applied_profit:.0%}</span>
          <span class="weight-chip">Risk weight {float(st.session_state["risk_weight"]):.2f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not profit_live and profit > 0:
        st.caption(
            "Moving the slider changes the requested profit weight, but that weight is not applied. "
            "Profit impact is not quantified until you enter a cost assumption. The score stays speed-focused so a missing cost is not invented."
        )
    else:
        st.caption("Requested and applied weights match while a cost assumption is on. Dollar results remain a sensitivity, not a validated profit.")

    with st.expander("Scenario assumptions — distance, handling, and cost", expanded=False):
        st.markdown(
            "These controls do **not** create historical evidence. They run a labeled what-if. "
            "Leave them off to see the model-only comparison. "
            "The example sensitivity makes one factory faster and more expensive, and another slower and cheaper, "
            "so the priority slider can change the order. Those numbers are not company costs."
        )
        action_cols = st.columns(2)
        if action_cols[0].button("Load example sensitivity", help="Fills illustrative numbers so you can see the slider change rank. Not company costs."):
            st.session_state["distance_enabled"] = False
            st.session_state["use_delay"] = True
            st.session_state["delay_cost"] = EXAMPLE_DELAY_COST
            st.session_state["use_freight"] = False
            st.session_state["use_factory_cost"] = True
            st.session_state["rank_by_scenario"] = True
            st.session_state["handling_table"] = _example_handling(factories)
            st.session_state.pop("handling_editor", None)
            st.rerun()
        if action_cols[1].button("Clear assumptions"):
            st.session_state["distance_enabled"] = False
            st.session_state["use_delay"] = False
            st.session_state["use_freight"] = False
            st.session_state["use_factory_cost"] = False
            st.session_state["rank_by_scenario"] = False
            st.session_state["centroid_ack"] = False
            st.session_state["handling_table"] = pd.DataFrame(
                {"factory": factories["factory"], "handling_days": 0.0, "cost_delta_per_unit": 0.0}
            )
            st.session_state.pop("handling_editor", None)
            st.rerun()

        st.checkbox(
            "Enable straight-line distance sensitivity",
            key="distance_enabled",
            help="Off by default. Great-circle distance is not road distance and is not a learned factory effect.",
        )
        st.radio("Coordinate source", list(SOURCE_LABELS), key="coordinate_source", horizontal=True)
        st.checkbox(
            "I understand state centroids are approximate reference points, not customer locations, and not road distance.",
            key="centroid_ack",
        )
        st.number_input("Assumed extra days per 1,000 km of great-circle distance", min_value=0.0, max_value=30.0, step=0.25, key="days_per_1000km")
        cost_cols = st.columns(2)
        with cost_cols[0]:
            st.checkbox("Apply a delay-cost assumption", key="use_delay")
            st.number_input("Cost per extra lead-time day ($/order)", min_value=0.0, step=1.0, key="delay_cost", disabled=not st.session_state["use_delay"])
            st.checkbox("Apply a freight assumption", key="use_freight")
            st.number_input("Incremental freight $ per 1,000 km", min_value=0.0, step=5.0, key="freight_cost", disabled=not st.session_state["use_freight"])
        with cost_cols[1]:
            st.checkbox("Apply factory cost deltas", key="use_factory_cost")
            st.slider("Risk penalty weight", 0.0, 1.0, key="risk_weight", step=0.05)
            st.selectbox("Reference order month", list(range(1, 13)), key="reference_month")
            st.selectbox("Reference weekday (0 = Monday)", list(range(7)), key="reference_dow")
        if "handling_table" not in st.session_state:
            st.session_state["handling_table"] = pd.DataFrame(
                {"factory": factories["factory"].tolist(), "handling_days": 0.0, "cost_delta_per_unit": 0.0}
            )
        st.caption("Handling days and cost deltas are extra assumptions versus the current factory. Zero means not applied.")
        edited = st.data_editor(
            st.session_state["handling_table"],
            hide_index=True,
            width="stretch",
            disabled=["factory"],
            num_rows="fixed",
            key="handling_editor",
            column_config={
                "handling_days": st.column_config.NumberColumn("Handling days (assumption)", format="%.1f"),
                "cost_delta_per_unit": st.column_config.NumberColumn("Cost delta $ / unit (assumption)", format="%.2f"),
            },
        )
        st.session_state["handling_live"] = edited
        st.checkbox(
            "Rank by the assumption scenario instead of the historical model",
            key="rank_by_scenario",
            help="Leave off to rank on model-supported days saved. Turn on only when you want the labeled what-if to order the list.",
        )


def weight_note(assumptions: ScenarioAssumptions) -> str:
    if assumptions.profit_weight > 0 and not profit_inputs_present(assumptions):
        return (
            f"Requested weights are speed {assumptions.speed_weight:.0%} and profit {assumptions.profit_weight:.0%}. "
            "Applied score is speed-focused because profit impact is not quantified."
        )
    return (
        f"Applied weights: speed {assumptions.speed_weight:.0%}, profit {assumptions.profit_weight:.0%}, "
        f"risk {assumptions.risk_weight:.2f}."
    )


def effective_eligibility(base: pd.DataFrame) -> pd.DataFrame:
    edited = st.session_state.get("eligibility_table")
    if isinstance(edited, pd.DataFrame) and not edited.empty and "status" in edited.columns:
        frame = edited.copy()
        if "product_name_key" not in frame.columns:
            frame["product_name_key"] = frame["product_name"].map(normalize_product_name)
        return frame
    return base


def predictors(bundle: dict[str, Any]):
    def predict(frame: pd.DataFrame):
        return predict_frame(bundle, frame, "lead_time_selected")

    def alternate(frame: pd.DataFrame):
        return predict_frame(bundle, frame, "lead_time_alternate")

    return predict, alternate


def manual_destination() -> tuple[float | None, float | None, str]:
    if not st.session_state.get("use_manual_coord"):
        return None, None, "none"
    lat = float(st.session_state.get("sim_lat", 0.0))
    lon = float(st.session_state.get("sim_lon", 0.0))
    if lat == 0.0 and lon == 0.0:
        return None, None, "none"
    return lat, lon, "User-entered coordinate. Straight-line, not a road distance."


def centroid_destination(centroids: pd.DataFrame, state: str | None) -> tuple[float | None, float | None, str]:
    if not state or centroids is None or centroids.empty:
        return None, None, "none"
    hit = centroids.loc[centroids["state"] == state]
    if hit.empty:
        return None, None, "none"
    return (
        float(hit.iloc[0]["latitude"]),
        float(hit.iloc[0]["longitude"]),
        f"Approximate centroid of {state}. Not a customer coordinate. Not road distance.",
    )


def batch_destination_lookup(
    orders: pd.DataFrame,
    assumptions: ScenarioAssumptions,
    centroids: pd.DataFrame,
    geocodes: pd.DataFrame | None,
) -> dict[tuple[str, str], tuple[float, float, str]] | None:
    if not assumptions.distance_enabled:
        return None
    lookup: dict[tuple[str, str], tuple[float, float, str]] = {}
    if assumptions.coordinate_source == "state_centroid":
        if not assumptions.centroid_acknowledged:
            return None
        merged = orders.merge(centroids[["state", "latitude", "longitude"]], on="state", how="inner")
        if merged.empty:
            return None
        for (region, state), sub in merged.groupby(["region", "state"]):
            lookup[(str(region), str(state))] = (
                float(sub["latitude"].mean()),
                float(sub["longitude"].mean()),
                f"Approximate centroid of {state}. Straight-line exploratory proxy, not a customer location or road distance.",
            )
        for region, sub in merged.groupby("region"):
            lookup[(str(region), "*")] = (
                float(sub["latitude"].mean()),
                float(sub["longitude"].mean()),
                "Order-weighted mean of approximate state centroids. Straight-line exploratory proxy, not road distance.",
            )
        return lookup
    if assumptions.coordinate_source == "uploaded" and geocodes is not None and not geocodes.empty:
        work = orders.copy()
        work["city_key"] = work["city"].astype("string").str.strip().str.lower()
        work["state_key"] = work["state"].astype("string").str.strip().str.lower()
        matched = work.merge(geocodes, on=["city_key", "state_key"], how="inner")
        if matched.empty:
            return {}
        for (region, state), sub in matched.groupby(["region", "state"]):
            lookup[(str(region), str(state))] = (
                float(sub["latitude"].mean()),
                float(sub["longitude"].mean()),
                "Mean of uploaded coordinates for matched orders in this state. Straight-line, not road distance.",
            )
        for region, sub in matched.groupby("region"):
            lookup[(str(region), "*")] = (
                float(sub["latitude"].mean()),
                float(sub["longitude"].mean()),
                "Mean of uploaded coordinates for matched orders in this region. Straight-line, not road distance.",
            )
        return lookup
    return None


def ship_mode_options(orders: pd.DataFrame) -> list[str]:
    present = set(orders["ship_mode"].dropna().unique())
    ordered = [mode for mode in SHIP_MODE_ORDER if mode in present]
    return ordered + sorted(present - set(ordered))


def fmt_days(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{float(value):,.1f}"


def recommendation_view(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table
    columns = [
        "rank",
        "product_name",
        "division",
        "region",
        "ship_mode",
        "current_factory",
        "candidate_factory",
        "eligibility",
        "sample_size",
        "predicted_current_lead_time_days",
        "predicted_alternative_lead_time_days",
        "days_saved",
        "lead_time_change_pct",
        "scenario_days_saved",
        "scenario_change_pct",
        "evidence_level",
        "risk_flags",
        "financial_impact_status",
        "sensitivity_profit_delta",
        "recommendation_score",
        "speed_score",
        "profit_score",
        "risk_penalty",
        "historical_gross_profit",
        "explanation",
    ]
    present = [col for col in columns if col in table.columns]
    renamed = table.loc[:, present].rename(
        columns={
            "rank": "Rank",
            "product_name": "Product",
            "division": "Division",
            "region": "Region",
            "ship_mode": "Ship mode",
            "current_factory": "Current factory",
            "candidate_factory": "Candidate factory",
            "eligibility": "Eligibility",
            "sample_size": "Sample size",
            "predicted_current_lead_time_days": "Predicted current lead time (days)",
            "predicted_alternative_lead_time_days": "Predicted alternative lead time (days)",
            "days_saved": "Days saved (model)",
            "lead_time_change_pct": "Change % (model)",
            "scenario_days_saved": "Days saved (scenario)",
            "scenario_change_pct": "Change % (scenario)",
            "evidence_level": "Evidence level",
            "risk_flags": "Risk flags",
            "financial_impact_status": "Financial impact",
            "sensitivity_profit_delta": "Sensitivity $ (not validated)",
            "recommendation_score": "Recommendation score",
            "speed_score": "Speed score",
            "profit_score": "Profit score",
            "risk_penalty": "Risk penalty",
            "historical_gross_profit": "Historical gross profit",
            "explanation": "Explanation",
        }
    )
    return renamed


def reason_sentences(codes: list[str] | str) -> list[str]:
    if isinstance(codes, str):
        items = [part.strip() for part in codes.split(";") if part.strip()]
    else:
        items = list(codes)
    return [f"{code}: {REASON_TEXT.get(code, 'See the model page for the definition.')}" for code in items]
