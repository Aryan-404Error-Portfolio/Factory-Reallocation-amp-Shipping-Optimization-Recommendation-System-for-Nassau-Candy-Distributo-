"""Module D — Recommendation dashboard."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from ..config import ADVISORY_TEXT
from ..simulator import build_recommendations
from ..ui import (
    assumptions_from_state,
    batch_destination_lookup,
    callout,
    effective_eligibility,
    page_intro,
    predictors,
    recommendation_view,
    render_assumption_controls,
    reason_sentences,
    ship_mode_options,
    show_table,
    weight_note,
)


def _table(ctx, assumptions) -> pd.DataFrame:
    eligibility = effective_eligibility(ctx.eligibility)
    geocodes = st.session_state.get("geocodes")
    lookup = batch_destination_lookup(ctx.orders, assumptions, ctx.centroids, geocodes)
    key = (
        assumptions.to_dict(),
        eligibility[["product_name_key", "factory", "status"]].to_csv(index=False) if "product_name_key" in eligibility else eligibility.to_csv(index=False),
        None if lookup is None else sorted((str(k), v[0], v[1], v[2]) for k, v in lookup.items()),
        ctx.dataset_sha256,
    )
    cache = st.session_state.get("rec_cache")
    if isinstance(cache, dict) and cache.get("key") == key:
        return cache["table"]
    predict, alternate = predictors(ctx.bundle)
    with st.spinner("Scoring observed product, region, and ship-mode lanes."):
        table = build_recommendations(
            ctx.orders,
            ctx.mapping,
            eligibility,
            ctx.factories,
            assumptions,
            predict,
            alternate_predict_fn=alternate,
            feature_columns=ctx.bundle["feature_columns"],
            destination_lookup=lookup,
        )
    st.session_state["rec_cache"] = {"key": key, "table": table}
    return table


def render_recommendations(ctx) -> None:
    page_intro(
        "Module D",
        "Recommendation dashboard",
        "Ranked alternatives for observed lanes. A row is an investigation item unless it meets the actionable-evidence rule. Financial impact stays 'Not quantified' when cost inputs are missing.",
    )
    if ctx.bundle is None:
        callout("warning", "No model is loaded", ctx.model_error or "Train a model before ranking recommendations.")
        return
    render_assumption_controls(ctx.factories)
    assumptions = assumptions_from_state(ctx.bundle, ctx.factories)
    st.caption(weight_note(assumptions))
    if assumptions.rank_by_scenario:
        callout(
            "warning",
            "Ranking basis: assumption scenario",
            "Order is using your distance or handling assumptions. It is not a historical factory effect. Evidence stays at Requires validation.",
        )
    else:
        callout(
            "info",
            "Ranking basis: historical model",
            "Days saved are model-supported. On this extract they are zero for every factory, because origin was not observed. Ties break toward higher historical gross profit so large lanes can be reviewed first. Exposure is not a profit-impact score.",
        )

    table = _table(ctx, assumptions)
    if table.empty:
        callout("warning", "No recommendations", "No mapped product had an eligible or requires-validation alternative.")
        return

    filters = st.columns(6)
    regions = ["All"] + sorted(table["region"].dropna().unique())
    modes = ["All"] + ship_mode_options(ctx.orders)
    divisions = ["All"] + sorted(table["division"].dropna().unique())
    evidence = ["All"] + sorted(table["evidence_level"].dropna().unique())
    eligibility = ["All"] + sorted(table["eligibility"].dropna().unique())
    region = filters[0].selectbox("Region", regions, key="rec_region")
    ship_mode = filters[1].selectbox("Ship mode", modes, key="rec_mode")
    division = filters[2].selectbox("Division", divisions, key="rec_division")
    evidence_choice = filters[3].selectbox("Evidence", evidence, key="rec_evidence")
    eligibility_choice = filters[4].selectbox("Eligibility", eligibility, key="rec_eligibility")
    min_gain = filters[5].number_input("Minimum days saved", value=0.0, step=0.5, key="rec_min_gain")
    query = st.text_input("Product contains", key="rec_query")
    sort_by = st.selectbox(
        "Sort by",
        ["Recommendation score", "Days saved", "Historical gross profit", "Sample size"],
        key="rec_sort",
    )

    filtered = table.copy()
    if region != "All":
        filtered = filtered.loc[filtered["region"] == region]
    if ship_mode != "All":
        filtered = filtered.loc[filtered["ship_mode"] == ship_mode]
    if division != "All":
        filtered = filtered.loc[filtered["division"] == division]
    if evidence_choice != "All":
        filtered = filtered.loc[filtered["evidence_level"] == evidence_choice]
    if eligibility_choice != "All":
        filtered = filtered.loc[filtered["eligibility"] == eligibility_choice]
    if query:
        filtered = filtered.loc[filtered["product_name"].str.contains(query, case=False, na=False)]
    gain_col = "scenario_days_saved" if assumptions.rank_by_scenario and filtered["scenario_days_saved"].notna().any() else "days_saved"
    filtered = filtered.loc[filtered[gain_col].fillna(-10**6) >= float(min_gain)]
    sort_map = {
        "Recommendation score": "recommendation_score",
        "Days saved": gain_col,
        "Historical gross profit": "historical_gross_profit",
        "Sample size": "sample_size",
    }
    filtered = filtered.sort_values([sort_map[sort_by], "historical_gross_profit"], ascending=[False, False]).reset_index(drop=True)
    filtered["rank"] = range(1, len(filtered) + 1)

    actionable = int(filtered["actionable"].sum()) if not filtered.empty else 0
    flagged = 0 if filtered.empty else int(filtered["risk_flags"].fillna("").astype(str).str.len().gt(0).mean() * 100)
    coverage = 1.0 if not table.empty else 0.0
    kpis = st.columns(4)
    kpis[0].metric("Rows in view", f"{len(filtered):,}")
    kpis[1].metric("Actionable in view", f"{actionable:,}")
    kpis[2].metric("Risk-flag rate", f"{flagged:.0f}%", help="Share of visible rows with at least one risk flag. Text flags are the signal, not color.")
    kpis[3].metric("Prediction coverage", f"{coverage:.0%}", help="Observed mapped lanes with a usable prediction, divided by requested lanes.")

    if filtered.empty:
        callout(
            "warning",
            "No rows match these filters",
            "If the minimum days saved is above zero while ranking on the historical model, an empty table is expected. The model does not support a factory gain on this extract.",
        )
        return

    show_table(recommendation_view(filtered), height=520)
    labels = [
        f"{int(row.rank)}. {row.product_name} → {row.candidate_factory} ({row.region}, {row.ship_mode})"
        for row in filtered.head(200).itertuples(index=False)
    ]
    choice = st.selectbox("Inspect a row", labels)
    chosen = filtered.iloc[labels.index(choice)]
    st.markdown(f"**{chosen['evidence_level']} · {chosen['financial_impact_status']}**")
    st.write(chosen["explanation"])
    for sentence in reason_sentences(chosen["reason_codes"]):
        st.markdown(f"- {sentence}")

    export = filtered.copy()
    export["generated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    export["advisory"] = ADVISORY_TEXT
    export["dataset_sha256"] = ctx.dataset_sha256
    export["ranking_basis"] = "assumption_scenario" if assumptions.rank_by_scenario else "model"
    if "reason_codes" in export.columns:
        export["reason_codes"] = export["reason_codes"].apply(lambda value: "; ".join(value) if isinstance(value, list) else value)
    st.download_button(
        "Download filtered recommendations",
        export.to_csv(index=False).encode("utf-8"),
        file_name="factory_recommendations.csv",
        mime="text/csv",
    )
    st.caption("The export includes weights, evidence, risk flags, model version, and the advisory. It does not include customer identifiers.")
