"""Module A — Overview."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ..clustering import cluster_routes
from ..features import slow_segments
from ..reporting import executive_findings, validation_markdown
from ..ui import callout, model_label, page_intro, show_table
from ..visualizations import (
    factory_map,
    lead_time_histogram,
    median_lead_by_region,
    product_exposure,
    residual_by_ship_mode,
    variance_share,
)


def render_overview(ctx) -> None:
    quality = ctx.quality
    orders = ctx.orders
    lead = quality["lead_time_profile"]
    peaks = ", ".join(f"{item['rows']:,} orders {item['label']}" for item in lead.get("peaks", []))
    page_intro(
        "Module A",
        "Overview",
        "A desk for testing whether a different factory could ship faster, without pretending the current extract already proves it.",
    )
    callout(
        "critical",
        "Unresolved date anomaly — read this before the charts",
        (
            f"{lead['n_valid']:,} lead times parse cleanly as day-month-year and then run from "
            f"{lead['min']:.0f} to {lead['max']:.0f} days (median {lead['median']:.0f}). "
            f"The mass sits in three bands: {peaks}. "
            "Rows were not deleted. These figures are not a credible candy ship-time ETA until the source owner confirms the dates."
        ),
    )
    raw_mae = None
    selected = None
    if ctx.bundle:
        selected = ctx.bundle["targets"]["lead_time_days"]["selected_model"]
        raw_mae = ctx.bundle["targets"]["lead_time_days"]["candidates"][selected]["test"]["mae"]
    actionable = 0 if not ctx.bundle or not ctx.bundle.get("factory_effect_learned") else None
    cards = st.columns(5)
    cards[0].metric("Valid records", f"{lead['n_valid']:,}", help="Rows with a parseable ship date and order date.")
    cards[1].metric("Median lead time", f"{lead['median']:.0f} days", help="Calendar days, ship date minus order date.")
    cards[2].metric(
        "Test MAE",
        "—" if raw_mae is None else f"{raw_mae:.1f} days",
        help="Held-out mean absolute error of the selected raw-target model. This describes the date anomaly, not a delivery promise.",
    )
    cards[3].metric(
        "Actionable reassignments",
        "0" if actionable == 0 else "—",
        help="A reassignment is actionable only with a learned factory effect, confirmed eligibility, and a positive gain. This extract has no origin factory.",
    )
    cards[4].metric("Data-quality warnings", f"{quality['warning_count']}", help="Critical and warning findings. Info notes are listed separately.")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(lead_time_histogram(orders), width="stretch", config={"displayModeBar": False})
        st.caption("Three spikes, not a smooth shipping distribution. The model cannot turn this into a factory ranking.")
    with right:
        st.plotly_chart(residual_by_ship_mode(orders), width="stretch", config={"displayModeBar": False})
        st.caption("Within each Order ID year, ship mode shifts lead time by a few days. Region and product do not.")

    st.plotly_chart(median_lead_by_region(orders), width="stretch", config={"displayModeBar": False})
    st.caption("Grouped medians by region and ship mode. The bands stay in the same place. That is a descriptive pattern, not a cause.")

    chart_left, chart_right = st.columns([1.15, 0.85])
    with chart_left:
        st.plotly_chart(variance_share(quality), width="stretch", config={"displayModeBar": False})
    with chart_right:
        st.plotly_chart(factory_map(ctx.factories), width="stretch", config={"displayModeBar": False})
        st.caption("Supplied factory coordinates. Orders do not record which factory shipped them.")

    st.subheader("What can be said today")
    say, cannot, next_step = st.columns(3)
    with say:
        callout(
            "info",
            "Supported",
            "Historical margin can be shown, because gross profit matches sales minus cost. Ship mode is the only stable within-cohort lead-time difference, and it is a few days. Sample sizes are large for the chocolate bars.",
        )
    with cannot:
        callout(
            "warning",
            "Not supported",
            "A claim that moving a product to another factory will shorten delivery or raise profit. Origin factory, customer coordinates, freight, capacity, and factory cost are missing. The raw lead-time target does not generalize forward.",
        )
    with next_step:
        callout(
            "neutral",
            "Collect before a pilot",
            "Actual origin factory, confirmed product-factory eligibility, customer coordinates or route distance, freight, factory cost, and a confirmed meaning for the ship date.",
        )

    findings = executive_findings(quality, ctx.bundle)
    with st.expander("Findings in plain language", expanded=False):
        for finding in findings:
            callout(finding["severity"] if finding["severity"] != "info" else "info", finding["title"], finding["body"])

    st.subheader("Lanes worth instrumenting")
    st.caption(
        "Sorted by historical gross profit, not by a predicted factory gain. Model-supported days saved is 0 because factory origin was not observed. "
        "Use this list to decide where to start collecting origin data. It is not a reassignment ranking."
    )
    lanes = (
        orders.groupby(["product_name", "region"], as_index=False)
        .agg(
            orders=("row_id", "size"),
            units=("units", "sum"),
            sales=("sales", "sum"),
            gross_profit=("gross_profit", "sum"),
            median_lead_time_days=("lead_time_days", "median"),
        )
        .merge(ctx.mapping[["product_name", "current_factory"]], on="product_name", how="left")
        .sort_values("gross_profit", ascending=False)
        .head(12)
    )
    lanes.insert(0, "model_supported_days_saved", 0.0)
    lanes["financial_impact"] = "Not quantified"
    show_table(
        lanes.rename(
            columns={
                "product_name": "Product",
                "region": "Region",
                "orders": "Orders",
                "units": "Units",
                "sales": "Sales",
                "gross_profit": "Gross profit",
                "median_lead_time_days": "Median lead time (days)",
                "current_factory": "Assumed current factory",
                "model_supported_days_saved": "Model-supported days saved",
                "financial_impact": "Financial impact",
            }
        ),
        height=360,
    )

    slow = slow_segments(orders)
    material = int(slow["materially_slower"].sum()) if not slow.empty else 0
    st.subheader("Slow product-region-ship-mode cells")
    if material == 0:
        callout(
            "info",
            "No material route exception after the year offset",
            "No product-region-ship-mode cell with at least 30 orders is more than one day slower than its own ship-mode norm. Raw lead-time gaps are the date artifact, not a slow factory.",
        )
    else:
        show_table(slow.loc[slow["materially_slower"]].head(20))
    with st.expander("Largest residual gaps, including small samples", expanded=False):
        show_table(slow.head(15))

    clustered, cluster_summary = cluster_routes(orders)
    st.subheader("Route patterns")
    st.caption(cluster_summary.get("note", "Clustering needs more groups above the sample threshold."))
    if cluster_summary.get("status") == "ok":
        summary = (
            clustered.dropna(subset=["cluster_label"])
            .groupby("cluster_label", as_index=False)
            .agg(routes=("product_name", "size"), orders=("sample_size", "sum"), median_residual=("median_residual_days", "median"))
        )
        show_table(summary)

    st.download_button(
        "Download validation report",
        validation_markdown(quality).encode("utf-8"),
        file_name="nassau_validation_report.md",
        mime="text/markdown",
    )
    if ctx.bundle:
        st.caption(f"Loaded model: {model_label(selected)}. Dataset {ctx.dataset_sha256[:12]}.")
    elif ctx.model_error:
        callout("warning", "Model not loaded", ctx.model_error)
