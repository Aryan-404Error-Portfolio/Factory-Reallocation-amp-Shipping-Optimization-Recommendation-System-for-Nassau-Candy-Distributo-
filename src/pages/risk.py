"""Module E — Risk and impact."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ..data_loader import normalize_product_name
from ..recommendations import reason_glossary
from ..ui import callout, effective_eligibility, page_intro, show_table
from ..visualizations import margin_bars


def render_risk(ctx) -> None:
    page_intro(
        "Module E",
        "Risk and impact",
        "Low evidence, missing costs, and unverified factories stay visible. Historical margin is shown only because sales, cost, and gross profit reconcile.",
    )
    quality = ctx.quality
    cards = st.columns(4)
    cards[0].metric("Critical findings", f"{quality['critical_count']}")
    cards[1].metric("Warnings", f"{quality['warning_count']}")
    cards[2].metric("Rows deleted", "0", help="Extreme lead times are flagged, not dropped.")
    cards[3].metric("Profit impact of reassignment", "Not quantified")

    callout(
        "warning",
        "Missing inputs that block a financial claim",
        "Freight charges, carrier rates, factory handling cost, capacity, and inventory are not in the extract. "
        "A sensitivity dollar appears only after you type an assumption, and it is labeled as not financially validated. "
        "Product gross profit is not incremental logistics profit.",
    )

    st.subheader("Findings")
    for item in quality["warnings"]:
        kind = item["severity"] if item["severity"] in {"critical", "warning", "info"} else "neutral"
        count = "" if item.get("count") is None else f" Count: {item['count']:,}."
        callout(kind, item["code"].replace("_", " ").title(), item["message"] + count)

    st.subheader("Historical margin")
    st.caption("Margin is gross profit divided by sales. It is product economics, not the profit effect of a factory move.")
    st.plotly_chart(margin_bars(ctx.orders), width="stretch", config={"displayModeBar": False})
    margin = (
        ctx.orders.groupby("product_name", as_index=False)
        .agg(sales=("sales", "sum"), gross_profit=("gross_profit", "sum"), orders=("row_id", "size"))
    )
    margin["margin"] = margin["gross_profit"] / margin["sales"]
    thin = margin.loc[margin["margin"] < 0.15].sort_values("margin")
    if thin.empty:
        st.caption("No product in this extract has a historical gross margin below 15%.")
    else:
        callout(
            "warning",
            "Thin historical margin",
            "These products have gross margin below 15%. That is a commercial risk flag, not evidence that a factory change would help or hurt.",
        )
        show_table(thin.rename(columns={"product_name": "Product", "orders": "Orders", "sales": "Sales", "gross_profit": "Gross profit", "margin": "Margin"}))

    st.subheader("Factory eligibility")
    st.caption(
        "Current factory comes from the supplied mapping. This editor does not change that baseline. "
        "Mark an alternative eligible only after operations confirms the product can be made there. "
        "Unverified alternatives stay in the exploratory list and are not actionable."
    )
    base = effective_eligibility(ctx.eligibility)
    editor_cols = [col for col in ["product_name", "factory", "status", "source", "notes"] if col in base.columns]
    edited = st.data_editor(
        base.loc[:, editor_cols],
        hide_index=True,
        width="stretch",
        disabled=[col for col in ["product_name", "factory", "source"] if col in editor_cols],
        num_rows="fixed",
        key="eligibility_editor",
        column_config={
            "status": st.column_config.SelectboxColumn(
                "Status",
                options=["confirmed_current", "eligible", "requires_validation", "ineligible"],
                required=True,
            )
        },
    )
    edited = edited.copy()
    edited["product_name_key"] = edited["product_name"].map(normalize_product_name)
    st.session_state["eligibility_table"] = edited
    unverified = int((edited["status"] == "requires_validation").sum())
    ineligible = int((edited["status"] == "ineligible").sum())
    st.caption(f"{unverified} alternatives require validation. {ineligible} are marked ineligible and drop out of the simulator.")
    st.download_button(
        "Download eligibility table",
        edited.to_csv(index=False).encode("utf-8"),
        file_name="factory_eligibility.csv",
        mime="text/csv",
    )
    if st.button("Reset eligibility to the supplied file"):
        st.session_state.pop("eligibility_table", None)
        st.rerun()

    st.subheader("Reason codes")
    show_table(reason_glossary().rename(columns={"code": "Code", "meaning": "Plain-language meaning"}))
    st.caption("A flag is a sentence, not a color. Higher evidence is not used for a factory move unless origin factory was in the training data.")
