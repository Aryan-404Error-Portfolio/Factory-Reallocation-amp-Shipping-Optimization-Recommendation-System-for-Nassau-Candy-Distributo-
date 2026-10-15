"""Plotly charts. Color is never the only carrier of meaning; titles and labels do that work."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PAPER = "rgba(0,0,0,0)"
PLOT = "#fffdf9"
INK = "#1c1917"
GRID = "#f0ebe3"
ACCENT = "#9a3412"
TEAL = "#115e59"
AMBER = "#b45309"
SLATE = "#44403c"

SHIP_MODE_COLORS = {
    "Same Day": "#115e59",
    "First Class": "#0f766e",
    "Second Class": "#b45309",
    "Standard Class": "#9a3412",
}
REGION_COLORS = {
    "Atlantic": "#1e3a5f",
    "Gulf": "#0f766e",
    "Interior": "#b45309",
    "Pacific": "#9a3412",
}


def style_fig(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=PAPER,
        plot_bgcolor=PLOT,
        font={"family": "Segoe UI, Helvetica Neue, Arial, sans-serif", "color": INK, "size": 13},
        height=height,
        margin={"l": 8, "r": 8, "t": 56, "b": 8},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        title={"font": {"size": 16, "color": INK}},
    )
    fig.update_xaxes(showgrid=False, linecolor="#e7e0d6", zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#e7e0d6")
    return fig


def lead_time_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df.dropna(subset=["lead_time_days"]),
        x="lead_time_days",
        nbins=40,
        color_discrete_sequence=[ACCENT],
        labels={"lead_time_days": "Calendar lead time (days)"},
        title="Calendar lead time is trimodal, not a normal shipping delay",
    )
    fig.update_traces(name="Orders")
    style_fig(fig, 340)
    fig.update_layout(showlegend=False, bargap=0.08)
    return fig


def residual_by_ship_mode(df: pd.DataFrame) -> go.Figure:
    work = df.dropna(subset=["descriptive_residual_days", "ship_mode"])
    fig = px.box(
        work,
        x="ship_mode",
        y="descriptive_residual_days",
        color="ship_mode",
        color_discrete_map=SHIP_MODE_COLORS,
        category_orders={"ship_mode": list(SHIP_MODE_COLORS)},
        labels={
            "ship_mode": "Ship mode",
            "descriptive_residual_days": "Days versus order-id-year median",
        },
        title="After the year offset, only ship mode still moves lead time",
        points=False,
    )
    style_fig(fig, 340)
    fig.update_layout(showlegend=False)
    return fig


def median_lead_by_region(df: pd.DataFrame) -> go.Figure:
    work = (
        df.dropna(subset=["lead_time_days"])
        .groupby(["region", "ship_mode"], as_index=False)["lead_time_days"]
        .median()
    )
    fig = px.bar(
        work,
        x="region",
        y="lead_time_days",
        color="ship_mode",
        barmode="group",
        color_discrete_map=SHIP_MODE_COLORS,
        category_orders={"ship_mode": list(SHIP_MODE_COLORS), "region": list(REGION_COLORS)},
        labels={"lead_time_days": "Median calendar lead time (days)", "region": "Destination region", "ship_mode": "Ship mode"},
        title="Region does not separate the raw lead-time bands",
    )
    style_fig(fig, 340)
    return fig


def product_exposure(df: pd.DataFrame) -> go.Figure:
    work = (
        df.groupby("product_name", as_index=False)
        .agg(gross_profit=("gross_profit", "sum"), orders=("row_id", "size"))
        .sort_values("gross_profit")
    )
    fig = px.bar(
        work,
        x="gross_profit",
        y="product_name",
        orientation="h",
        color_discrete_sequence=[TEAL],
        labels={"gross_profit": "Historical gross profit", "product_name": "Product"},
        title="Exposure is concentrated in the chocolate bars",
        hover_data={"orders": True, "gross_profit": ":.0f"},
    )
    style_fig(fig, 460)
    fig.update_layout(showlegend=False, yaxis_title="")
    return fig


def variance_share(quality: dict) -> go.Figure:
    variance = quality.get("variance_decomposition", {})
    rows = [
        {"driver": "Order ID year (identifier, not a feature)", "r2": variance.get("order_id_year_r2") or 0, "target": "Raw lead time"},
        {"driver": "Ship mode", "r2": variance.get("ship_mode_r2_raw") or 0, "target": "Raw lead time"},
        {"driver": "Region", "r2": variance.get("region_r2_raw") or 0, "target": "Raw lead time"},
        {"driver": "Product", "r2": variance.get("product_r2_raw") or 0, "target": "Raw lead time"},
        {"driver": "Ship mode", "r2": variance.get("ship_mode_r2_residual") or 0, "target": "Within-cohort residual"},
        {"driver": "Region", "r2": variance.get("region_r2_residual") or 0, "target": "Within-cohort residual"},
        {"driver": "Product", "r2": variance.get("product_r2_residual") or 0, "target": "Within-cohort residual"},
    ]
    frame = pd.DataFrame(rows)
    fig = px.bar(
        frame,
        x="r2",
        y="driver",
        color="target",
        barmode="group",
        color_discrete_sequence=[ACCENT, TEAL],
        labels={"r2": "Share of variance explained by group means (R²)", "driver": ""},
        title="What actually lines up with lead time",
    )
    style_fig(fig, 380)
    fig.update_xaxes(range=[-0.05, 1.05])
    return fig


def factory_map(factories: pd.DataFrame) -> go.Figure:
    fig = px.scatter_geo(
        factories,
        lat="latitude",
        lon="longitude",
        text="factory",
        scope="usa",
        title="Supplied factory coordinates",
    )
    fig.update_traces(marker={"size": 12, "color": ACCENT, "line": {"width": 1, "color": INK}})
    fig.update_layout(
        paper_bgcolor=PAPER,
        font={"family": "Segoe UI, Helvetica Neue, Arial, sans-serif", "color": INK},
        height=360,
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        title={"font": {"size": 16}},
    )
    return fig


def simulator_bars(rows: pd.DataFrame) -> go.Figure:
    if rows.empty:
        return style_fig(go.Figure(), 320)
    melted = rows.melt(
        id_vars=["candidate_factory", "role"],
        value_vars=["predicted_alternative_lead_time_days"],
        var_name="measure",
        value_name="days",
    )
    melted["measure"] = "Model prediction"
    if rows["scenario_alternative_lead_time_days"].notna().any():
        extra = rows.melt(
            id_vars=["candidate_factory", "role"],
            value_vars=["scenario_alternative_lead_time_days"],
            var_name="measure",
            value_name="days",
        )
        extra["measure"] = "Assumption scenario"
        melted = pd.concat([melted, extra], ignore_index=True)
    melted["factory_label"] = melted.apply(
        lambda row: f"{row['candidate_factory']} (current)" if row["role"] == "current" else row["candidate_factory"],
        axis=1,
    )
    fig = px.bar(
        melted,
        x="factory_label",
        y="days",
        color="measure",
        barmode="group",
        text_auto=".1f",
        color_discrete_sequence=[SLATE, AMBER],
        labels={"days": "Lead time (days)", "factory_label": "Factory", "measure": ""},
        title="Same model context for every factory",
    )
    style_fig(fig, 380)
    return fig


def whatif_bars(rows: pd.DataFrame) -> go.Figure:
    return simulator_bars(rows)


def metrics_bars(candidates: dict, title: str) -> go.Figure:
    records = []
    for name, payload in candidates.items():
        test = payload.get("test", {})
        if test.get("mae") is None:
            continue
        records.append({"model": name, "mae": test["mae"], "split": "Test MAE"})
        cv = payload.get("cv", {})
        if cv.get("mae") is not None:
            records.append({"model": name, "mae": cv["mae"], "split": "CV MAE"})
    fig = px.bar(
        pd.DataFrame(records),
        x="model",
        y="mae",
        color="split",
        barmode="group",
        color_discrete_sequence=[ACCENT, TEAL],
        labels={"mae": "MAE (days)", "model": "Model"},
        title=title,
    )
    style_fig(fig, 360)
    return fig


def subgroup_mae_bar(records: list[dict], key: str, title: str) -> go.Figure:
    frame = pd.DataFrame(records)
    if frame.empty:
        return style_fig(go.Figure(), 300)
    fig = px.bar(
        frame.sort_values("mae"),
        x="mae",
        y=key,
        orientation="h",
        text_auto=".1f",
        color_discrete_sequence=[TEAL],
        labels={"mae": "Test MAE (days)", key: ""},
        title=title,
        hover_data={"n": True},
    )
    style_fig(fig, 340)
    fig.update_layout(showlegend=False)
    return fig


def importance_bar(records: list[dict]) -> go.Figure:
    frame = pd.DataFrame(records)
    if frame.empty:
        return style_fig(go.Figure(), 280)
    fig = px.bar(
        frame.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        color_discrete_sequence=[SLATE],
        labels={"importance": "Relative importance", "feature": ""},
        title="Selected-model importance (not a causal effect)",
    )
    style_fig(fig, 380)
    fig.update_layout(showlegend=False)
    return fig


def pred_vs_actual(y_true, y_pred, title: str) -> go.Figure:
    frame = pd.DataFrame({"actual": y_true, "predicted": y_pred})
    fig = px.scatter(
        frame,
        x="actual",
        y="predicted",
        opacity=0.35,
        color_discrete_sequence=[ACCENT],
        labels={"actual": "Actual lead time (days)", "predicted": "Predicted lead time (days)"},
        title=title,
    )
    low = float(min(frame["actual"].min(), frame["predicted"].min()))
    high = float(max(frame["actual"].max(), frame["predicted"].max()))
    fig.add_shape(type="line", x0=low, y0=low, x1=high, y1=high, line={"color": TEAL, "dash": "dash"})
    style_fig(fig, 380)
    fig.update_layout(showlegend=False)
    return fig


def margin_bars(df: pd.DataFrame) -> go.Figure:
    work = (
        df.groupby("product_name", as_index=False)
        .agg(sales=("sales", "sum"), gross_profit=("gross_profit", "sum"), orders=("row_id", "size"))
    )
    work["margin"] = work["gross_profit"] / work["sales"]
    work = work.sort_values("margin")
    fig = px.bar(
        work,
        x="margin",
        y="product_name",
        orientation="h",
        color_discrete_sequence=[AMBER],
        labels={"margin": "Gross margin (gross profit / sales)", "product_name": ""},
        title="Historical product margin, where the identity holds",
        hover_data={"orders": True},
    )
    fig.update_xaxes(tickformat=".0%")
    style_fig(fig, 440)
    fig.update_layout(showlegend=False)
    return fig
