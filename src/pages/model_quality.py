"""Module F — Model and data quality."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ..clustering import cluster_routes
from ..config import FORBIDDEN_FEATURES, MODEL_VERSION
from ..features import apply_train_residual, model_ready_frame
from ..reporting import model_card_markdown, validation_markdown
from ..train import jsonable, predict_frame
from ..ui import callout, model_label, page_intro, preview_orders, show_table
from ..visualizations import importance_bar, metrics_bars, pred_vs_actual, subgroup_mae_bar


def _metrics_frame(candidates: dict) -> pd.DataFrame:
    rows = []
    for name, payload in candidates.items():
        cv = payload.get("cv", {})
        test = payload.get("test", {})
        train = payload.get("train", {})
        rows.append(
            {
                "Model": model_label(name),
                "CV MAE": cv.get("mae"),
                "Train MAE": train.get("mae"),
                "Test MAE": test.get("mae"),
                "Test RMSE": test.get("rmse"),
                "Test R²": test.get("r2"),
                "Test rows": test.get("n"),
            }
        )
    return pd.DataFrame(rows)


def render_model_quality(ctx) -> None:
    page_intro(
        "Module F",
        "Model and data quality",
        "Held-out error, the baseline comparison, leakage checks, and the date anomaly. If training did not run, that is stated here rather than hidden.",
    )
    quality = ctx.quality
    if not quality.get("training_allowed"):
        callout("critical", "Training is blocked", " ".join(quality.get("training_blockers") or ["Target fields are unusable."]))
    if ctx.model_error and ctx.bundle is None:
        callout("warning", "Model not trained", ctx.model_error)

    retrain = st.button("Retrain on the loaded extract")
    if retrain:
        if not quality.get("training_allowed"):
            st.error("Training is blocked by the validation checks above.")
        else:
            from ..app_state import clear_model_cache, train_default_bundle
            from ..train import train_and_evaluate

            with st.spinner("Training baseline, linear regression, random forest, and gradient boosting on a chronological split."):
                if ctx.using_default_dataset:
                    train_default_bundle(ctx.orders, quality, ctx.dataset_sha256)
                else:
                    bundle = train_and_evaluate(ctx.orders, quality, ctx.dataset_sha256, tiny=False)
                    st.session_state["session_bundle"] = bundle
                    st.session_state["session_bundle_sha"] = ctx.dataset_sha256
            clear_model_cache()
            st.rerun()

    if ctx.bundle is None:
        st.subheader("Extract preview")
        st.caption("Customer ID and postal code are hidden.")
        show_table(preview_orders(ctx.orders))
        return

    bundle = ctx.bundle
    split = bundle["split"]
    raw = bundle["targets"]["lead_time_days"]
    residual = bundle["targets"]["operational_residual_days"]
    selected = raw["selected_model"]
    selected_metrics = raw["candidates"][selected]["test"]
    baseline_metrics = raw["candidates"]["group_median_baseline"]["test"]

    cards = st.columns(4)
    cards[0].metric("Selected raw model", model_label(selected))
    cards[1].metric("Test MAE", f"{selected_metrics['mae']:.1f} days")
    cards[2].metric("Test RMSE", f"{selected_metrics['rmse']:.1f} days")
    cards[3].metric("Test R²", f"{selected_metrics['r2']:.3f}", help="R² can be negative when the model is worse than predicting the test-set mean.")

    callout(
        "warning",
        "The holdout is a different lead-time regime",
        (
            f"Training orders run {split['train_start']} to {split['train_end']} "
            f"(median lead time {split['train_target_median']:.0f} days). "
            f"The test period starts {split['cutoff']} and ends {split['test_end']} "
            f"(median {split['test_target_median']:.0f} days). "
            f"Ship-mode median baseline test MAE is {baseline_metrics['mae']:.1f} days. "
            "Complex models did not beat that baseline on the raw target. "
            "The error is the date anomaly failing to generalize, not a factory ranking."
        ),
    )
    st.caption(raw.get("selection_reason", ""))
    st.caption(split.get("grouping_note", ""))

    st.subheader("Raw lead time")
    show_table(_metrics_frame(raw["candidates"]))
    st.plotly_chart(metrics_bars(raw["candidates"], "Raw target — lower MAE is better, and still not an ETA"), width="stretch", config={"displayModeBar": False})

    st.subheader("Within-cohort residual (sensitivity, not a replacement target)")
    st.caption(residual.get("selection_reason", ""))
    show_table(_metrics_frame(residual["candidates"]))
    residual_selected = residual["candidates"][residual["selected_model"]]["test"]
    st.caption(
        f"Selected residual model: {model_label(residual['selected_model'])}. "
        f"Test MAE {residual_selected['mae']:.2f} days. "
        "This is the few-day ship-mode pattern left after removing the Order ID year median. It is not a factory effect."
    )

    ready = model_ready_frame(ctx.orders)
    cutoff = pd.Timestamp(split["cutoff"])
    test = ready.loc[ready["order_date"] >= cutoff].copy()
    if not test.empty:
        predicted = predict_frame(bundle, test, "lead_time_selected")
        st.plotly_chart(
            pred_vs_actual(test["lead_time_days"], predicted, "Test set: actual calendar lead time versus selected model"),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.caption("A flat prediction against two actual bands means the model cannot see the cohort shift. The dashed line is perfect prediction.")
        test = apply_train_residual(test, bundle.get("cohort_medians_train", {}), bundle.get("global_median_train", 0))
        residual_pred = predict_frame(bundle, test, "residual_selected")
        st.plotly_chart(
            pred_vs_actual(
                test["operational_residual_days"],
                residual_pred,
                "Sensitivity: within-cohort residual versus residual model",
            ),
            width="stretch",
            config={"displayModeBar": False},
        )

    sub_left, sub_right = st.columns(2)
    with sub_left:
        st.plotly_chart(
            subgroup_mae_bar(raw["subgroup_mae"].get("region", []), "region", "Test MAE by region"),
            width="stretch",
            config={"displayModeBar": False},
        )
    with sub_right:
        st.plotly_chart(
            subgroup_mae_bar(raw["subgroup_mae"].get("ship_mode", []), "ship_mode", "Test MAE by ship mode"),
            width="stretch",
            config={"displayModeBar": False},
        )
    st.caption("Subgroup error is still measured in days on the anomalous target. Similar error across regions is not evidence that region is irrelevant to a real carrier ETA. It means this target does not vary by region.")

    importance = raw.get("feature_importance") or []
    if importance:
        st.plotly_chart(importance_bar(importance), width="stretch", config={"displayModeBar": False})
        st.caption("Importance is association inside the fitted model, not a causal factory or region effect.")
    else:
        callout(
            "info",
            "No feature-importance vector",
            f"The selected raw model is {model_label(selected)}. A ship-mode median has no coefficient list. Ship mode is the only input it uses.",
        )

    st.subheader("Leakage check")
    audit = bundle.get("leakage_audit", {})
    if audit.get("passed"):
        callout("info", "Passed", audit.get("rule", "Forbidden post-outcome fields are not inputs."))
    else:
        callout("critical", "Failed", "Forbidden features found: " + ", ".join(audit.get("forbidden_found", [])))
    st.write("Features used: " + ", ".join(bundle.get("feature_columns", [])))
    st.write("Explicitly excluded: " + ", ".join(sorted(FORBIDDEN_FEATURES)))
    st.write(f"Factory effect learned: {bundle.get('factory_effect_learned')}. Origin column: {bundle.get('origin_column') or 'none'}.")

    st.subheader("Outlier treatment")
    treatment = bundle.get("outlier_treatment", {})
    callout(
        "warning",
        treatment.get("rule", "Outlier rule"),
        (
            f"Flagged {treatment.get('rows_flagged', 0):,} rows. "
            f"Retained if excluded: {treatment.get('rows_retained_if_excluded', 0):,}. "
            f"Deleted: {treatment.get('rows_deleted', 0)}. "
            + treatment.get("reason", "")
        ),
    )

    st.subheader("Split and version")
    st.json(
        {
            "model_version": bundle.get("model_version", MODEL_VERSION),
            "trained_at": bundle.get("trained_at"),
            "dataset_sha256": bundle.get("dataset_sha256"),
            "seed": bundle.get("seed"),
            "split": split,
            "target_credibility": bundle.get("target_credibility"),
        }
    )

    clustered, summary = cluster_routes(ctx.orders)
    st.subheader("Route clusters")
    st.caption(summary.get("note", ""))
    if summary.get("status") == "ok":
        show_table(
            clustered.dropna(subset=["cluster_label"])
            .groupby(["cluster_label", "ship_mode"], as_index=False)
            .agg(routes=("product_name", "size"), orders=("sample_size", "sum"))
        )

    st.subheader("Extract preview")
    st.caption("Customer identifiers are omitted. Postal code is omitted. Source product spelling is kept in product_name_source.")
    show_table(preview_orders(ctx.orders, 20))

    card = model_card_markdown({k: v for k, v in bundle.items() if k != "models"}, quality)
    downloads = st.columns(3)
    downloads[0].download_button("Download model card", card.encode("utf-8"), file_name="model_card.md", mime="text/markdown")
    downloads[1].download_button(
        "Download metrics JSON",
        json.dumps(jsonable({k: v for k, v in bundle.items() if k != "models"}), indent=2).encode("utf-8"),
        file_name="metrics.json",
        mime="application/json",
    )
    downloads[2].download_button(
        "Download validation report",
        validation_markdown(quality).encode("utf-8"),
        file_name="validation_report.md",
        mime="text/markdown",
    )
