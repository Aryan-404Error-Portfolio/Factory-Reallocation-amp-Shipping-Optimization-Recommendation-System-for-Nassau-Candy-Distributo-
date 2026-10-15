"""Streamlit entry point for the Nassau Candy factory-reallocation desk."""

from __future__ import annotations

from io import BytesIO

import streamlit as st

from src.app_state import (
    AppContext,
    cached_bundle,
    cached_default_orders,
    ensure_reference_files,
    train_default_bundle,
)
from src.config import DEFAULT_ORDERS_PATH, MODEL_VERSION, MODELS_DIR
from src.data_loader import bytes_sha256, file_sha256, load_centroids, load_eligibility, load_factories, load_geocodes, load_mapping, load_orders, validate_orders
from src.pages.model_quality import render_model_quality
from src.pages.overview import render_overview
from src.pages.recommendations_page import render_recommendations
from src.pages.risk import render_risk
from src.pages.simulator_page import render_simulator
from src.pages.whatif import render_whatif
from src.theme import inject_css
from src.ui import PAGES, footer, init_state, model_label, render_header

RENDERERS = {
    "Overview": render_overview,
    "Factory simulator": render_simulator,
    "What-if analysis": render_whatif,
    "Recommendations": render_recommendations,
    "Risk and impact": render_risk,
    "Model and data quality": render_model_quality,
}


def _capture_uploads() -> None:
    st.sidebar.markdown("**Data**")
    orders_file = st.sidebar.file_uploader("Orders extract", type=["csv"], key="orders_uploader", help="Optional. The supplied Nassau Candy file loads if this is empty.")
    if orders_file is not None:
        payload = orders_file.getvalue()
        digest = bytes_sha256(payload)
        if st.session_state.get("upload_sha") != digest:
            frame = load_orders(BytesIO(payload))
            st.session_state["uploaded_orders"] = frame
            st.session_state["uploaded_quality"] = validate_orders(frame)
            st.session_state["upload_sha"] = digest
            st.session_state["upload_name"] = orders_file.name
            st.session_state["session_bundle"] = None
            st.session_state["session_bundle_sha"] = None
    if st.session_state.get("uploaded_orders") is not None and st.sidebar.button("Use supplied extract"):
        for key in ("uploaded_orders", "uploaded_quality", "upload_sha", "upload_name", "session_bundle", "session_bundle_sha"):
            st.session_state.pop(key, None)
        st.rerun()

    geocode_file = st.sidebar.file_uploader(
        "Customer coordinates",
        type=["csv"],
        key="geo_uploader",
        help="Optional. Columns: City, State/Province, Latitude, Longitude. Not used unless you enable a distance scenario.",
    )
    if geocode_file is not None:
        try:
            st.session_state["geocodes"] = load_geocodes(geocode_file)
            st.session_state["geocode_error"] = None
        except Exception as exc:  # noqa: BLE001 - show the file error instead of crashing the desk
            st.session_state["geocode_error"] = str(exc)
            st.session_state["geocodes"] = None
    if st.session_state.get("geocode_error"):
        st.sidebar.warning(st.session_state["geocode_error"])
    geocodes = st.session_state.get("geocodes")
    if geocodes is not None and not geocodes.empty:
        st.sidebar.caption(f"{len(geocodes):,} coordinate rows loaded. Straight-line only.")


def _build_context() -> AppContext:
    ensure_reference_files()
    factories = load_factories()
    mapping = load_mapping()
    eligibility = load_eligibility()
    centroids = load_centroids()
    uploaded = st.session_state.get("uploaded_orders")
    if uploaded is not None:
        orders = uploaded
        quality = st.session_state["uploaded_quality"]
        digest = st.session_state.get("upload_sha", "")
        label = st.session_state.get("upload_name", "Uploaded extract")
        using_default = False
        bundle = st.session_state.get("session_bundle")
        if st.session_state.get("session_bundle_sha") != digest:
            bundle = None
        model_error = None if bundle is not None else "This upload has no trained model yet. Retrain it on Model and data quality."
    else:
        digest = file_sha256(DEFAULT_ORDERS_PATH)
        orders, quality = cached_default_orders(str(DEFAULT_ORDERS_PATH), digest)
        label = "Supplied Nassau Candy extract"
        using_default = True
        bundle = cached_bundle(str(MODELS_DIR), digest, MODEL_VERSION)
        model_error = None
        if bundle is None and quality.get("training_allowed") and not st.session_state.get("train_failed"):
            try:
                with st.spinner("Training models on the supplied extract. This is saved for the next launch."):
                    train_default_bundle(orders, quality, digest)
                bundle = cached_bundle(str(MODELS_DIR), digest, MODEL_VERSION)
            except Exception as exc:  # noqa: BLE001 - keep descriptive analysis available
                st.session_state["train_failed"] = str(exc)
                model_error = str(exc)
        elif bundle is None:
            model_error = st.session_state.get("train_failed") or "Saved model does not match this extract."
    return AppContext(
        orders=orders,
        quality=quality,
        factories=factories,
        mapping=mapping,
        eligibility=eligibility,
        centroids=centroids,
        bundle=bundle,
        dataset_label=label,
        dataset_sha256=digest,
        model_error=model_error,
        using_default_dataset=using_default,
    )


def _sidebar_status(ctx: AppContext) -> None:
    st.sidebar.divider()
    st.sidebar.markdown("**Status**")
    st.sidebar.caption(ctx.dataset_label)
    st.sidebar.caption(f"{ctx.quality['n_rows']:,} rows · {ctx.dataset_sha256[:12]}")
    if ctx.bundle:
        selected = ctx.bundle["targets"]["lead_time_days"]["selected_model"]
        st.sidebar.caption(f"Model {ctx.bundle.get('model_version')} · {model_label(selected)}")
        st.sidebar.caption("Factory effect learned: no" if not ctx.bundle.get("factory_effect_learned") else "Factory effect learned: yes")
    else:
        st.sidebar.caption("Model not loaded")
    st.sidebar.caption("Advisory only. This desk does not change a live assignment.")


def main() -> None:
    st.set_page_config(
        page_title="Factory Reallocation Desk · Nassau Candy",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    init_state()
    st.sidebar.markdown("**Navigate**")
    page = st.sidebar.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
    _capture_uploads()
    ctx = _build_context()
    render_header()
    _sidebar_status(ctx)
    RENDERERS[page](ctx)
    footer(ctx.bundle.get("model_version") if ctx.bundle else MODEL_VERSION)


if __name__ == "__main__":
    main()
