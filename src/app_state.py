"""Cached loading and the shared application context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from .config import (
    CENTROIDS_PATH,
    DEFAULT_ORDERS_PATH,
    ELIGIBILITY_PATH,
    FACTORY_REFERENCE_PATH,
    MODEL_VERSION,
    MODELS_DIR,
    PRODUCT_MAPPING_PATH,
)
from .data_loader import (
    file_sha256,
    load_centroids,
    load_eligibility,
    load_factories,
    load_geocodes,
    load_mapping,
    load_orders,
    validate_orders,
    write_reference_files,
)
from .train import load_bundle, save_bundle, train_and_evaluate


@dataclass
class AppContext:
    orders: pd.DataFrame
    quality: dict[str, Any]
    factories: pd.DataFrame
    mapping: pd.DataFrame
    eligibility: pd.DataFrame
    centroids: pd.DataFrame
    bundle: dict[str, Any] | None
    dataset_label: str
    dataset_sha256: str
    model_error: str | None
    using_default_dataset: bool


@st.cache_data(show_spinner=False)
def cached_default_orders(path: str, sha: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    del sha  # cache key
    frame = load_orders(path)
    return frame, validate_orders(frame)


@st.cache_resource(show_spinner=False)
def cached_bundle(directory: str, sha: str, version: str) -> dict[str, Any] | None:
    bundle = load_bundle(MODELS_DIR if directory == str(MODELS_DIR) else MODELS_DIR)
    if not bundle:
        return None
    if bundle.get("dataset_sha256") != sha or bundle.get("model_version") != version:
        return None
    return bundle


def clear_model_cache() -> None:
    cached_bundle.clear()


def ensure_reference_files() -> None:
    if not all(path.exists() for path in (FACTORY_REFERENCE_PATH, PRODUCT_MAPPING_PATH, ELIGIBILITY_PATH, CENTROIDS_PATH)):
        write_reference_files()


def train_default_bundle(orders: pd.DataFrame, quality: dict[str, Any], sha: str) -> dict[str, Any]:
    bundle = train_and_evaluate(orders, quality, sha, tiny=False)
    save_bundle(bundle, quality=quality)
    clear_model_cache()
    return bundle


def read_geocode_upload(upload: Any) -> pd.DataFrame:
    return load_geocodes(upload)
