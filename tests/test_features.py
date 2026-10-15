"""Leakage checks, splits, and segment calculations."""

from __future__ import annotations

import pandas as pd

from src.config import BASE_FEATURES, FORBIDDEN_FEATURES
from src.features import chronological_split, leakage_audit, slow_segments
from src.data_loader import validate_orders


def test_base_features_exclude_leakage():
    audit = leakage_audit(BASE_FEATURES)
    assert audit["passed"]
    assert not (set(BASE_FEATURES) & FORBIDDEN_FEATURES)


def test_leakage_audit_catches_target_and_identifiers():
    audit = leakage_audit(BASE_FEATURES + ["lead_time_days", "customer_id", "cohort_year"])
    assert audit["passed"] is False
    assert "lead_time_days" in audit["forbidden_found"]
    assert "customer_id" in audit["forbidden_found"]
    assert "cohort_year" in audit["forbidden_found"]


def test_chronological_split_keeps_future_out_of_train(orders):
    ready = orders.dropna(subset=["lead_time_days", "order_date"]).copy()
    train, test, info = chronological_split(ready)
    assert train["order_date"].max() < test["order_date"].min()
    assert info["cutoff"] == str(pd.Timestamp(test["order_date"].min()).date())
    assert len(train) > len(test) > 0
    # Same order date is not split.
    overlap = set(train["order_date"]).intersection(set(test["order_date"]))
    assert overlap == set()


def test_slow_segments_do_not_treat_year_offset_as_slowness(orders):
    quality = validate_orders(orders)
    assert quality["variance_decomposition"]["ship_mode_r2_raw"] < 0.05
    segments = slow_segments(orders)
    assert "materially_slower" in segments.columns
    # With the year offset removed, product-region cells are not a day slower than ship mode.
    assert int(segments["materially_slower"].sum()) == 0
