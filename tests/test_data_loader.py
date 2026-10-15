"""Validation, date parsing, and reference-data tests."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import DATE_FORMAT, DEFAULT_ORDERS_PATH, FACTORIES, PRODUCT_MAPPINGS
from src.data_loader import (
    DataSchemaError,
    load_eligibility,
    load_factories,
    load_mapping,
    load_orders,
    normalize_product_name,
    validate_orders,
)
from tests.conftest import frame_from_rows, raw_order


def test_product_name_normalization_joins_spacing_variant():
    assert normalize_product_name("Wonka Bar -Scrumdiddlyumptious") == normalize_product_name(
        "Wonka Bar - Scrumdiddlyumptious"
    )


def test_day_first_dates_and_lead_time():
    frame = frame_from_rows(
        [raw_order(**{"Order Date": "31-05-2024", "Ship Date": "30-06-2026", "Row ID": 7})]
    )
    assert frame.loc[0, "order_date"] == pd.Timestamp("2024-05-31")
    assert frame.loc[0, "ship_date"] == pd.Timestamp("2026-06-30")
    expected = (pd.Timestamp("2026-06-30") - pd.Timestamp("2024-05-31")).days
    assert int(frame.loc[0, "lead_time_days"]) == expected
    assert bool(frame.loc[0, "long_lead_time"])
    assert bool(frame.loc[0, "multi_year_lead_time"])


def test_negative_lead_time_is_flagged_not_dropped():
    frame = frame_from_rows(
        [raw_order(**{"Order Date": "10-02-2025", "Ship Date": "01-02-2025"})]
    )
    assert len(frame) == 1
    assert int(frame.loc[0, "lead_time_days"]) < 0
    assert bool(frame.loc[0, "negative_lead_time"])


def test_postal_code_stays_text_with_leading_zero(tmp_path):
    path = tmp_path / "orders.csv"
    row = raw_order(**{"Postal Code": "01234"})
    pd.DataFrame([row]).to_csv(path, index=False)
    frame = load_orders(path)
    assert frame.loc[0, "postal_code"] == "01234"
    assert str(frame["postal_code"].dtype) == "string" or frame["postal_code"].dtype == object


def test_missing_columns_raise():
    with pytest.raises(DataSchemaError) as exc:
        frame_from_rows([]) if False else None
        from src.data_loader import standardize_orders

        standardize_orders(pd.DataFrame({"Row ID": [1], "Order ID": ["x"]}))
    assert "Ship Date" in exc.value.missing


def test_gross_profit_discrepancy_is_reported_not_rewritten():
    frame = frame_from_rows([raw_order(**{"Sales": 10.0, "Cost": 4.0, "Gross Profit": 1.0})])
    assert frame.loc[0, "gross_profit"] == 1.0
    quality = validate_orders(frame)
    assert quality["financial_profile"]["gp_discrepancy_count"] == 1
    assert any(item["code"] == "GP_DISCREPANCY" for item in quality["warnings"])


def test_real_extract_profile(orders):
    quality = validate_orders(orders)
    assert quality["n_rows"] == 10194
    assert quality["n_columns_source"] == 18
    assert quality["date_profile"]["order_invalid"] == 0
    assert quality["date_profile"]["ship_invalid"] == 0
    assert quality["date_profile"]["mdy_order_invalid"] > 0
    assert quality["lead_time_profile"]["min"] >= 900
    assert quality["lead_time_profile"]["multi_year"] == 10194
    assert quality["financial_profile"]["gp_discrepancy_count"] == 0
    assert quality["duplicate_profile"]["duplicate_row_id"] == 0
    assert quality["training_allowed"] is True
    assert quality["rows_deleted"] == 0
    assert quality["product_profile"]["unmapped_products"] == []
    codes = {item["code"] for item in quality["warnings"]}
    assert "DATE_RANGE_ANOMALY" in codes
    assert "FACTORY_ORIGIN_ABSENT" in codes
    assert "FREIGHT_COST_ABSENT" in codes
    assert "ORDER_ID_YEAR_MISMATCH" in codes
    assert quality["variance_decomposition"]["order_id_year_r2"] > 0.99
    assert orders["customer_id"].dtype == "string" or str(orders["customer_id"].dtype).startswith("string")


def test_mapping_covers_supplied_products():
    mapping = load_mapping()
    assert set(mapping["current_factory"]) <= {row["factory"] for row in FACTORIES}
    assert len(mapping) == len(PRODUCT_MAPPINGS)
    eligibility = load_eligibility()
    current = eligibility.loc[eligibility["status"] == "confirmed_current"]
    assert len(current) == len(PRODUCT_MAPPINGS)
    alternatives = eligibility.loc[eligibility["status"] == "requires_validation"]
    assert len(alternatives) == len(PRODUCT_MAPPINGS) * (len(FACTORIES) - 1)


def test_default_orders_file_exists():
    assert DEFAULT_ORDERS_PATH.exists()
    assert DATE_FORMAT == "%d-%m-%Y"
