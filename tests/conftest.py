"""Shared fixtures for calculation and validation tests."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import DEFAULT_ORDERS_PATH
from src.data_loader import load_orders, standardize_orders


def raw_order(**overrides) -> dict:
    row = {
        "Row ID": 1,
        "Order ID": "US-2021-100-CHO-MIL-31000",
        "Order Date": "03-01-2024",
        "Ship Date": "30-06-2026",
        "Ship Mode": "Standard Class",
        "Customer ID": "CUST-100",
        "Country/Region": "United States",
        "City": "Houston",
        "State/Province": "Texas",
        "Postal Code": "77095",
        "Division": "Chocolate",
        "Region": "Interior",
        "Product ID": "CHO-MIL-31000",
        "Product Name": "Wonka Bar - Milk Chocolate",
        "Sales": 10.0,
        "Units": 2,
        "Gross Profit": 6.0,
        "Cost": 4.0,
    }
    row.update(overrides)
    return row


@pytest.fixture(scope="session")
def orders() -> pd.DataFrame:
    return load_orders(DEFAULT_ORDERS_PATH)


def frame_from_rows(rows: list[dict]) -> pd.DataFrame:
    return standardize_orders(pd.DataFrame(rows))
