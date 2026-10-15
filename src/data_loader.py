"""Load the Nassau Candy extract and produce an explicit validation report."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    CENTROID_NOTE,
    CENTROIDS_PATH,
    COLUMN_MAP,
    DATE_FORMAT,
    DATE_FORMAT_LABEL,
    DEFAULT_ORDERS_PATH,
    ELIGIBILITY_PATH,
    FACTORIES,
    FACTORY_REFERENCE_PATH,
    GP_TOLERANCE,
    LONG_LEAD_TIME_DAYS,
    MIN_ROWS_TO_TRAIN,
    MULTI_YEAR_LEAD_TIME_DAYS,
    ORIGIN_COLUMN_CANDIDATES,
    PRODUCT_MAPPING_PATH,
    PRODUCT_MAPPINGS,
    REQUIRED_COLUMNS,
    STATE_CENTROIDS,
)

_HYPHEN_RE = re.compile(r"-\s*")
_SPACE_RE = re.compile(r"\s+")
_ORDER_YEAR_RE = re.compile(r"-(\d{4})-")


class DataSchemaError(Exception):
    """Raised when a required column is missing. Descriptive analysis cannot proceed."""

    def __init__(self, missing: list[str]):
        self.missing = list(missing)
        joined = ", ".join(self.missing)
        super().__init__(f"Required columns are missing: {joined}")


def normalize_product_name(name: object) -> str:
    """Collapse spacing so mapping names join to extract names."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    text = str(name).strip().lower().replace("–", "-").replace("—", "-")
    text = _SPACE_RE.sub(" ", text)
    text = _HYPHEN_RE.sub("- ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def file_sha256(source: str | Path) -> str:
    digest = hashlib.sha256()
    with open(source, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _rewind(source: Any) -> None:
    if hasattr(source, "seek"):
        source.seek(0)


def display_name_map() -> dict[str, str]:
    return {normalize_product_name(row["product_name"]): row["product_name"] for row in PRODUCT_MAPPINGS}


def reference_factories() -> pd.DataFrame:
    return pd.DataFrame(FACTORIES)


def reference_mapping() -> pd.DataFrame:
    frame = pd.DataFrame(PRODUCT_MAPPINGS)
    frame["product_name_key"] = frame["product_name"].map(normalize_product_name)
    frame = frame.rename(columns={"division": "mapping_division", "factory": "current_factory"})
    return frame


def reference_centroids() -> pd.DataFrame:
    frame = pd.DataFrame(STATE_CENTROIDS)
    frame["note"] = CENTROID_NOTE
    return frame


def default_eligibility() -> pd.DataFrame:
    mapping = reference_mapping()
    factories = reference_factories()
    rows = []
    for product in mapping.itertuples(index=False):
        for factory in factories["factory"]:
            is_current = factory == product.current_factory
            rows.append(
                {
                    "product_name": product.product_name,
                    "product_name_key": product.product_name_key,
                    "factory": factory,
                    "status": "confirmed_current" if is_current else "requires_validation",
                    "source": "supplied_mapping" if is_current else "default_unverified",
                    "notes": (
                        "Supplied current assignment. Confirm with the business owner before production use."
                        if is_current
                        else "Alternative is not confirmed as operationally or legally eligible."
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_reference_files() -> None:
    """Write the editable reference CSVs from the supplied specification."""
    DATA_DIR = FACTORY_REFERENCE_PATH.parent
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    reference_factories().to_csv(FACTORY_REFERENCE_PATH, index=False)
    mapping = reference_mapping().rename(
        columns={"mapping_division": "division", "current_factory": "factory"}
    )
    mapping[["division", "product_name", "factory"]].to_csv(PRODUCT_MAPPING_PATH, index=False)
    default_eligibility().drop(columns=["product_name_key"]).to_csv(ELIGIBILITY_PATH, index=False)
    reference_centroids().to_csv(CENTROIDS_PATH, index=False)


def load_factories() -> pd.DataFrame:
    if FACTORY_REFERENCE_PATH.exists():
        frame = pd.read_csv(FACTORY_REFERENCE_PATH)
    else:
        frame = reference_factories()
    frame["factory"] = frame["factory"].astype(str)
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    return frame


def load_mapping() -> pd.DataFrame:
    if PRODUCT_MAPPING_PATH.exists():
        frame = pd.read_csv(PRODUCT_MAPPING_PATH)
        frame = frame.rename(columns={"division": "mapping_division", "factory": "current_factory"})
    else:
        frame = reference_mapping()
    frame["product_name"] = frame["product_name"].astype(str)
    frame["product_name_key"] = frame["product_name"].map(normalize_product_name)
    frame["current_factory"] = frame["current_factory"].astype(str)
    return frame


def load_eligibility() -> pd.DataFrame:
    if ELIGIBILITY_PATH.exists():
        frame = pd.read_csv(ELIGIBILITY_PATH)
    else:
        frame = default_eligibility()
    if "product_name_key" not in frame.columns:
        frame["product_name_key"] = frame["product_name"].map(normalize_product_name)
    frame["status"] = frame["status"].astype(str)
    frame["factory"] = frame["factory"].astype(str)
    return frame


def load_centroids() -> pd.DataFrame:
    if CENTROIDS_PATH.exists():
        frame = pd.read_csv(CENTROIDS_PATH)
    else:
        frame = reference_centroids()
    frame["state"] = frame["state"].astype(str)
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    return frame


def load_orders(source: str | Path | Any | None = None) -> pd.DataFrame:
    """Load and standardize an orders extract. Does not drop rows."""
    if source is None:
        source = DEFAULT_ORDERS_PATH
    _rewind(source)
    header = pd.read_csv(source, nrows=0)
    _rewind(source)
    dtype = {"Postal Code": "string"} if "Postal Code" in header.columns else None
    raw = pd.read_csv(source, dtype=dtype)
    return standardize_orders(raw)


def standardize_orders(raw: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise DataSchemaError(missing)

    source_columns = list(raw.columns)
    origin_source = next((col for col in ORIGIN_COLUMN_CANDIDATES if col in raw.columns), None)
    frame = raw.rename(columns=COLUMN_MAP).copy()
    frame.attrs["source_columns"] = source_columns
    frame.attrs["origin_source_column"] = origin_source

    frame["row_id"] = pd.to_numeric(frame["row_id"], errors="coerce")
    frame["customer_id"] = frame["customer_id"].astype("string")
    frame["postal_code"] = (
        frame["postal_code"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    )
    for column in ["sales", "units", "gross_profit", "cost"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["order_date"] = pd.to_datetime(frame["order_date_raw"], format=DATE_FORMAT, errors="coerce")
    frame["ship_date"] = pd.to_datetime(frame["ship_date_raw"], format=DATE_FORMAT, errors="coerce")
    frame["order_date_mdy"] = pd.to_datetime(frame["order_date_raw"], format="%m-%d-%Y", errors="coerce")
    frame["ship_date_mdy"] = pd.to_datetime(frame["ship_date_raw"], format="%m-%d-%Y", errors="coerce")
    frame["lead_time_days"] = (frame["ship_date"] - frame["order_date"]).dt.days.astype("Float64")

    frame["product_name_key"] = frame["product_name_source"].map(normalize_product_name)
    names = display_name_map()
    frame["product_name"] = frame["product_name_key"].map(names).fillna(frame["product_name_source"])
    frame["cohort_year"] = frame["order_id"].astype("string").str.extract(_ORDER_YEAR_RE, expand=False)
    frame["order_year"] = frame["order_date"].dt.year.astype("Int64").astype("string")
    frame["order_month"] = frame["order_date"].dt.month.astype("Float64")
    frame["order_dow"] = frame["order_date"].dt.dayofweek.astype("Float64")
    frame["order_quarter"] = frame["order_date"].dt.quarter.astype("Float64")

    sales_cost = frame["sales"] - frame["cost"]
    frame["gp_gap"] = (frame["gross_profit"] - sales_cost).abs()
    frame["margin"] = np.where(frame["sales"] > 0, frame["gross_profit"] / frame["sales"], np.nan)
    frame["negative_lead_time"] = frame["lead_time_days"] < 0
    frame["zero_lead_time"] = frame["lead_time_days"] == 0
    frame["long_lead_time"] = frame["lead_time_days"] > LONG_LEAD_TIME_DAYS
    frame["multi_year_lead_time"] = frame["lead_time_days"] > MULTI_YEAR_LEAD_TIME_DAYS
    frame["invalid_order_date"] = frame["order_date"].isna()
    frame["invalid_ship_date"] = frame["ship_date"].isna()

    if origin_source and origin_source in COLUMN_MAP:
        frame["origin_factory"] = frame[COLUMN_MAP[origin_source]]
    elif origin_source and origin_source in raw.columns:
        frame["origin_factory"] = raw[origin_source].astype("string")
    else:
        frame["origin_factory"] = pd.NA

    valid = frame.dropna(subset=["lead_time_days", "cohort_year"])
    if valid.empty:
        frame["descriptive_cohort_median"] = np.nan
        frame["descriptive_residual_days"] = np.nan
    else:
        medians = valid.groupby("cohort_year")["lead_time_days"].transform("median")
        frame["descriptive_cohort_median"] = medians
        frame["descriptive_residual_days"] = frame["lead_time_days"] - frame["descriptive_cohort_median"]
    return frame


def _grouped_r2(y: pd.Series, group: pd.Series) -> float | None:
    mask = y.notna() & group.notna()
    y = y.loc[mask].astype(float)
    group = group.loc[mask]
    if len(y) < 2 or y.nunique() <= 1:
        return None
    pred = y.groupby(group).transform("mean")
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot == 0:
        return None
    ss_res = float(((y - pred) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def _iqr_outlier_count(series: pd.Series) -> int:
    values = series.dropna().astype(float)
    if values.empty:
        return 0
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    mask = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
    return int(mask.sum())


def _warning(code: str, severity: str, message: str, count: int | None = None) -> dict[str, Any]:
    return {"code": code, "severity": severity, "message": message, "count": count}


def validate_orders(df: pd.DataFrame) -> dict[str, Any]:
    """Profile the extract. Extreme lead times are flagged, never silently deleted."""
    n_rows = int(len(df))
    source_columns = list(df.attrs.get("source_columns", []))
    valid_lt = df["lead_time_days"].dropna()
    invalid_dates = int((df["invalid_order_date"] | df["invalid_ship_date"]).sum())
    invalid_rate = invalid_dates / n_rows if n_rows else 1.0
    gp_bad = int((df["gp_gap"] > GP_TOLERANCE).sum())
    mapping = load_mapping()
    merged = df.merge(
        mapping[["product_name_key", "mapping_division", "current_factory"]],
        on="product_name_key",
        how="left",
    )
    unmapped = sorted(merged.loc[merged["current_factory"].isna(), "product_name_source"].dropna().unique().tolist())
    division_mismatch = merged.loc[
        merged["current_factory"].notna() & (merged["division"] != merged["mapping_division"]),
        ["product_name", "division", "mapping_division"],
    ].drop_duplicates()
    name_fixes = int((df["product_name_source"] != df["product_name"]).sum())
    year_mismatch = int(
        (
            df["cohort_year"].notna()
            & df["order_date"].notna()
            & (df["cohort_year"].astype("string") != df["order_year"].astype("string"))
        ).sum()
    )
    business_key = [
        "order_id",
        "order_date_raw",
        "ship_date_raw",
        "ship_mode",
        "city",
        "product_id",
        "sales",
        "units",
    ]
    duplicate_business = int(df.duplicated(business_key).sum())
    us = df["country"].astype("string") == "United States"
    canada = df["country"].astype("string") == "Canada"
    postal_len = df["postal_code"].astype("string").str.len()
    us_short_postal = int((us & postal_len.eq(4)).sum())
    canada_short_postal = int((canada & postal_len.le(3)).sum())
    origin_present = bool(df["origin_factory"].notna().any())
    origin_usable = bool(
        origin_present
        and df["origin_factory"].notna().mean() >= 0.70
        and df["origin_factory"].nunique(dropna=True) >= 2
    )
    multi_year = int(df["multi_year_lead_time"].fillna(False).sum())
    long_lead = int(df["long_lead_time"].fillna(False).sum())
    negative = int(df["negative_lead_time"].fillna(False).sum())
    zero = int(df["zero_lead_time"].fillna(False).sum())
    iqr_outliers = _iqr_outlier_count(valid_lt)

    warnings: list[dict[str, Any]] = []
    if multi_year:
        warnings.append(
            _warning(
                "DATE_RANGE_ANOMALY",
                "critical",
                (
                    f"Ship dates parse as {DATE_FORMAT_LABEL}, and {multi_year:,} of {n_rows:,} "
                    f"lead times exceed {MULTI_YEAR_LEAD_TIME_DAYS} days "
                    f"(range {float(valid_lt.min()):.0f}–{float(valid_lt.max()):.0f}). "
                    "This may be a source-system offset, synthetic data, or a real delay. "
                    "It is flagged, not deleted, and it is not treated as a credible delivery ETA."
                ),
                multi_year,
            )
        )
    if year_mismatch:
        warnings.append(
            _warning(
                "ORDER_ID_YEAR_MISMATCH",
                "critical",
                (
                    f"{year_mismatch:,} rows have an Order ID year that does not match the order-date year. "
                    "The Order ID year lines up with the multi-year lead-time bands. "
                    "It is an identifier, not a model feature."
                ),
                year_mismatch,
            )
        )
    if not origin_usable:
        warnings.append(
            _warning(
                "FACTORY_ORIGIN_ABSENT",
                "critical",
                (
                    "The order extract has no usable origin-factory field. "
                    "Current factory is joined from the supplied product mapping and is an assumption, "
                    "not an observed shipment fact. The model must not invent a factory effect by swapping labels."
                ),
                n_rows,
            )
        )
    warnings.append(
        _warning(
            "CUSTOMER_COORDINATES_ABSENT",
            "warning",
            (
                "Customer latitude and longitude are not in the extract. "
                "Great-circle distance runs only when the user supplies coordinates or explicitly "
                "accepts approximate state centroids. Straight-line distance is not road distance."
            ),
        )
    )
    warnings.append(
        _warning(
            "FREIGHT_COST_ABSENT",
            "warning",
            (
                "Freight charges, carrier rates, and factory-specific operating costs are not in the extract. "
                "Profit impact of a reassignment is not quantified unless the user enters a labeled sensitivity assumption."
            ),
        )
    )
    if long_lead:
        warnings.append(
            _warning(
                "LONG_LEAD_TIME",
                "warning",
                (
                    f"{long_lead:,} lead times exceed {LONG_LEAD_TIME_DAYS} days, the business review threshold "
                    "for a ship-date target. Rows are retained."
                ),
                long_lead,
            )
        )
    if name_fixes:
        warnings.append(
            _warning(
                "PRODUCT_NAME_SPACING",
                "warning",
                (
                    f"{name_fixes:,} rows use a product name whose spacing differs from the supplied mapping "
                    "(Wonka Bar -Scrumdiddlyumptious). Names are joined on a normalized key and the source spelling is preserved."
                ),
                name_fixes,
            )
        )
    if not division_mismatch.empty:
        pairs = "; ".join(
            f"{row.product_name}: extract {row.division}, mapping {row.mapping_division}"
            for row in division_mismatch.itertuples(index=False)
        )
        warnings.append(
            _warning(
                "DIVISION_MAPPING_MISMATCH",
                "warning",
                f"Division does not match the supplied mapping for: {pairs}. Observed division is kept.",
                int(len(division_mismatch)),
            )
        )
    if unmapped:
        warnings.append(
            _warning(
                "UNMAPPED_PRODUCT",
                "warning",
                "Products with no supplied factory mapping: " + ", ".join(unmapped),
                len(unmapped),
            )
        )
    if us_short_postal:
        warnings.append(
            _warning(
                "US_POSTAL_POSSIBLE_MISSING_ZERO",
                "warning",
                (
                    f"{us_short_postal:,} United States postal codes have 4 characters. "
                    "Northeast codes may have lost a leading zero in the source extract. "
                    "Postal code is stored as text and is not a model feature."
                ),
                us_short_postal,
            )
        )
    if canada_short_postal:
        warnings.append(
            _warning(
                "CANADA_POSTAL_TRUNCATED",
                "warning",
                f"{canada_short_postal:,} Canadian postal values are 3 characters or shorter and look truncated.",
                canada_short_postal,
            )
        )
    if duplicate_business:
        warnings.append(
            _warning(
                "DUPLICATE_BUSINESS_ROWS",
                "warning",
                (
                    f"{duplicate_business:,} rows repeat the same order, dates, ship mode, city, product, sales, and units. "
                    "They are retained so sample sizes are not silently reduced. Row ID itself is unique."
                ),
                duplicate_business,
            )
        )
    if negative:
        warnings.append(_warning("NEGATIVE_LEAD_TIME", "warning", f"{negative:,} negative lead times.", negative))
    if zero:
        warnings.append(_warning("ZERO_LEAD_TIME", "info", f"{zero:,} zero-day lead times.", zero))
    if invalid_dates:
        warnings.append(
            _warning(
                "INVALID_DATES",
                "warning",
                f"{invalid_dates:,} rows have an order date or ship date that does not parse as {DATE_FORMAT_LABEL}.",
                invalid_dates,
            )
        )
    if gp_bad:
        warnings.append(
            _warning(
                "GP_DISCREPANCY",
                "warning",
                (
                    f"{gp_bad:,} rows have Gross Profit different from Sales minus Cost by more than {GP_TOLERANCE}. "
                    "Source values are preserved."
                ),
                gp_bad,
            )
        )
    else:
        warnings.append(
            _warning(
                "FINANCIAL_IDENTITY_HOLDS",
                "info",
                "Gross Profit matches Sales minus Cost within one cent on every row. Source values were not rewritten.",
                0,
            )
        )
    warnings.append(
        _warning(
            "IQR_OUTLIERS",
            "info",
            (
                f"{iqr_outliers:,} lead times fall outside the 1.5 IQR fence. "
                "A multi-year shift that affects most rows is not an IQR outlier, so the business rule is reported separately."
            ),
            iqr_outliers,
        )
    )

    residual = df["descriptive_residual_days"]
    variance = {
        "order_id_year_r2": _grouped_r2(df["lead_time_days"], df["cohort_year"]),
        "ship_mode_r2_raw": _grouped_r2(df["lead_time_days"], df["ship_mode"]),
        "region_r2_raw": _grouped_r2(df["lead_time_days"], df["region"]),
        "product_r2_raw": _grouped_r2(df["lead_time_days"], df["product_name"]),
        "ship_mode_r2_residual": _grouped_r2(residual, df["ship_mode"]),
        "region_r2_residual": _grouped_r2(residual, df["region"]),
        "product_r2_residual": _grouped_r2(residual, df["product_name"]),
        "note": (
            "Order ID year R² is a diagnostic of the date artifact. "
            "That identifier is not used as a model feature."
        ),
    }
    peaks = []
    if not valid_lt.empty:
        for label, low, high in (("about 908 days", 900, 950), ("about 1273 days", 1260, 1290), ("about 1639 days", 1630, 1660)):
            peaks.append({"label": label, "rows": int(((valid_lt >= low) & (valid_lt < high)).sum())})

    blockers = []
    if n_rows == 0:
        blockers.append("The extract has no rows.")
    if valid_lt.shape[0] < MIN_ROWS_TO_TRAIN:
        blockers.append(f"Fewer than {MIN_ROWS_TO_TRAIN} rows have a usable lead time.")
    if invalid_rate > 0.50:
        blockers.append("More than half of the rows have an unparseable order or ship date.")
    training_allowed = not blockers

    quality = {
        "n_rows": n_rows,
        "n_columns_source": len(source_columns) if source_columns else int(df.shape[1]),
        "source_columns": source_columns,
        "date_format": DATE_FORMAT_LABEL,
        "date_profile": {
            "order_invalid": int(df["invalid_order_date"].sum()),
            "ship_invalid": int(df["invalid_ship_date"].sum()),
            "order_min": _date_text(df["order_date"].min()),
            "order_max": _date_text(df["order_date"].max()),
            "ship_min": _date_text(df["ship_date"].min()),
            "ship_max": _date_text(df["ship_date"].max()),
            "mdy_order_invalid": int(df["order_date_mdy"].isna().sum()),
            "mdy_ship_invalid": int(df["ship_date_mdy"].isna().sum()),
            "format_decision": (
                "Day-month-year is used because it parses every supplied date, including day values above 12. "
                "Month-first parsing leaves those dates invalid."
            ),
        },
        "lead_time_profile": {
            "n_valid": int(valid_lt.shape[0]),
            "min": float(valid_lt.min()) if not valid_lt.empty else None,
            "p25": float(valid_lt.quantile(0.25)) if not valid_lt.empty else None,
            "median": float(valid_lt.median()) if not valid_lt.empty else None,
            "p75": float(valid_lt.quantile(0.75)) if not valid_lt.empty else None,
            "max": float(valid_lt.max()) if not valid_lt.empty else None,
            "negative": negative,
            "zero": zero,
            "long": long_lead,
            "multi_year": multi_year,
            "iqr_outliers": iqr_outliers,
            "peaks": peaks,
            "unique_values": int(valid_lt.nunique()) if not valid_lt.empty else 0,
        },
        "financial_profile": {
            "gp_discrepancy_count": gp_bad,
            "negative_sales": int((df["sales"] < 0).sum()),
            "negative_cost": int((df["cost"] < 0).sum()),
            "nonpositive_units": int((df["units"] <= 0).sum()),
            "sales_sum": float(df["sales"].sum(skipna=True)),
            "gross_profit_sum": float(df["gross_profit"].sum(skipna=True)),
            "units_sum": float(df["units"].sum(skipna=True)),
            "margin_median": float(df["margin"].median()) if df["margin"].notna().any() else None,
        },
        "duplicate_profile": {
            "duplicate_row_id": int(df["row_id"].duplicated().sum()),
            "duplicate_business_rows": duplicate_business,
            "unique_order_id": int(df["order_id"].nunique(dropna=True)),
            "unique_customer_id": int(df["customer_id"].nunique(dropna=True)),
        },
        "product_profile": {
            "products": int(df["product_name"].nunique(dropna=True)),
            "unmapped_products": unmapped,
            "name_spacing_rows": name_fixes,
            "division_mismatches": division_mismatch.to_dict(orient="records"),
        },
        "postal_profile": {
            "us_four_character": us_short_postal,
            "canada_three_character_or_shorter": canada_short_postal,
        },
        "origin_profile": {
            "source_column": df.attrs.get("origin_source_column"),
            "present": origin_present,
            "usable_for_training": origin_usable,
        },
        "variance_decomposition": variance,
        "order_id_year_mismatch_rows": year_mismatch,
        "warnings": warnings,
        "warning_count": sum(1 for item in warnings if item["severity"] in {"critical", "warning"}),
        "critical_count": sum(1 for item in warnings if item["severity"] == "critical"),
        "training_allowed": training_allowed,
        "training_blockers": blockers,
        "target_credibility": "low" if multi_year > n_rows * 0.5 else "review",
        "target_quality_penalty": 0.85 if multi_year > n_rows * 0.5 else (0.60 if long_lead > n_rows * 0.5 else 0.15),
        "rows_deleted": 0,
    }
    return quality


def _date_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def load_geocodes(source: Any) -> pd.DataFrame:
    """Load a user-supplied coordinate file. Empty or invalid files return an empty frame."""
    if source is None:
        return pd.DataFrame(columns=["city", "state", "latitude", "longitude", "city_key", "state_key"])
    _rewind(source)
    raw = pd.read_csv(source)
    renamed = {col: col.strip().lower() for col in raw.columns}
    raw = raw.rename(columns=renamed)
    column_aliases = {
        "city": "city",
        "state/province": "state",
        "state": "state",
        "province": "state",
        "latitude": "latitude",
        "lat": "latitude",
        "longitude": "longitude",
        "lon": "longitude",
        "lng": "longitude",
    }
    mapped = {}
    for col in raw.columns:
        key = column_aliases.get(str(col).strip().lower())
        if key and key not in mapped:
            mapped[key] = col
    if not {"city", "state", "latitude", "longitude"} <= set(mapped):
        raise ValueError("Geocode file must include City, State/Province, Latitude, and Longitude.")
    frame = pd.DataFrame(
        {
            "city": raw[mapped["city"]].astype("string"),
            "state": raw[mapped["state"]].astype("string"),
            "latitude": pd.to_numeric(raw[mapped["latitude"]], errors="coerce"),
            "longitude": pd.to_numeric(raw[mapped["longitude"]], errors="coerce"),
        }
    )
    frame = frame.dropna(subset=["latitude", "longitude", "city", "state"])
    frame["city_key"] = frame["city"].str.strip().str.lower()
    frame["state_key"] = frame["state"].str.strip().str.lower()
    return frame
