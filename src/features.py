"""Target creation, leakage checks, and chronological splitting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import (
    BASE_FEATURES,
    FORBIDDEN_FEATURES,
    MATERIAL_SLOW_GAP_DAYS,
    MIN_ROWS_TO_TRAIN,
    MIN_SAMPLE_WARNING,
    TEST_FRACTION,
)


def leakage_audit(feature_columns: list[str]) -> dict[str, Any]:
    found = sorted(set(feature_columns) & FORBIDDEN_FEATURES)
    return {
        "passed": len(found) == 0,
        "forbidden_found": found,
        "feature_columns": list(feature_columns),
        "rule": (
            "Ship date, lead time, order identifiers, customer id, postal code, "
            "cohort year, and financial outcome fields are not model inputs."
        ),
    }


def fit_cohort_medians(df: pd.DataFrame) -> dict[str, float]:
    valid = df.dropna(subset=["lead_time_days", "cohort_year"])
    if valid.empty:
        return {}
    medians = valid.groupby("cohort_year")["lead_time_days"].median()
    return {str(k): float(v) for k, v in medians.items()}


def apply_train_residual(
    df: pd.DataFrame,
    cohort_medians: dict[str, float],
    global_median: float,
) -> pd.DataFrame:
    """Attach a residual using training-only cohort medians. Does not add a feature."""
    out = df.copy()
    mapped = out["cohort_year"].astype("string").map(cohort_medians)
    out["cohort_median_days"] = mapped.astype(float).fillna(global_median)
    out["operational_residual_days"] = out["lead_time_days"] - out["cohort_median_days"]
    return out


def model_ready_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = ["lead_time_days", "order_date", "ship_mode", "region", "product_name"]
    ready = df.dropna(subset=required).copy()
    ready = ready.loc[np.isfinite(ready["lead_time_days"])].copy()
    return ready


def chronological_split(
    df: pd.DataFrame,
    test_fraction: float = TEST_FRACTION,
    date_col: str = "order_date",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Hold out the latest orders by date. A calendar date is never split across sets."""
    if df.empty:
        raise ValueError("Cannot split an empty frame.")
    ordered = df.sort_values([date_col, "row_id"] if "row_id" in df.columns else [date_col])
    pos = int(len(ordered) * (1.0 - test_fraction))
    pos = min(max(pos, 1), len(ordered) - 1)
    cutoff = pd.Timestamp(ordered.iloc[pos][date_col])
    train = df.loc[df[date_col] < cutoff].copy()
    test = df.loc[df[date_col] >= cutoff].copy()
    if train.empty or test.empty:
        unique_dates = np.array(sorted(df[date_col].dropna().unique()))
        if len(unique_dates) < 2:
            raise ValueError("Not enough distinct order dates for a chronological split.")
        cutoff = pd.Timestamp(unique_dates[max(1, int(len(unique_dates) * (1.0 - test_fraction)))])
        train = df.loc[df[date_col] < cutoff].copy()
        test = df.loc[df[date_col] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty partition.")
    info = {
        "strategy": "chronological holdout by order date",
        "date_column": date_col,
        "test_fraction_requested": test_fraction,
        "cutoff": pd.Timestamp(cutoff).date().isoformat(),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_start": pd.Timestamp(train[date_col].min()).date().isoformat(),
        "train_end": pd.Timestamp(train[date_col].max()).date().isoformat(),
        "test_start": pd.Timestamp(test[date_col].min()).date().isoformat(),
        "test_end": pd.Timestamp(test[date_col].max()).date().isoformat(),
        "train_target_median": float(train["lead_time_days"].median()) if "lead_time_days" in train else None,
        "test_target_median": float(test["lead_time_days"].median()) if "lead_time_days" in test else None,
        "grouping_note": (
            "Rows that share an order date stay in the same partition because the "
            "cutoff is a date boundary. Order lines in this extract share one order date."
        ),
    }
    return train, test, info


def date_folds(df: pd.DataFrame, fractions: tuple[float, ...] = (0.55, 0.70, 0.85)) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window folds inside the training set. Validation dates are strictly later."""
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    ordered = df.sort_values("order_date")
    for frac in fractions:
        pos = int(len(ordered) * frac)
        pos = min(max(pos, 1), len(ordered) - 1)
        cutoff = pd.Timestamp(ordered.iloc[pos]["order_date"])
        train = df.loc[df["order_date"] < cutoff]
        valid = df.loc[df["order_date"] >= cutoff]
        if len(train) < MIN_ROWS_TO_TRAIN or len(valid) < 20:
            continue
        folds.append((train, valid))
    return folds


def slow_segments(df: pd.DataFrame, min_n: int = MIN_SAMPLE_WARNING) -> pd.DataFrame:
    """Compare each route cell with its ship-mode residual norm.

    Raw calendar lead time is not used as the slowness measure, because the
    year-offset artifact would be mistaken for a slow route.
    """
    work = df.dropna(subset=["descriptive_residual_days", "ship_mode", "product_name", "region"]).copy()
    if work.empty:
        return pd.DataFrame()
    ship_norm = work.groupby("ship_mode")["descriptive_residual_days"].median().rename("ship_mode_residual_median")
    grouped = (
        work.groupby(["product_name", "region", "ship_mode"], dropna=False)
        .agg(
            sample_size=("descriptive_residual_days", "size"),
            median_lead_time_days=("lead_time_days", "median"),
            median_residual_days=("descriptive_residual_days", "median"),
            gross_profit=("gross_profit", "sum"),
        )
        .reset_index()
    )
    grouped = grouped.merge(ship_norm, on="ship_mode", how="left")
    grouped["residual_gap_days"] = grouped["median_residual_days"] - grouped["ship_mode_residual_median"]
    grouped["materially_slower"] = (grouped["sample_size"] >= min_n) & (
        grouped["residual_gap_days"] > MATERIAL_SLOW_GAP_DAYS
    )
    grouped["sample_warning"] = grouped["sample_size"] < min_n
    grouped["interpretation"] = np.where(
        grouped["materially_slower"],
        "Slower than the ship-mode norm by more than 1 day. Descriptive only, not a factory effect.",
        np.where(
            grouped["sample_warning"],
            "Sample is below the minimum. Do not treat the gap as a stable route problem.",
            "Not materially slower than the ship-mode norm after removing the year-offset artifact.",
        ),
    )
    return grouped.sort_values(["materially_slower", "residual_gap_days"], ascending=[False, False]).reset_index(drop=True)


def feature_frame(df: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    columns = feature_columns or BASE_FEATURES
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    frame = df.loc[:, columns].copy()
    for col in frame.columns:
        if col in {"units", "order_month", "order_dow", "order_quarter"}:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        else:
            frame[col] = frame[col].astype("string").fillna("Unknown")
    return frame
