"""Descriptive route clustering. Clusters are labels, not causes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .config import MIN_SAMPLE_WARNING, SEED


def cluster_routes(df: pd.DataFrame, min_n: int = MIN_SAMPLE_WARNING, k: int = 3) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Cluster product-region-ship-mode cells with enough history.

    In this extract the clusters mostly rediscover ship mode. That is reported
    in operational language rather than left as an unlabeled number.
    """
    work = df.dropna(subset=["descriptive_residual_days", "lead_time_days"]).copy()
    grouped = (
        work.groupby(["product_name", "region", "ship_mode"], dropna=False)
        .agg(
            sample_size=("lead_time_days", "size"),
            median_lead_time_days=("lead_time_days", "median"),
            median_residual_days=("descriptive_residual_days", "median"),
            std_residual_days=("descriptive_residual_days", "std"),
            gross_profit=("gross_profit", "sum"),
            sales=("sales", "sum"),
        )
        .reset_index()
    )
    grouped["std_residual_days"] = grouped["std_residual_days"].fillna(0.0)
    eligible = grouped.loc[grouped["sample_size"] >= min_n].copy()
    summary: dict[str, Any] = {
        "min_sample": min_n,
        "eligible_routes": int(len(eligible)),
        "excluded_routes": int((grouped["sample_size"] < min_n).sum()),
        "status": "ok",
    }
    if len(eligible) < k + 2:
        summary["status"] = "insufficient_groups"
        summary["note"] = "Not enough route cells above the sample threshold to cluster."
        grouped["cluster_id"] = pd.NA
        grouped["cluster_label"] = "Not clustered"
        return grouped, summary

    features = pd.DataFrame(
        {
            "median_residual_days": eligible["median_residual_days"],
            "std_residual_days": eligible["std_residual_days"],
            "log_n": np.log1p(eligible["sample_size"]),
        }
    )
    scaled = StandardScaler().fit_transform(features)
    labels = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(scaled)
    eligible["cluster_id"] = labels
    names = _name_clusters(eligible)
    eligible["cluster_label"] = eligible["cluster_id"].map(names)
    grouped = grouped.merge(
        eligible[["product_name", "region", "ship_mode", "cluster_id", "cluster_label"]],
        on=["product_name", "region", "ship_mode"],
        how="left",
    )
    grouped["cluster_label"] = grouped["cluster_label"].fillna("Below sample threshold")
    ship_purity = _ship_mode_purity(eligible)
    summary.update(
        {
            "cluster_names": names,
            "ship_mode_purity": ship_purity,
            "note": (
                "Clusters describe historical route cells. They are not causal factory groups. "
                f"Ship mode accounts for about {ship_purity:.0%} of the cluster assignment, "
                "which means the groups largely rediscover service mode rather than a region or product problem."
            ),
        }
    )
    return grouped, summary


def _name_clusters(frame: pd.DataFrame) -> dict[int, str]:
    names: dict[int, str] = {}
    used: dict[str, int] = {}
    volume_cut = float(frame["sample_size"].median())
    for cluster_id, sub in frame.groupby("cluster_id"):
        residual = float(sub["median_residual_days"].mean())
        if residual <= -2:
            label = "Faster handling"
        elif residual >= 0.5:
            label = "Slower handling"
        else:
            label = "Typical handling"
        if float(sub["sample_size"].mean()) >= volume_cut:
            label = f"Higher-volume, {label[0].lower()}{label[1:]}"
        else:
            label = f"Lower-volume, {label[0].lower()}{label[1:]}"
        used[label] = used.get(label, 0) + 1
        if used[label] > 1:
            label = f"{label} ({used[label]})"
        names[int(cluster_id)] = label
    return names


def _ship_mode_purity(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    pure = 0
    for _, sub in frame.groupby("cluster_id"):
        pure += int(sub["ship_mode"].value_counts().iloc[0])
    return pure / len(frame)
