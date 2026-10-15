"""Operational model card, validation export, and plain-language findings.

This is audit documentation for the decision-support app. It is not a research paper.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .config import ADVISORY_TEXT, DATE_FORMAT_LABEL, MODEL_VERSION


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if pd.isna(value):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def quality_json(quality: dict[str, Any]) -> str:
    return json.dumps(_clean(quality), indent=2)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _metric_table(candidates: dict[str, Any]) -> str:
    lines = ["| Model | CV MAE | Test MAE | Test RMSE | Test R² | Test rows |", "|---|---:|---:|---:|---:|---:|"]
    for name, payload in candidates.items():
        cv = payload.get("cv", {})
        test = payload.get("test", {})
        lines.append(
            f"| {name} | {_fmt(cv.get('mae'))} | {_fmt(test.get('mae'))} | {_fmt(test.get('rmse'))} | "
            f"{_fmt(test.get('r2'), 3)} | {test.get('n', 'n/a')} |"
        )
    return "\n".join(lines)


def model_card_markdown(meta: dict[str, Any], quality: dict[str, Any] | None = None) -> str:
    raw = meta.get("targets", {}).get("lead_time_days", {})
    residual = meta.get("targets", {}).get("operational_residual_days", {})
    split = meta.get("split", {})
    treatment = meta.get("outlier_treatment", {})
    audit = meta.get("leakage_audit", {})
    lines = [
        "# Model card",
        "",
        f"Version {meta.get('model_version', MODEL_VERSION)}. Trained {meta.get('trained_at', 'unknown')}.",
        "",
        ADVISORY_TEXT,
        "",
        "## Target",
        "",
        "Primary target: `LeadTimeDays = Ship Date − Order Date`, in calendar days.",
        f"Dates are parsed as {DATE_FORMAT_LABEL}.",
        "Ship date, lead time, order identifiers, customer id, postal code, and the order-id year are not inputs.",
        "",
        "A second target, the within-cohort operational residual, subtracts the training-set median lead time of the Order ID year. "
        "That identifier is used only to decompose the anomalous target. It is not a feature. "
        "The residual model is a sensitivity analysis, not a replacement ETA.",
        "",
        "## Split",
        "",
        f"- Strategy: {split.get('strategy', 'chronological holdout')}",
        f"- Cutoff: {split.get('cutoff', 'n/a')} (test starts on this date)",
        f"- Train rows: {split.get('train_rows', 'n/a')} ({split.get('train_start', '')} to {split.get('train_end', '')})",
        f"- Test rows: {split.get('test_rows', 'n/a')} ({split.get('test_start', '')} to {split.get('test_end', '')})",
        f"- Train median lead time: {_fmt(split.get('train_target_median'), 1)} days",
        f"- Test median lead time: {_fmt(split.get('test_target_median'), 1)} days",
        "",
        split.get("grouping_note", ""),
        "",
        "Model choice uses chronological cross-validation MAE on the training period. "
        "The holdout is reported once and is not used to pick the model.",
        "",
        "## Raw lead-time results",
        "",
        f"Selected model: `{raw.get('selected_model', 'n/a')}`.",
        "",
        raw.get("selection_reason", ""),
        "",
        _metric_table(raw.get("candidates", {})),
        "",
        "## Within-cohort residual results",
        "",
        f"Selected model: `{residual.get('selected_model', 'n/a')}`.",
        "",
        residual.get("selection_reason", ""),
        "",
        _metric_table(residual.get("candidates", {})),
        "",
        "## Leakage check",
        "",
        f"Passed: {audit.get('passed')}. Forbidden features found: {audit.get('forbidden_found') or 'none'}.",
        f"Features: {', '.join(audit.get('feature_columns') or meta.get('feature_columns') or [])}.",
        f"Factory effect learned from history: {meta.get('factory_effect_learned')}.",
        "",
        "## Outlier treatment",
        "",
        treatment.get("rule", ""),
        f"Flagged rows: {treatment.get('rows_flagged', 'n/a')}. "
        f"Rows retained if excluded: {treatment.get('rows_retained_if_excluded', 'n/a')}. "
        f"Rows deleted: {treatment.get('rows_deleted', 0)}.",
        treatment.get("reason", ""),
        "",
        "## Limitations",
        "",
        "- The raw target is dominated by a multi-year ship-date gap. Metrics describe that gap. They are not a carrier ETA.",
        "- Factory origin is absent unless a future extract supplies it. Recommendations do not invent a factory effect.",
        "- Customer coordinates, freight, capacity, and factory cost are absent. Profit impact stays unquantified unless a user labels a sensitivity assumption.",
        "- A lower scenario lead time is a reason to investigate, not proof that a transfer will improve delivery or profit.",
        "",
    ]
    if quality:
        lines.extend(
            [
                "## Data-quality snapshot",
                "",
                f"Rows: {quality.get('n_rows')}. Source columns: {quality.get('n_columns_source')}.",
                f"Training allowed: {quality.get('training_allowed')}. Target credibility: {quality.get('target_credibility')}.",
                f"Critical findings: {quality.get('critical_count')}. Warning-level findings: {quality.get('warning_count')}.",
                "",
            ]
        )
    return "\n".join(lines)


def validation_markdown(quality: dict[str, Any]) -> str:
    lead = quality.get("lead_time_profile", {})
    dates = quality.get("date_profile", {})
    lines = [
        "# Data validation report",
        "",
        ADVISORY_TEXT,
        "",
        f"Rows: {quality.get('n_rows')}. Source columns: {quality.get('n_columns_source')}.",
        f"Date format: {quality.get('date_format')}.",
        f"Order dates: {dates.get('order_min')} to {dates.get('order_max')}. Invalid order dates: {dates.get('order_invalid')}.",
        f"Ship dates: {dates.get('ship_min')} to {dates.get('ship_max')}. Invalid ship dates: {dates.get('ship_invalid')}.",
        dates.get("format_decision", ""),
        "",
        f"Lead time (days): min {lead.get('min')}, median {lead.get('median')}, max {lead.get('max')}.",
        f"Negative: {lead.get('negative')}. Zero: {lead.get('zero')}. Longer than 60 days: {lead.get('long')}. "
        f"Longer than 365 days: {lead.get('multi_year')}. IQR outliers: {lead.get('iqr_outliers')}.",
        "No rows were deleted.",
        "",
        "## Findings",
        "",
    ]
    for item in quality.get("warnings", []):
        count = "" if item.get("count") is None else f" (count {item['count']})"
        lines.append(f"- **{item['severity']} · {item['code']}**{count}: {item['message']}")
    lines.extend(["", "## Variance diagnostic", ""])
    for key, value in quality.get("variance_decomposition", {}).items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def executive_findings(quality: dict[str, Any], bundle: dict[str, Any] | None) -> list[dict[str, str]]:
    lead = quality.get("lead_time_profile", {})
    variance = quality.get("variance_decomposition", {})
    findings = [
        {
            "severity": "critical",
            "title": "Ship dates sit years after order dates",
            "body": (
                f"All parsed lead times run from {lead.get('min'):.0f} to {lead.get('max'):.0f} days "
                f"(median {lead.get('median'):.0f}). Month-first parsing fails on dates such as 31-05-2024, "
                f"so the file is read as {DATE_FORMAT_LABEL}. The values are retained and flagged. "
                "They are not treated as a credible ship-time ETA."
            ),
        },
        {
            "severity": "critical",
            "title": "The year gap tracks an identifier, not a factory",
            "body": (
                "The year embedded in Order ID explains essentially all of the raw lead-time variance "
                f"(R² {_fmt(variance.get('order_id_year_r2'), 3)}). Ship mode, region, and product do not. "
                "That year is not used as a model feature. After it is removed for a sensitivity view, "
                f"ship mode explains the remaining few days (residual R² {_fmt(variance.get('ship_mode_r2_residual'), 3)})."
            ),
        },
        {
            "severity": "critical",
            "title": "Factory origin was not observed",
            "body": (
                "Current factory is joined from the supplied product mapping. It is an assumption, not a shipment fact. "
                "The model does not change its prediction when the factory label changes, because it never saw a factory."
            ),
        },
    ]
    if bundle:
        raw = bundle.get("targets", {}).get("lead_time_days", {})
        selected = raw.get("selected_model", "n/a")
        test = raw.get("candidates", {}).get(selected, {}).get("test", {})
        baseline = raw.get("candidates", {}).get("group_median_baseline", {}).get("test", {})
        findings.append(
            {
                "severity": "warning",
                "title": "Held-out error is large on the raw target",
                "body": (
                    f"The selected raw-target model is {selected}. Test MAE is {_fmt(test.get('mae'), 1)} days "
                    f"versus {_fmt(baseline.get('mae'), 1)} days for the ship-mode median baseline. "
                    "A chronological holdout is used so later orders are not seen in training. "
                    "These errors describe the date anomaly. They are not a promise of delivery performance."
                ),
            }
        )
    else:
        findings.append(
            {
                "severity": "warning",
                "title": "Models are not loaded",
                "body": "Descriptive analysis is available. Train or load a model before relying on predicted lead time.",
            }
        )
    findings.append(
        {
            "severity": "info",
            "title": "Profit impact is not quantified",
            "body": (
                "Sales, cost, and gross profit reconcile, so historical margin can be shown. "
                "Incremental freight and factory cost are missing, so a faster scenario is not converted into a profit claim "
                "unless you enter a labeled sensitivity assumption."
            ),
        }
    )
    return findings
