"""Train, compare, and persist lead-time models without target leakage."""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    BASE_FEATURES,
    CATEGORICAL_FEATURES,
    CV_MAE_TOLERANCE_DAYS,
    LONG_LEAD_TIME_DAYS,
    MIN_ROWS_TO_TRAIN,
    MODEL_SIMPLICITY,
    MODEL_VERSION,
    MODELS_DIR,
    NUMERIC_FEATURES,
    REPORTS_DIR,
    SEED,
)
from .features import (
    apply_train_residual,
    chronological_split,
    date_folds,
    feature_frame,
    fit_cohort_medians,
    leakage_audit,
    model_ready_frame,
)
from .reporting import model_card_markdown, quality_json

RAW_TARGET = "lead_time_days"
RESIDUAL_TARGET = "operational_residual_days"


class GroupMedianBaseline:
    """Median lead time by ship mode, with a global median fallback.

    Ship mode is the relevant historical group. The baseline does not use
    factory, because factory origin is not observed.
    """

    def __init__(self, group_cols: list[str] | None = None):
        self.group_cols = list(group_cols or ["ship_mode"])
        self.global_median_: float | None = None
        self.medians_: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "GroupMedianBaseline":
        y_arr = np.asarray(y, dtype=float)
        self.global_median_ = float(np.median(y_arr))
        grouped = X.loc[:, self.group_cols].copy()
        for col in self.group_cols:
            grouped[col] = grouped[col].astype("string")
        grouped["_y"] = y_arr
        self.medians_ = grouped.groupby(self.group_cols, dropna=False)["_y"].median()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.medians_ is None or self.global_median_ is None:
            raise RuntimeError("Baseline must be fit before predict.")
        keys = X.loc[:, self.group_cols].copy()
        for col in self.group_cols:
            keys[col] = keys[col].astype("string")
        medians = self.medians_.rename("_pred").reset_index()
        merged = keys.reset_index(drop=True).merge(medians, on=self.group_cols, how="left")
        return merged["_pred"].fillna(self.global_median_).to_numpy(dtype=float)


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float | int | None]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"mae": None, "rmse": None, "r2": None, "n": 0}
    r2 = None
    if len(y_true) > 1 and float(np.var(y_true)) > 0:
        r2 = float(r2_score(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2,
        "n": int(len(y_true)),
    }


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not rows:
        return {"mae": None, "rmse": None, "r2": None, "n_folds": 0}
    def _avg(key: str) -> float | None:
        vals = [row[key] for row in rows if row.get(key) is not None]
        return float(np.mean(vals)) if vals else None
    return {
        "mae": _avg("mae"),
        "rmse": _avg("rmse"),
        "r2": _avg("r2"),
        "n_folds": len(rows),
    }


def make_regressor_pipeline(estimator: Any, feature_columns: list[str], scale_numeric: bool) -> Pipeline:
    cat_cols = [col for col in feature_columns if col in CATEGORICAL_FEATURES or col == "origin_factory"]
    num_cols = [col for col in feature_columns if col in NUMERIC_FEATURES]
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
            ("num", Pipeline(steps=numeric_steps), num_cols),
        ]
    )
    return Pipeline([("prep", preprocessor), ("model", estimator)])


def _builders(feature_columns: list[str], tiny: bool) -> dict[str, Any]:
    rf_trees = 8 if tiny else 80
    gb_trees = 8 if tiny else 80
    rf_depth = 4 if tiny else 8
    gb_depth = 2 if tiny else 3

    def baseline() -> GroupMedianBaseline:
        return GroupMedianBaseline(["ship_mode"])

    def linear() -> Pipeline:
        return make_regressor_pipeline(LinearRegression(), feature_columns, scale_numeric=True)

    def forest() -> Pipeline:
        return make_regressor_pipeline(
            RandomForestRegressor(
                n_estimators=rf_trees,
                max_depth=rf_depth,
                min_samples_leaf=5,
                random_state=SEED,
                n_jobs=1,
            ),
            feature_columns,
            scale_numeric=False,
        )

    def boosting() -> Pipeline:
        return make_regressor_pipeline(
            GradientBoostingRegressor(
                n_estimators=gb_trees,
                learning_rate=0.1 if tiny else 0.08,
                max_depth=gb_depth,
                random_state=SEED,
            ),
            feature_columns,
            scale_numeric=False,
        )

    return {
        "group_median_baseline": baseline,
        "linear_regression": linear,
        "random_forest": forest,
        "gradient_boosting": boosting,
    }


def _fit_predict(model: Any, train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str) -> np.ndarray:
    X_train = feature_frame(train, features)
    y_train = train[target].astype(float)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        model.fit(X_train, y_train)
        return np.asarray(model.predict(feature_frame(test, features)), dtype=float)


def _select_model(candidates: dict[str, dict[str, Any]]) -> tuple[str, str]:
    scored = []
    for name, payload in candidates.items():
        mae = payload.get("cv", {}).get("mae")
        if mae is None:
            mae = payload.get("train", {}).get("mae")
        scored.append((name, mae))
    scored = [item for item in scored if item[1] is not None]
    if not scored:
        return "group_median_baseline", "No validation score was available; the ship-mode median baseline is used."
    best_mae = min(mae for _, mae in scored)
    within = [name for name, mae in scored if mae <= best_mae + CV_MAE_TOLERANCE_DAYS]
    selected = sorted(within, key=lambda name: (MODEL_SIMPLICITY.get(name, 99), name))[0]
    reason = (
        f"Selected {selected} by mean chronological cross-validation MAE on the training period. "
        f"Models within {CV_MAE_TOLERANCE_DAYS:.0f} day of the best CV MAE are treated as tied, "
        "and the simpler model is kept. The test set was not used for selection."
    )
    return selected, reason


def _subgroup_mae(frame: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, columns: list[str]) -> dict[str, list[dict[str, Any]]]:
    work = frame.loc[:, columns].copy()
    work["y_true"] = y_true
    work["y_pred"] = y_pred
    work["abs_error"] = (work["y_true"] - work["y_pred"]).abs()
    out: dict[str, list[dict[str, Any]]] = {}
    for col in columns:
        grouped = (
            work.groupby(col, dropna=False)
            .agg(mae=("abs_error", "mean"), n=("abs_error", "size"))
            .reset_index()
            .sort_values("mae", ascending=False)
        )
        records = []
        for row in grouped.itertuples(index=False):
            records.append({col: getattr(row, col), "mae": float(row.mae), "n": int(row.n)})
        out[col] = records
    return out


def feature_importance(model: Any, top_n: int = 12) -> list[dict[str, float | str]]:
    if not isinstance(model, Pipeline) or "prep" not in model.named_steps:
        return []
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        kind = "impurity_importance"
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype=float)).ravel()
        kind = "absolute_coefficient"
    else:
        return []
    names = model.named_steps["prep"].get_feature_names_out()
    order = np.argsort(values)[::-1][:top_n]
    return [
        {"feature": str(names[idx]), "importance": float(values[idx]), "kind": kind}
        for idx in order
    ]


def outlier_treatment_summary(df: pd.DataFrame) -> dict[str, Any]:
    flagged = df["lead_time_days"] > LONG_LEAD_TIME_DAYS
    retained = int((~flagged.fillna(False)).sum())
    return {
        "rule": f"Lead time greater than {LONG_LEAD_TIME_DAYS} calendar days is unusually long for a ship-date target.",
        "rows_flagged": int(flagged.fillna(False).sum()),
        "rows_retained_if_excluded": retained,
        "rows_deleted": 0,
        "model_trained_on_retained_rows": retained >= MIN_ROWS_TO_TRAIN,
        "reason": (
            "No rows were deleted. Excluding unusually long lead times does not leave enough rows to train "
            "when the date anomaly affects the whole extract. A parallel within-cohort residual target is "
            "reported as a documented sensitivity analysis, not as a replacement for the raw target."
            if retained < MIN_ROWS_TO_TRAIN
            else "A sensitivity model can be trained on retained rows. The primary model still keeps every valid row."
        ),
    }


def _evaluate_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str,
    tiny: bool,
) -> dict[str, Any]:
    builders = _builders(features, tiny=tiny)
    folds = date_folds(train, fractions=(0.7,) if tiny else (0.55, 0.70, 0.85))
    candidates: dict[str, dict[str, Any]] = {}
    fitted: dict[str, Any] = {}
    test_predictions: dict[str, np.ndarray] = {}
    for name, builder in builders.items():
        cv_rows = []
        for fold_train, fold_valid in folds:
            fold_model = builder()
            pred = _fit_predict(fold_model, fold_train, fold_valid, features, target)
            cv_rows.append(regression_metrics(fold_valid[target], pred))
        model = builder()
        train_pred = _fit_predict(model, train, train, features, target)
        test_pred = np.asarray(model.predict(feature_frame(test, features)), dtype=float)
        fitted[name] = model
        test_predictions[name] = test_pred
        candidates[name] = {
            "cv": _mean_metrics(cv_rows),
            "train": regression_metrics(train[target], train_pred),
            "test": regression_metrics(test[target], test_pred),
        }
    global_median = float(np.median(train[target].astype(float)))
    global_pred = np.full(len(test), global_median)
    candidates["global_median"] = {
        "cv": {"mae": None, "rmse": None, "r2": None, "n_folds": 0},
        "train": regression_metrics(train[target], np.full(len(train), global_median)),
        "test": regression_metrics(test[target], global_pred),
        "note": "Reference only. Not eligible for selection; the grouped ship-mode median is the baseline.",
    }
    selected, reason = _select_model({k: v for k, v in candidates.items() if k != "global_median"})
    selected_pred = test_predictions[selected]
    residuals = test[target].to_numpy(dtype=float) - selected_pred
    alternate_name = "linear_regression" if selected != "linear_regression" else "group_median_baseline"
    return {
        "selected_model": selected,
        "alternate_model": alternate_name,
        "selection_reason": reason,
        "candidates": candidates,
        "test_residual_p10": float(np.nanpercentile(residuals, 10)),
        "test_residual_p90": float(np.nanpercentile(residuals, 90)),
        "subgroup_mae": _subgroup_mae(test, test[target].to_numpy(dtype=float), selected_pred, ["region", "ship_mode", "division", "product_name"]),
        "feature_importance": feature_importance(fitted[selected]),
        "models": {
            "selected": fitted[selected],
            "alternate": fitted[alternate_name],
            "baseline": fitted["group_median_baseline"],
        },
    }


def train_and_evaluate(
    df: pd.DataFrame,
    quality: dict[str, Any],
    dataset_sha256: str,
    tiny: bool = False,
) -> dict[str, Any]:
    if not quality.get("training_allowed", False):
        raise RuntimeError("Training is blocked: " + "; ".join(quality.get("training_blockers", [])))
    ready = model_ready_frame(df)
    if len(ready) < MIN_ROWS_TO_TRAIN:
        raise RuntimeError("Not enough rows with a usable lead time.")

    origin_usable = bool(quality.get("origin_profile", {}).get("usable_for_training"))
    features = list(BASE_FEATURES)
    if origin_usable:
        features.append("origin_factory")
    audit = leakage_audit(features)
    if not audit["passed"]:
        raise RuntimeError(f"Leakage check failed: {audit['forbidden_found']}")

    train, test, split_info = chronological_split(ready)
    cohort_medians = fit_cohort_medians(train)
    global_median = float(train["lead_time_days"].median())
    train = apply_train_residual(train, cohort_medians, global_median)
    test = apply_train_residual(test, cohort_medians, global_median)

    raw = _evaluate_target(train, test, features, RAW_TARGET, tiny=tiny)
    residual = _evaluate_target(train, test, features, RESIDUAL_TARGET, tiny=tiny)
    treatment = outlier_treatment_summary(ready)
    if treatment["model_trained_on_retained_rows"]:
        retained = ready.loc[ready["lead_time_days"] <= LONG_LEAD_TIME_DAYS].copy()
        try:
            retained_train, retained_test, _ = chronological_split(retained)
            retained_eval = _evaluate_target(retained_train, retained_test, features, RAW_TARGET, tiny=True)
            treatment["retained_test_mae"] = retained_eval["candidates"][retained_eval["selected_model"]]["test"]["mae"]
            treatment["retained_selected_model"] = retained_eval["selected_model"]
        except Exception as exc:  # noqa: BLE001 - report the limitation instead of hiding it
            treatment["retained_training_error"] = str(exc)
    else:
        treatment["retained_test_mae"] = None
        treatment["retained_selected_model"] = None

    trained_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    bundle = {
        "model_version": MODEL_VERSION,
        "trained_at": trained_at,
        "dataset_sha256": dataset_sha256,
        "seed": SEED,
        "tiny": tiny,
        "factory_effect_learned": origin_usable,
        "origin_column": quality.get("origin_profile", {}).get("source_column") if origin_usable else None,
        "feature_columns": features,
        "leakage_audit": audit,
        "split": split_info,
        "cohort_medians_train": cohort_medians,
        "global_median_train": global_median,
        "outlier_treatment": treatment,
        "target_credibility": quality.get("target_credibility"),
        "target_quality_penalty": quality.get("target_quality_penalty"),
        "targets": {
            RAW_TARGET: {k: v for k, v in raw.items() if k != "models"},
            RESIDUAL_TARGET: {k: v for k, v in residual.items() if k != "models"},
        },
        "models": {
            "lead_time_selected": raw["models"]["selected"],
            "lead_time_alternate": raw["models"]["alternate"],
            "lead_time_baseline": raw["models"]["baseline"],
            "residual_selected": residual["models"]["selected"],
            "residual_alternate": residual["models"]["alternate"],
            "residual_baseline": residual["models"]["baseline"],
        },
    }
    return bundle


def predict_frame(bundle: dict[str, Any], frame: pd.DataFrame, which: str = "lead_time_selected") -> np.ndarray:
    model = bundle["models"][which]
    columns = bundle["feature_columns"]
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    return np.asarray(model.predict(feature_frame(frame, columns)), dtype=float)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def save_bundle(bundle: dict[str, Any], directory: Path | None = None, quality: dict[str, Any] | None = None) -> Path:
    directory = directory or MODELS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    models = bundle["models"]
    meta = {key: value for key, value in bundle.items() if key != "models"}
    (directory / "metrics.json").write_text(json.dumps(jsonable(meta), indent=2), encoding="utf-8")
    joblib.dump(models, directory / "models.joblib")
    card = model_card_markdown(meta, quality)
    (REPORTS_DIR / "model_card.md").write_text(card, encoding="utf-8")
    (directory / "MODEL_CARD.md").write_text(card, encoding="utf-8")
    if quality is not None:
        (REPORTS_DIR / "data_quality.json").write_text(quality_json(quality), encoding="utf-8")
    return directory


def load_bundle(directory: Path | None = None) -> dict[str, Any] | None:
    directory = directory or MODELS_DIR
    metrics_path = directory / "metrics.json"
    model_path = directory / "models.joblib"
    if not metrics_path.exists() or not model_path.exists():
        return None
    meta = json.loads(metrics_path.read_text(encoding="utf-8"))
    meta["models"] = joblib.load(model_path)
    return meta


def main() -> None:
    from .data_loader import file_sha256, load_orders, validate_orders, write_reference_files
    from .config import DEFAULT_ORDERS_PATH

    write_reference_files()
    orders = load_orders(DEFAULT_ORDERS_PATH)
    quality = validate_orders(orders)
    digest = file_sha256(DEFAULT_ORDERS_PATH)
    print(f"Training on {len(orders):,} rows, sha256={digest[:12]}")
    bundle = train_and_evaluate(orders, quality, digest, tiny=False)
    save_bundle(bundle, quality=quality)
    raw = bundle["targets"][RAW_TARGET]
    residual = bundle["targets"][RESIDUAL_TARGET]
    print(f"Raw target selected: {raw['selected_model']}")
    print(f"Residual target selected: {residual['selected_model']}")
    for name, payload in raw["candidates"].items():
        test = payload["test"]
        print(f"  raw {name:24} test MAE={test['mae']:.2f} RMSE={test['rmse']:.2f} R2={test['r2']}")
    for name, payload in residual["candidates"].items():
        test = payload["test"]
        print(f"  residual {name:20} test MAE={test['mae']:.3f} RMSE={test['rmse']:.3f} R2={test['r2']}")
    print(f"Saved artifacts to {MODELS_DIR}")


if __name__ == "__main__":
    # Running this file directly would pickle GroupMedianBaseline as __main__.
    # Import the package path so saved artifacts load inside the app.
    from src.train import main as train_main

    train_main()
