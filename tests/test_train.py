"""Training pipeline, held-out metrics, and artifact round trip."""

from __future__ import annotations

import numpy as np

from src.config import MODEL_VERSION
from src.data_loader import validate_orders
from src.features import leakage_audit
from src.train import outlier_treatment_summary, predict_frame, train_and_evaluate


def test_saved_bundle_unpickles_from_package_path():
    from src.config import MODELS_DIR
    from src.train import GroupMedianBaseline, load_bundle

    bundle = load_bundle(MODELS_DIR)
    assert bundle is not None
    assert bundle["factory_effect_learned"] is False
    assert isinstance(bundle["models"]["lead_time_selected"], GroupMedianBaseline)
    assert type(bundle["models"]["lead_time_selected"]).__module__ == "src.train"


def test_outlier_treatment_does_not_delete_rows(orders):
    summary = outlier_treatment_summary(orders.dropna(subset=["lead_time_days"]))
    assert summary["rows_deleted"] == 0
    assert summary["rows_flagged"] == len(orders)
    assert summary["model_trained_on_retained_rows"] is False


def test_tiny_training_is_leakage_safe_and_compares_required_models(orders):
    sample = orders.sort_values("order_date").iloc[::5].copy()
    quality = validate_orders(sample)
    bundle = train_and_evaluate(sample, quality, dataset_sha256="test-sha", tiny=True)
    audit = leakage_audit(bundle["feature_columns"])
    assert audit["passed"]
    assert bundle["factory_effect_learned"] is False
    assert bundle["model_version"] == MODEL_VERSION
    raw = bundle["targets"]["lead_time_days"]
    for name in ["group_median_baseline", "linear_regression", "random_forest", "gradient_boosting"]:
        assert name in raw["candidates"]
        assert raw["candidates"][name]["test"]["mae"] is not None
        assert raw["candidates"][name]["cv"]["mae"] is not None
    assert raw["selected_model"] in raw["candidates"]
    assert "test set was not used for selection" in raw["selection_reason"].lower()
    residual = bundle["targets"]["operational_residual_days"]
    assert residual["candidates"]["group_median_baseline"]["test"]["mae"] < 5
    frame = sample.head(3).copy()
    frame["origin_factory"] = "Sugar Shack"
    first = predict_frame(bundle, frame)
    frame = frame.copy()
    frame["origin_factory"] = "Secret Factory"
    second = predict_frame(bundle, frame)
    assert np.allclose(first, second)
