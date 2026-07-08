"""
Basic unit tests for the parts of the pipeline that don't require a live
MLflow/DVC connection: model training and evaluation logic.
"""

import sys
from pathlib import Path

# افزودن مسیر src به sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from pipeline import MLOpsPipeline


def _make_pipeline_without_init():
    """Build an MLOpsPipeline instance without running __init__."""
    pipeline = MLOpsPipeline.__new__(MLOpsPipeline)
    return pipeline


def test_train_model_returns_fitted_estimator():
    pipeline = _make_pipeline_without_init()
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    model = pipeline._train_model(X, y, {"n_estimators": 10, "max_depth": 3, "random_state": 42})
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "classes_")


def test_evaluate_model_returns_expected_keys():
    pipeline = _make_pipeline_without_init()
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    model = pipeline._train_model(X, y, {"n_estimators": 10, "max_depth": 3, "random_state": 42})
    metrics = pipeline._evaluate_model(model, X, y)

    assert set(metrics.keys()) == {"accuracy", "f1_score", "roc_auc"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0