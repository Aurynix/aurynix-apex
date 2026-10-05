import numpy as np
import pandas as pd

from apex.config import load_config
from apex.models.cv import N_FOLDS, all_features, cross_validate


def test_cross_validate_returns_one_row_per_fold():
    rng = np.random.default_rng(0)
    time = rng.integers(0, 2000, 300)
    X = pd.DataFrame(
        {
            "Lead Origin": rng.choice(["API", "Landing Page Submission"], 300),
            "TotalVisits": rng.integers(1, 6, 300).astype(float),
            "Total Time Spent on Website": time,
        }
    )
    y = pd.Series((time > 1000).astype(int))
    folds = cross_validate(X, y)
    assert len(folds) == N_FOLDS
    assert folds["pr_auc"].mean() > 0.9


def test_all_features_undoes_feature_selection_only():
    cfg = load_config()
    full = all_features(cfg)
    assert full["features"]["drop"] == ["What matters most to you in choosing a course"]
    assert full["features"]["presence_only"] == []
    assert len(cfg["features"]["drop"]) > 1  # the original config is not changed


def test_tune_tries_every_combination_best_first(tmp_path, monkeypatch):
    import mlflow

    from apex.models import cv

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test")
    monkeypatch.setattr(cv, "GRID", {"C": [0.01, 1.0], "class_weight": [None, "balanced"]})

    rng = np.random.default_rng(0)
    time = rng.integers(0, 2000, 200)
    X = pd.DataFrame(
        {"TotalVisits": rng.integers(1, 6, 200).astype(float), "Total Time Spent on Website": time}
    )
    y = pd.Series((time > 1000).astype(int))

    table = cv.tune(X, y, load_config())
    assert len(table) == 4
    assert table["pr_auc"].is_monotonic_decreasing
    assert set(table["class_weight"]) == {"none", "balanced"}
