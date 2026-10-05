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
