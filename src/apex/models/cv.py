"""Stratified K-fold cross-validation (CV) on the train split.

A single validation score moves by about ±0.01 PR-AUC depending on which
leads land in it, so model choices are made on the mean ± std over 5 folds
of the train split. The validation split confirms the final choice, and the
test split stays untouched.

    python -m apex.models.cv features   # step 2.4: all vs. selected features (`make cv`)
    python -m apex.models.cv tune       # step 2.5: grid over C and class_weight (`make tune`)

Every result is logged to MLflow; decisions are in docs/models.md.
"""

import copy
import itertools
import sys
from typing import Any

import mlflow
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from apex.config import load_config
from apex.data.split import load_splits
from apex.models.evaluate import evaluate
from apex.models.train import EXPERIMENT, make_model

N_FOLDS = 5
GRID = {"C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0], "class_weight": [None, "balanced"]}
SHOWN = ["pr_auc", "roc_auc", "brier", "precision_top20", "recall_top20"]


def cross_validate(
    X: pd.DataFrame, y: pd.Series, config: dict[str, Any] | None = None, **params: Any
) -> pd.DataFrame:
    """Metrics for each fold (one row per fold)."""
    cfg = config or load_config()
    folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=cfg["split"]["random_state"])
    rows = []
    for fit_idx, score_idx in folds.split(X, y):
        model = make_model(cfg, **params).fit(X.iloc[fit_idx], y.iloc[fit_idx])
        rows.append(evaluate(y.iloc[score_idx], model.predict_proba(X.iloc[score_idx])[:, 1]))
    return pd.DataFrame(rows)


def all_features(config: dict[str, Any]) -> dict[str, Any]:
    """The config before feature selection: only the redundant column is dropped."""
    cfg = copy.deepcopy(config)
    cfg["features"]["drop"] = cfg["features"]["drop"][:1]
    cfg["features"]["presence_only"] = []
    return cfg


def _cv_and_log(name: str, X, y, config, params: dict[str, Any], extra: dict) -> pd.Series:
    """Cross-validate one setup, log it to MLflow, and return mean metrics (+ pr_auc_std)."""
    folds = cross_validate(X, y, config, **params)
    mean, std = folds.mean(), folds.std()
    with mlflow.start_run(run_name=name):
        mlflow.log_params({"model": "logistic_regression", "folds": N_FOLDS, **params, **extra})
        mlflow.log_metrics({f"cv_{k}_mean": v for k, v in mean.items()})
        mlflow.log_metrics({f"cv_{k}_std": v for k, v in std.items()})
    return pd.concat([mean[SHOWN], pd.Series({"pr_auc_std": std["pr_auc"]})])


def compare_features(X, y, cfg) -> pd.DataFrame:
    """Step 2.4: all features vs. the selected features."""
    rows = {}
    for name, variant in {"all_features": all_features(cfg), "selected_features": cfg}.items():
        n_features = len(make_model(variant)[0].fit(X)[-1].get_feature_names_out())
        extra = {"n_features": n_features, "dropped": variant["features"]["drop"]}
        rows[name] = _cv_and_log(f"cv_{name}", X, y, variant, {}, extra)
        rows[name]["features"] = n_features
    return pd.DataFrame(rows).T


def tune(X, y, cfg) -> pd.DataFrame:
    """Step 2.5: every combination in GRID, best CV PR-AUC first."""
    rows = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values, strict=True))
        name = "tune_" + "_".join(f"{k}={v}" for k, v in params.items())
        shown = {k: "none" if v is None else v for k, v in params.items()}
        rows.append({**shown, **_cv_and_log(name, X, y, cfg, params, {})})
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False, ignore_index=True)


def run(command: str = "features") -> pd.DataFrame:
    """Run one CV command on the train split and print its table."""
    cfg = load_config()
    mlflow.set_tracking_uri(cfg["paths"]["mlflow_tracking_uri"])
    mlflow.set_experiment(EXPERIMENT)

    train = load_splits()["train"]
    target = cfg["data"]["target"]
    X, y = train.drop(columns=[target]), train[target]

    table = {"features": compare_features, "tune": tune}[command](X, y, cfg)
    print(f"{N_FOLDS}-fold CV on the train split ({len(X):,} leads)\n")
    print(table.round(4).to_string())
    return table


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "features")
