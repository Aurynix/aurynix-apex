"""Stratified K-fold cross-validation on the train split (step 2.4).

A single validation score moves by about ±0.01 PR-AUC depending on which
leads land in it, so model choices are made on the mean ± std over 5 folds
of the train split. The validation split confirms the final choice, and the
test split stays untouched.

Run `python -m apex.models.cv` (or `make cv`) to compare all features with the
selected features (docs/models.md) and log both to MLflow.
"""

import copy
from typing import Any

import mlflow
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from apex.config import load_config
from apex.data.split import load_splits
from apex.models.evaluate import evaluate
from apex.models.train import EXPERIMENT, make_model

N_FOLDS = 5


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


def run() -> pd.DataFrame:
    """Cross-validate all vs. selected features, log both to MLflow, print a table."""
    cfg = load_config()
    mlflow.set_tracking_uri(cfg["paths"]["mlflow_tracking_uri"])
    mlflow.set_experiment(EXPERIMENT)

    train = load_splits()["train"]
    target = cfg["data"]["target"]
    X, y = train.drop(columns=[target]), train[target]

    rows = {}
    for name, variant in {"all_features": all_features(cfg), "selected_features": cfg}.items():
        folds = cross_validate(X, y, variant)
        mean, std = folds.mean(), folds.std()
        n_features = len(make_model(variant)[0].fit(X)[-1].get_feature_names_out())
        with mlflow.start_run(run_name=f"cv_{name}"):
            mlflow.log_params({"model": "logistic_regression", "folds": N_FOLDS})
            mlflow.log_params({"n_features": n_features, "dropped": variant["features"]["drop"]})
            mlflow.log_metrics({f"cv_{k}_mean": v for k, v in mean.items()})
            mlflow.log_metrics({f"cv_{k}_std": v for k, v in std.items()})
        rows[name] = {
            "features": n_features,
            "pr_auc": f"{mean['pr_auc']:.4f} ± {std['pr_auc']:.4f}",
            "roc_auc": f"{mean['roc_auc']:.4f}",
            "brier": f"{mean['brier']:.4f}",
            "precision_top20": f"{mean['precision_top20']:.4f}",
            "recall_top20": f"{mean['recall_top20']:.4f}",
        }

    table = pd.DataFrame(rows).T
    print(f"{N_FOLDS}-fold CV on the train split ({len(X):,} leads)\n")
    print(table.to_string())
    return table


if __name__ == "__main__":
    run()
