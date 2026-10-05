"""Train models and log them to MLflow.

The model is Logistic Regression on the feature pipeline (ADR-003). Step 2.3
compares it with the no-skill baseline: both are fitted on the train split,
scored on the validation split, and logged to MLflow.

Run `python -m apex.models.train baselines` (or `make baselines`), then
`make mlflow-ui` to browse the runs.
"""

import sys
from typing import Any

import mlflow
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

from apex.config import load_config
from apex.data.features import build_pipeline
from apex.data.split import load_splits
from apex.models.evaluate import evaluate

EXPERIMENT = "apex-lead-scoring"


def make_model(config: dict[str, Any] | None = None, **params: Any) -> Pipeline:
    """Feature pipeline + Logistic Regression.

    Settings come from `config.json → model` (chosen in step 2.5); extra params override them.
    """
    cfg = config or load_config()
    model = LogisticRegression(max_iter=1000, random_state=cfg["split"]["random_state"])
    return make_pipeline(build_pipeline(cfg), model.set_params(**{**cfg["model"], **params}))


def make_baselines(config: dict[str, Any] | None = None) -> dict[str, Pipeline]:
    """The bar to beat: no skill (predicts the conversion rate for everyone) and plain LR."""
    cfg = config or load_config()
    return {
        "no_skill": make_pipeline(build_pipeline(cfg), DummyClassifier(strategy="prior")),
        "logistic_regression": make_model(cfg),
    }


def fit_and_log(name: str, model: Pipeline, parts: dict[str, pd.DataFrame], target: str) -> dict:
    """Fit on train, score on val, and log params + metrics as one MLflow run."""
    X_train, y_train = parts["train"].drop(columns=[target]), parts["train"][target]
    X_val, y_val = parts["val"].drop(columns=[target]), parts["val"][target]

    model.fit(X_train, y_train)
    metrics = evaluate(y_val, model.predict_proba(X_val)[:, 1])

    with mlflow.start_run(run_name=name):
        mlflow.log_params(
            {"model": name, "train_rows": len(X_train), "val_rows": len(X_val), **_params(model)}
        )
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
    return metrics


def _params(model: Pipeline) -> dict[str, Any]:
    """Main settings of the final estimator, for MLflow."""
    keep = ("C", "penalty", "class_weight", "solver", "max_iter", "strategy")
    return {k: v for k, v in model[-1].get_params().items() if k in keep}


def run_baselines() -> pd.DataFrame:
    """Fit and log every baseline; print a comparison table on the validation split."""
    cfg = load_config()
    mlflow.set_tracking_uri(cfg["paths"]["mlflow_tracking_uri"])
    mlflow.set_experiment(EXPERIMENT)

    parts = load_splits()
    target = cfg["data"]["target"]
    results = pd.DataFrame(
        {
            name: fit_and_log(name, model, parts, target)
            for name, model in make_baselines(cfg).items()
        }
    ).T
    print(f"Validation split ({len(parts['val']):,} leads)\n")
    print(results.to_string())
    return results


if __name__ == "__main__":
    commands = {"baselines": run_baselines}
    command = sys.argv[1] if len(sys.argv) > 1 else "baselines"
    commands[command]()
