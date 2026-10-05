"""Train models and log them to MLflow.

The model is Logistic Regression on the feature pipeline (ADR-003). Step 2.3
compares it with the no-skill baseline: both are fitted on the train split,
scored on the validation split, and logged to MLflow.

Commands:
    python -m apex.models.train baselines   # step 2.3: train → validation (`make baselines`)
    python -m apex.models.train test        # step 3.1: train + val → test, ONCE (`make evaluate`)
    python -m apex.models.train calibration # step 3.2: calibration check (`make calibration`)

Then `make mlflow-ui` to browse the runs.
"""

import sys
from typing import Any

import mlflow
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline

from apex.config import load_config, path
from apex.data.features import build_pipeline
from apex.data.split import load_splits
from apex.models.evaluate import calibration_table, evaluate, plot_calibration, plot_test_report

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


def run_test() -> dict[str, float]:
    """Step 3.1: retrain on train + validation, score the test split once, log and plot."""
    cfg = load_config()
    mlflow.set_tracking_uri(cfg["paths"]["mlflow_tracking_uri"])
    mlflow.set_experiment(EXPERIMENT)

    parts = load_splits(include_test=True)
    target = cfg["data"]["target"]
    fit = pd.concat([parts["train"], parts["val"]])
    test = parts["test"]

    model = make_model(cfg).fit(fit.drop(columns=[target]), fit[target])
    scores = model.predict_proba(test.drop(columns=[target]))[:, 1]
    metrics = evaluate(test[target], scores)

    figure = path("figures_dir") / "test_evaluation.png"
    plot_test_report(test[target], scores, figure, f"Test set ({len(test):,} leads, used once)")

    with mlflow.start_run(run_name="final_test"):
        mlflow.log_params({"model": "logistic_regression", "fit_rows": len(fit), **_params(model)})
        mlflow.log_params({"test_rows": len(test)})
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.log_artifact(str(figure))

    print(f"Fitted on train + validation ({len(fit):,} leads); test split ({len(test):,} leads)\n")
    print(pd.Series(metrics).to_string())
    print(f"\nfigure: {figure}")
    return metrics


def run_calibration() -> pd.DataFrame:
    """Step 3.2: compare raw, Platt, and isotonic probabilities on out-of-fold predictions.

    Every lead in train + validation is scored by a model that did not see it
    (5-fold), so calibration is checked on held-out data without the test split.
    """
    cfg = load_config()
    mlflow.set_tracking_uri(cfg["paths"]["mlflow_tracking_uri"])
    mlflow.set_experiment(EXPERIMENT)

    parts = load_splits()
    target = cfg["data"]["target"]
    data = pd.concat([parts["train"], parts["val"]])
    X, y = data.drop(columns=[target]), data[target]
    folds = StratifiedKFold(5, shuffle=True, random_state=cfg["split"]["random_state"])

    candidates = {
        "raw": make_model(cfg),
        "platt": CalibratedClassifierCV(make_model(cfg), method="sigmoid", cv=5),
        "isotonic": CalibratedClassifierCV(make_model(cfg), method="isotonic", cv=5),
    }
    rows, curves = {}, {}
    for name, model in candidates.items():
        scores = cross_val_predict(model, X, y, cv=folds, method="predict_proba")[:, 1]
        metrics = evaluate(y, scores)
        rows[name] = {k: metrics[k] for k in ("ece", "brier", "pr_auc", "roc_auc")}
        rows[name]["mean_predicted"] = round(float(scores.mean()), 4)
        curves[name] = calibration_table(y, scores)
        with mlflow.start_run(run_name=f"calibration_{name}"):
            mlflow.log_params({"model": "logistic_regression", "calibration": name})
            mlflow.log_metrics({f"oof_{k}": v for k, v in metrics.items()})

    figure = path("figures_dir") / "calibration.png"
    plot_calibration(curves, figure, f"Calibration, out-of-fold ({len(X):,} leads)")

    print(f"Out-of-fold predictions on train + validation ({len(X):,} leads)")
    print(f"actual conversion rate: {y.mean():.4f}\n")
    print(pd.DataFrame(rows).T.to_string())
    print("\nraw model, per bin:\n", curves["raw"].round(3).to_string())
    print(f"\nfigure: {figure}")
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    commands = {"baselines": run_baselines, "test": run_test, "calibration": run_calibration}
    command = sys.argv[1] if len(sys.argv) > 1 else "baselines"
    commands[command]()
