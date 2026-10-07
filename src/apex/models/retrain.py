"""Safe retraining: replace the saved model only if a new one is at least as good.

    python -m apex.models.retrain   (or `make retrain`)

1. The current model (champion) and a new model trained on the current data
   (challenger) are scored on the same held-out leads:
   - real outcomes from POST /outcomes, when there are at least `monitoring.min_outcomes`
     (production leads that neither model was trained on: the fairest test);
   - otherwise a stratified `retrain.holdout_share` of the data file, kept out of the
     challenger's training. The champion may have seen these rows, which favors it:
     replacing the model is made harder, never easier.
2. If challenger PR-AUC ≥ champion PR-AUC + `retrain.min_gain`, the current artifacts
   are copied to models/archive/<trained_at>/ and the final model is refitted on all
   data (`make train`). Otherwise nothing changes.

The API keeps serving the model it loaded; restart it to use a new model.
"""

import json
import shutil
from typing import Any

import joblib
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from apex.api.database import connect, read_outcome_leads
from apex.config import load_config, path
from apex.data.load import load_raw
from apex.models.evaluate import evaluate
from apex.models.train import experiment, make_model, run_final


def holdout(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """(training rows for the challenger, held-out rows, where the held-out rows come from)."""
    target = cfg["data"]["target"]
    raw = load_raw()
    db_file = path("database")
    outcomes = read_outcome_leads(connect(db_file)) if db_file.exists() else []
    if len(outcomes) >= cfg["monitoring"]["min_outcomes"]:
        leads = pd.DataFrame([lead for lead, _ in outcomes])
        leads[target] = [converted for _, converted in outcomes]
        if leads[target].nunique() == 2:
            return raw, leads, f"{len(leads):,} real outcomes"
    fit, held = train_test_split(
        raw,
        test_size=cfg["retrain"]["holdout_share"],
        stratify=raw[target],
        random_state=cfg["split"]["random_state"],
    )
    return fit, held, f"{len(held):,} held-out rows of {raw.shape[0]:,} in the data file"


def compare(champion, challenger, held: pd.DataFrame, target: str) -> dict[str, dict]:
    """Metrics of both models on the same held-out rows."""
    X, y = held.drop(columns=[target]), held[target]
    return {
        name: evaluate(y, model.predict_proba(X)[:, 1])
        for name, model in {"champion": champion, "challenger": challenger}.items()
    }


def archive_current() -> str:
    """Copy the current artifacts to models/archive/<trained_at>/; return that folder."""
    cfg = load_config()
    out, files = path("models_dir"), cfg["files"]
    trained_at = json.loads((out / files["model_meta"]).read_text())["trained_at"]
    folder = out / "archive" / trained_at.replace(":", "-")
    folder.mkdir(parents=True, exist_ok=True)
    for key in ("model", "model_meta", "reference_profile", "model_card"):
        if (out / files[key]).exists():
            shutil.copy2(out / files[key], folder / files[key])
    return str(folder)


def run() -> dict[str, Any]:
    cfg = load_config()
    target = cfg["data"]["target"]
    current = path("models_dir") / cfg["files"]["model"]
    if not current.exists():
        print("No saved model yet: training the first one.")
        run_final()
        return {"decision": "first model"}

    fit, held, source = holdout(cfg)
    challenger = make_model(cfg).fit(fit.drop(columns=[target]), fit[target])
    scores = compare(joblib.load(current), challenger, held, target)
    champion_pr, challenger_pr = scores["champion"]["pr_auc"], scores["challenger"]["pr_auc"]
    promote = challenger_pr >= champion_pr + cfg["retrain"]["min_gain"]

    mlflow.set_tracking_uri(cfg["paths"]["mlflow_tracking_uri"])
    mlflow.set_experiment(experiment())
    with mlflow.start_run(run_name="retrain"):
        mlflow.log_params({"holdout": source, "promoted": promote})
        for name, metrics in scores.items():
            mlflow.log_metrics({f"{name}_{k}": v for k, v in metrics.items()})

    print(f"Compared on {source}\n")
    print(pd.DataFrame(scores).loc[["pr_auc", "roc_auc", "recall_top20", "precision_top20"]])
    if not promote:
        print(
            f"\nKept the current model: the new one is not better "
            f"(PR-AUC {challenger_pr} vs {champion_pr})."
        )
        return {"decision": "kept", "scores": scores, "holdout": source}

    folder = archive_current()
    print(f"\nNew model is at least as good ({challenger_pr} vs {champion_pr}).")
    print(f"Previous model archived in {folder}; refitting the new model on all data:\n")
    run_final()
    print("\nRestart the API to serve the new model.")
    return {"decision": "promoted", "scores": scores, "holdout": source, "archive": folder}


if __name__ == "__main__":
    run()
