"""Score leads with the saved model (offline batch scoring, step 3.5).

    python -m apex.models.predict [input.csv] [output.csv]   (or `make predict`)

Defaults: score data/raw/Leads.csv and write data/predictions.csv with, per
lead, the probability, the High / Medium / Low segment, and the strongest
reason up and down. The API (step 4) uses the same `load_model` / `predict`.
"""

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from apex.config import load_config, path
from apex.data.load import load_raw
from apex.models.explain import Explainer, labels
from apex.models.segment import assign


def load_model() -> tuple[Pipeline, dict[str, Any]]:
    """The saved pipeline and its metadata (run `make train` first)."""
    files, out = load_config()["files"], path("models_dir")
    if not (out / files["model"]).exists():
        raise FileNotFoundError(f"{out / files['model']} not found. Run `make train` first.")
    with (out / files["model_meta"]).open(encoding="utf-8") as f:
        meta = json.load(f)
    return joblib.load(out / files["model"]), meta


def explainer_from(model: Pipeline, meta: dict[str, Any]) -> Explainer:
    """Rebuild the explainer from the saved background means."""
    return Explainer(model, np.array(list(meta["explainer_means"].values())))


def predict(leads: pd.DataFrame, model: Pipeline, meta: dict[str, Any]) -> pd.DataFrame:
    """Probability, segment, and the strongest reason up / down for each lead."""
    scores = model.predict_proba(leads)[:, 1]
    impact = explainer_from(model, meta).contributions(leads).rename(columns=labels())
    return pd.DataFrame(
        {
            "probability": scores.round(4),
            "segment": assign(scores, meta["segments"]["thresholds"]),
            "top_reason_up": impact.idxmax(axis=1).where(impact.max(axis=1) > 0),
            "top_reason_down": impact.idxmin(axis=1).where(impact.min(axis=1) < 0),
        },
        index=leads.index,
    )


def run(source: str | None = None, target: str | None = None) -> pd.DataFrame:
    """Score a CSV of leads (default: the raw Kaggle file) and save the predictions."""
    cfg = load_config()
    leads = pd.read_csv(source) if source else load_raw()
    model, meta = load_model()

    id_col = cfg["data"]["id_columns"][0]
    result = predict(leads, model, meta)
    if id_col in leads.columns:
        result.insert(0, id_col, leads[id_col])

    out = Path(target) if target else path("raw_dir").parent / cfg["files"]["predictions"]
    result.to_csv(out, index=False)
    print(f"Scored {len(result):,} leads with {meta['model_version']}")
    print(result["segment"].value_counts().reindex(["High", "Medium", "Low"]).to_string())
    print(f"\nsaved: {out}")
    return result


if __name__ == "__main__":
    run(*sys.argv[1:3])
