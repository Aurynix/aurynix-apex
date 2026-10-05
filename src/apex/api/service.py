"""ScoringService: all the logic behind the API.

Routers only validate the request, call the service, and return its result;
they do not know how the model works. The service:

    score → segment → explain → log prediction → response dict

It is created once at startup from the saved artifacts (`make train`); the API
never trains.
"""

import sqlite3
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from apex.api.database import log_predictions
from apex.models.explain import Explainer
from apex.models.predict import explainer_from
from apex.models.segment import assign


class ScoringService:
    def __init__(self, model: Pipeline, meta: dict[str, Any], db: sqlite3.Connection):
        self.model = model
        self.meta = meta
        self.db = db
        self.explainer: Explainer = explainer_from(model, meta)
        self.version: str = meta["model_version"]

    def score(self, rows: list[dict]) -> list[dict]:
        """Score leads (raw column names), log them, and return one result per lead."""
        leads = pd.DataFrame(rows, columns=self.meta["input_fields"])
        scores = self.model.predict_proba(leads)[:, 1].round(4)
        segments = np.char.lower(assign(scores, self.meta["segments"]["thresholds"]))
        explanations = self.explainer.explain(leads)

        log_predictions(self.db, rows, scores.tolist(), segments.tolist(), self.version)
        return [
            {
                "score": float(score),
                "segment": str(segment),
                "reasons": {
                    "up": [_text(r) for r in reasons["reasons_up"]],
                    "down": [_text(r) for r in reasons["reasons_down"]],
                },
                "model_version": self.version,
            }
            for score, segment, reasons in zip(scores, segments, explanations, strict=True)
        ]

    def info(self) -> dict[str, Any]:
        """What the API needs to say about the loaded model."""
        meta = self.meta
        return {
            "model_version": self.version,
            "trained_at": meta["trained_at"],
            "training_rows": meta["data"]["rows"],
            "input_fields": meta["input_fields"],
            "segment_thresholds": meta["segments"]["thresholds"],
            "test_metrics": meta["performance"]["test"],
        }


def _text(reason: dict) -> str:
    """A reason as one line of text, e.g. "Occupation = Working Professional"."""
    return f"{reason['feature']} = {reason['value']}"
