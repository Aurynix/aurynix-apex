"""Metrics for a ranked list of leads (see docs/problem_framing.md section 5).

- PR-AUC (primary), ROC-AUC, Brier score: standard model quality.
- Precision and recall at the top k%: the business view. If the sales team
  only calls the top 20% of leads, how many of those calls are buyers
  (precision), and what share of all buyers do they reach (recall)?
- Precision and recall at probability >= 0.5: the standard classifier view.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def top_share(y_score, share: float) -> np.ndarray:
    """1 for the `share` of leads with the highest scores, 0 for the rest."""
    y_score = np.asarray(y_score)
    top = np.argsort(-y_score, kind="stable")[: int(round(share * len(y_score)))]
    called = np.zeros(len(y_score), dtype=int)
    called[top] = 1
    return called


def evaluate(y_true, y_score, shares: tuple[float, ...] = (0.2, 0.5)) -> dict[str, float]:
    """All metrics for one set of predictions, as a flat dict (ready for MLflow)."""
    at_half = (np.asarray(y_score) >= 0.5).astype(int)
    metrics = {
        "pr_auc": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "brier": brier_score_loss(y_true, y_score),
        "precision_at_0.5": precision_score(y_true, at_half, zero_division=0),
        "recall_at_0.5": recall_score(y_true, at_half),
    }
    for share in shares:
        pct = round(share * 100)
        called = top_share(y_score, share)
        recall = recall_score(y_true, called)
        metrics[f"precision_top{pct}"] = precision_score(y_true, called, zero_division=0)
        metrics[f"recall_top{pct}"] = recall
        metrics[f"lift_top{pct}"] = recall / share
    return {name: round(float(value), 4) for name, value in metrics.items()}
