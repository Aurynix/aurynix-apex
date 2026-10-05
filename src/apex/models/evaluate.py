"""Metrics for a ranked list of leads (see docs/problem_framing.md section 5).

- PR-AUC (primary), ROC-AUC, Brier score: standard model quality.
- Capture and lift at the top k%: the business view. If the sales team only
  calls the top 20% of leads, what share of all buyers do they reach?
"""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def capture_at(y_true, y_score, share: float) -> float:
    """Share of all conversions found in the top `share` of leads ranked by score."""
    y_true = np.asarray(y_true)
    top = np.argsort(-np.asarray(y_score), kind="stable")[: int(round(share * len(y_true)))]
    return float(y_true[top].sum() / y_true.sum())


def evaluate(y_true, y_score, shares: tuple[float, ...] = (0.2, 0.5)) -> dict[str, float]:
    """All metrics for one set of predictions, as a flat dict (ready for MLflow)."""
    metrics = {
        "pr_auc": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "brier": brier_score_loss(y_true, y_score),
    }
    for share in shares:
        pct = round(share * 100)
        captured = capture_at(y_true, y_score, share)
        metrics[f"capture_top{pct}"] = captured
        metrics[f"lift_top{pct}"] = captured / share
    return {name: round(float(value), 4) for name, value in metrics.items()}
