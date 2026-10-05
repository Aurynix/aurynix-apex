"""Metrics for a ranked list of leads (see docs/problem_framing.md section 5).

- PR-AUC (primary), ROC-AUC: ranking quality.
- Brier score, expected calibration error (ECE): are the probabilities honest?
- Precision and recall at the top k%: the business view. If the sales team
  only calls the top 20% of leads, how many of those calls are buyers
  (precision), and what share of all buyers do they reach (recall)?
- Precision and recall at probability >= 0.5: the standard classifier view.
"""

import numpy as np
import pandas as pd
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


def calibration_table(y_true, y_score, n_bins: int = 10) -> pd.DataFrame:
    """Leads, mean predicted probability, and actual conversion rate per 0.1-wide bin."""
    y_score = np.asarray(y_score)
    bins = np.minimum((y_score * n_bins).astype(int), n_bins - 1)
    df = pd.DataFrame({"bin": bins / n_bins, "actual": np.asarray(y_true), "predicted": y_score})
    return df.groupby("bin").agg(
        leads=("actual", "size"), predicted=("predicted", "mean"), actual=("actual", "mean")
    )


def expected_calibration_error(y_true, y_score, n_bins: int = 10) -> float:
    """Average gap between predicted and actual rate, weighted by leads per bin."""
    table = calibration_table(y_true, y_score, n_bins)
    weights = table["leads"] / table["leads"].sum()
    return float((weights * (table["predicted"] - table["actual"]).abs()).sum())


def evaluate(y_true, y_score, shares: tuple[float, ...] = (0.2, 0.5)) -> dict[str, float]:
    """All metrics for one set of predictions, as a flat dict (ready for MLflow)."""
    at_half = (np.asarray(y_score) >= 0.5).astype(int)
    metrics = {
        "pr_auc": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "brier": brier_score_loss(y_true, y_score),
        "ece": expected_calibration_error(y_true, y_score),
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


def plot_test_report(y_true, y_score, file, title: str) -> None:
    """PR curve, ROC curve, cumulative gain and lift in one figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    blue, ink, muted, grid, surface = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    y_true = np.asarray(y_true)
    rate = y_true.mean()

    order = np.argsort(-np.asarray(y_score), kind="stable")
    share = np.arange(1, len(y_true) + 1) / len(y_true)
    gain = np.cumsum(y_true[order]) / y_true.sum()

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    panels = [
        ("Precision–recall curve", recall, precision, rate, "Recall", "Precision"),
        ("ROC curve", fpr, tpr, None, "False positive rate", "True positive rate"),
        ("Cumulative gain", share, gain, None, "Share of leads called", "Share of buyers found"),
        ("Lift", share, gain / share, 1.0, "Share of leads called", "Lift vs. random"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), facecolor=surface)
    for ax, (name, x, y, flat, xlabel, ylabel) in zip(axes, panels, strict=True):
        ax.plot(x, y, color=blue, linewidth=2)
        if flat is not None:
            ax.axhline(flat, color=muted, linestyle="--", linewidth=1)
        else:
            ax.plot([0, 1], [0, 1], color=muted, linestyle="--", linewidth=1)
        ax.set_title(name, loc="left", fontsize=10, color=ink, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=8, color=muted)
        ax.set_ylabel(ylabel, fontsize=8, color=muted)
        ax.set_facecolor(surface)
        ax.tick_params(colors=muted, labelsize=8, length=0)
        ax.grid(color=grid, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    gain_ax, lift_ax = axes[2], axes[3]
    for cut in (0.2, 0.5):
        i = int(round(cut * len(y_true))) - 1
        gain_ax.plot(share[i], gain[i], "o", color=blue, markersize=8)
        gain_ax.annotate(
            f"top {cut:.0%} → {gain[i]:.0%} of buyers",
            (share[i], gain[i]),
            xytext=(8, -14),
            textcoords="offset points",
            fontsize=8,
            color=ink,
        )
    lift_ax.set_ylim(0, 1 / rate + 0.3)
    fig.suptitle(title, x=0.01, ha="left", color=ink)
    fig.tight_layout()
    fig.savefig(file, dpi=120)
    plt.close(fig)


def plot_calibration(curves: dict[str, pd.DataFrame], file, title: str) -> None:
    """Reliability diagram: predicted probability vs. actual conversion rate per bin."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(6.5, 6), facecolor=surface)
    ax.plot([0, 1], [0, 1], color=muted, linestyle="--", linewidth=1, label="Perfect")
    for (name, table), color in zip(curves.items(), colors, strict=False):
        ax.plot(table["predicted"], table["actual"], "o-", color=color, linewidth=2, label=name)
    ax.set_xlabel("Predicted probability", fontsize=9, color=muted)
    ax.set_ylabel("Actual conversion rate", fontsize=9, color=muted)
    ax.set_title(title, loc="left", fontsize=10, color=ink, fontweight="bold")
    ax.set_facecolor(surface)
    ax.grid(color=grid, linewidth=0.6)
    ax.tick_params(colors=muted, labelsize=8, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, labelcolor=ink)
    fig.tight_layout()
    fig.savefig(file, dpi=120)
    plt.close(fig)
