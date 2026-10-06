"""Map probabilities to High / Medium / Low using sales capacity (step 3.3).

Segments are cut by **share of leads**, not by fixed probabilities: if the team
can call 20% of incoming leads, the top 20% of scores are High. The shares
come from `config.json → segmentation`; the score thresholds are derived once
from out-of-fold scores (scores from models that did not see the lead) and
then stay fixed, so a new lead is segmented by comparing its score with them.

Run `python -m apex.models.segment` (or `make segments`) to derive the
thresholds and print the segment tables.
"""

from typing import Any

import numpy as np
import pandas as pd

from apex.config import load_config

SEGMENTS = ["High", "Medium", "Low"]


def thresholds(scores, config: dict[str, Any] | None = None) -> dict[str, float]:
    """Score cut-offs: the top `high_share` of scores are High, the next `medium_share` Medium."""
    seg = (config or load_config())["segmentation"]
    scores = np.asarray(scores)
    return {
        "high": float(np.quantile(scores, 1 - seg["high_share"])),
        "medium": float(np.quantile(scores, 1 - seg["high_share"] - seg["medium_share"])),
    }


def assign(scores, cutoffs: dict[str, float]) -> np.ndarray:
    """High / Medium / Low for each score."""
    scores = np.asarray(scores)
    return np.select(
        [scores >= cutoffs["high"], scores >= cutoffs["medium"]], ["High", "Medium"], "Low"
    )


def segment_table(y_true, segments) -> pd.DataFrame:
    """Per segment: leads, share of leads, conversion rate, and share of all conversions."""
    df = pd.DataFrame({"segment": segments, "converted": np.asarray(y_true)})
    out = df.groupby("segment")["converted"].agg(leads="size", conversions="sum", rate="mean")
    out = out.reindex(SEGMENTS)
    out.insert(1, "share_of_leads", out["leads"] / out["leads"].sum())
    out["share_of_conversions"] = out["conversions"] / out["conversions"].sum()
    return out


def run() -> dict[str, float]:
    """Derive thresholds from out-of-fold scores and print the segment tables."""
    from apex.data.split import load_splits
    from apex.models.train import make_model, out_of_fold_scores

    cfg = load_config()
    target = cfg["data"]["target"]
    parts = load_splits(include_test=True)  # test: report only, thresholds are fixed first
    data = pd.concat([parts["train"], parts["val"]])
    X, y = data.drop(columns=[target]), data[target]

    oof = out_of_fold_scores(make_model(cfg), X, y, cfg)
    cutoffs = thresholds(oof, cfg)
    print(f"Thresholds: High ≥ {cutoffs['high']:.3f}, Medium ≥ {cutoffs['medium']:.3f}\n")
    print(f"Out-of-fold, train + validation ({len(X):,} rows)")
    print(segment_table(y, assign(oof, cutoffs)).round(3).to_string(), "\n")

    test = parts["test"]
    model = make_model(cfg).fit(X, y)  # the step 3.1 model
    test_scores = model.predict_proba(test.drop(columns=[target]))[:, 1]
    print(f"Test, step 3.1 model ({len(test):,} rows)")
    print(segment_table(test[target], assign(test_scores, cutoffs)).round(3).to_string())
    return cutoffs


if __name__ == "__main__":
    run()
