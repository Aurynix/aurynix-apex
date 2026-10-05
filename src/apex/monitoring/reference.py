"""Build the reference profile at training time (step 3.5).

The profile records what the training data looked like, so drift monitoring
(step 4.4) can compare new leads against it with PSI:

- numeric inputs: fixed bin edges (training quantiles) and the share per bin
- categorical inputs: the share of each known category
- missing rate of each raw input field (before cleaning)
- score distribution (same bins idea) and High / Medium / Low shares

Bin edges are fixed here and reused for every later comparison.
"""

from typing import Any

import numpy as np
import pandas as pd


def numeric_profile(values: pd.Series, n_bins: int) -> dict[str, list[float]]:
    """Quantile bin edges (outer edges open) and the share of values in each bin."""
    inner = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1]))
    edges = [-np.inf, *inner, np.inf]
    shares = pd.cut(values, edges, include_lowest=True).value_counts(normalize=True, sort=False)
    return {"edges": [float(e) for e in edges], "shares": [round(float(s), 6) for s in shares]}


def categorical_profile(values: pd.Series) -> dict[str, float]:
    """Share of each category."""
    shares = values.astype(str).value_counts(normalize=True)
    return {k: round(float(v), 6) for k, v in shares.items()}


def build_profile(
    raw: pd.DataFrame,
    model_inputs: pd.DataFrame,
    scores,
    segments,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Profile of the training data.

    `raw`: raw input fields (for missing rates); `model_inputs`: the same leads after
    cleaning + feature engineering (what the model sees); `scores` / `segments`:
    out-of-fold scores and their segments.
    """
    n_bins = config["monitoring"]["n_bins"]
    numeric = model_inputs.select_dtypes("number").columns
    categorical = model_inputs.select_dtypes(exclude="number").columns
    raw_fields = [c for c in config["serving"]["input_fields"] if c in raw.columns]
    hidden = config["data"]["missing_placeholders"]
    return {
        "model_version": config["project"]["model_version"],
        "rows": len(raw),
        "numeric": {c: numeric_profile(model_inputs[c], n_bins) for c in numeric},
        "categorical": {c: categorical_profile(model_inputs[c]) for c in categorical},
        "missing_rate": {
            c: round(float((raw[c].isna() | raw[c].isin(hidden)).mean()), 6) for c in raw_fields
        },
        "score": numeric_profile(pd.Series(scores), n_bins),
        "segment_shares": categorical_profile(pd.Series(segments)),
    }
