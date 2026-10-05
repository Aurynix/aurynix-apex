"""PSI (Population Stability Index) and drift statuses.

PSI compares two distributions over the same bins:

    PSI = Σ (current_share − reference_share) × ln(current_share / reference_share)

0 means identical. Rule of thumb (config.json → monitoring):
< 0.10 ok · 0.10–0.25 warning · ≥ 0.25 drift. Empty bins get a tiny share
(EPSILON) so the log never sees 0.
"""

from typing import Any

import numpy as np
import pandas as pd

EPSILON = 1e-4


def psi(reference: list[float], current: list[float]) -> float:
    """PSI between two lists of shares over the same bins."""
    ref = np.clip(np.asarray(reference, dtype=float), EPSILON, None)
    cur = np.clip(np.asarray(current, dtype=float), EPSILON, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def bin_shares(values: pd.Series, edges: list[float]) -> list[float]:
    """Share of values in each bin, using the fixed edges from the reference profile."""
    counts = pd.cut(values, edges, include_lowest=True).value_counts(sort=False)
    return (counts / max(len(values), 1)).tolist()


def category_shares(values: pd.Series, known: list[str], other: str) -> dict[str, float]:
    """Share of each known category; anything never seen in training goes to `other`."""
    values = values.astype(str).where(values.astype(str).isin(known), other)
    shares = values.value_counts(normalize=True)
    return {k: float(shares.get(k, 0.0)) for k in [*known, other]}


def status(value: float, config: dict[str, Any]) -> str:
    """ok / warning / drift for a PSI value."""
    mon = config["monitoring"]
    if value >= mon["psi_drift"]:
        return "drift"
    return "warning" if value >= mon["psi_warning"] else "ok"


def worst(statuses: list[str]) -> str:
    """The most serious status in a list (drift > warning > ok)."""
    order = ["ok", "warning", "drift"]
    return max(statuses, key=order.index, default="ok")
