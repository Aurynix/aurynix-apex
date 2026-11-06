"""Generate the synthetic leads of the public demo (deploy/demo_seed.json).

    python -m apex.monitoring.synthetic   (or `make demo-seed-file`)

No real rows are used or copied: the only inputs are the saved model and its reference
profile in deploy/models/ (shares and bin edges of the training data), and the seed is
fixed, so anyone can regenerate the same file without the original dataset.

1. Build a large pool of leads, drawing each input field from its training distribution:
   categories by their shares, numbers from their quantile bins. Visits and time on
   site are both 0 for leads with no web activity; otherwise fields are independent.
2. Select `SEED_ROWS` leads from the pool so their scores follow the training score
   distribution. Independent fields lose the links between them (a Lead Add Form is
   usually a Reference with no visits), which alone shifts the scores (PSI ≈ 0.2) and
   would make the demo's report warn about drift for reasons unrelated to monitoring.
3. Draw each outcome from the model's predicted probability for that lead.

By design, these leads look like training data and agree with the model, so a report
on them shows how monitoring works, and says nothing about how good the model is.
"""

import json
import math
from typing import Any

import joblib
import numpy as np
import pandas as pd

from apex.config import load_config, path

SEED = 20261106
SEED_ROWS = 600
POOL_ROWS = 20_000
TOP_BIN_STRETCH = 1.25  # the top bin has no upper edge: end it this far above its lower edge
MISSING = "Missing"


def draw_category(rng: np.random.Generator, shares: dict[str, float], n: int) -> list[str | None]:
    """`n` values by their shares; the cleaned label "Missing" becomes an empty field."""
    names = list(shares)
    weights = np.array([shares[name] for name in names])
    return [
        None if value == MISSING else str(value)
        for value in rng.choice(names, n, p=weights / weights.sum())
    ]


def draw_number(rng: np.random.Generator, profile: dict[str, list[float]], n: int) -> list[int]:
    """`n` positive whole numbers from the quantile bins, without the first bin (zero)."""
    edges, shares = profile["edges"], np.array(profile["shares"][1:])
    bins = rng.choice(len(shares), n, p=shares / shares.sum()) + 1
    values = []
    for b in bins:
        low, high = edges[b], edges[b + 1]
        if math.isinf(high):
            high = math.ceil(low * TOP_BIN_STRETCH)
        values.append(int(rng.integers(math.floor(low) + 1, math.floor(high) + 1)))
    return values


def draw_pool(rng: np.random.Generator, profile: dict[str, Any], n: int) -> pd.DataFrame:
    """Leads with the 7 raw input fields, each from its training distribution."""
    categories, numbers = profile["categorical"], profile["numeric"]
    inactive = rng.random(n) < numbers["TotalVisits"]["shares"][0]
    opted_out = rng.random(n) < numbers["Do Not Email"]["shares"][1]
    visits = draw_number(rng, numbers["TotalVisits"], n)
    time_on_site = draw_number(rng, numbers["Total Time Spent on Website"], n)
    return pd.DataFrame(
        {
            "Lead Origin": draw_category(rng, categories["Lead Origin"], n),
            "Lead Source": draw_category(rng, categories["Lead Source"], n),
            "Do Not Email": ["Yes" if value else "No" for value in opted_out],
            "TotalVisits": [0 if none else v for none, v in zip(inactive, visits, strict=True)],
            "Total Time Spent on Website": [
                0 if none else t for none, t in zip(inactive, time_on_site, strict=True)
            ],
            "Specialization": draw_category(rng, categories["Specialization"], n),
            "What is your current occupation": draw_category(
                rng, categories["What is your current occupation"], n
            ),
        }
    )


def select_by_score(
    rng: np.random.Generator, scores: np.ndarray, profile: dict[str, list[float]], n: int
) -> np.ndarray:
    """Positions of `n` pool rows whose scores follow the training score distribution."""
    bins = np.digitize(scores, profile["edges"][1:-1], right=True)
    shares = np.array(profile["shares"])
    wanted = shares / shares.sum() * n
    counts = np.floor(wanted).astype(int)
    for b in np.argsort(counts - wanted)[: n - counts.sum()]:
        counts[b] += 1
    chosen = [np.flatnonzero(bins == b)[:k] for b, k in enumerate(counts)]
    return rng.permutation(np.concatenate(chosen))


def generate(
    model, meta: dict[str, Any], profile: dict[str, Any], n: int = SEED_ROWS, seed: int = SEED
) -> list[dict[str, Any]]:
    """Synthetic rows: [{"lead": {7 input fields}, "converted": 0 or 1}, ...]."""
    rng = np.random.default_rng(seed)
    pool = draw_pool(rng, profile, POOL_ROWS)[meta["input_fields"]]
    positions = select_by_score(rng, model.predict_proba(pool)[:, 1], profile["score"], n)
    leads = pool.iloc[positions]
    probabilities = model.predict_proba(leads)[:, 1]
    converted = rng.random(n) < probabilities
    rows = leads.astype(object).where(leads.notna(), None).to_dict(orient="records")
    return [
        {"lead": lead, "converted": int(result)}
        for lead, result in zip(rows, converted, strict=True)
    ]


def load_artifacts() -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """The committed model, metadata, and reference profile in deploy/models/."""
    files, folder = load_config()["files"], path("deploy_models")
    meta = json.loads((folder / files["model_meta"]).read_text(encoding="utf-8"))
    profile = json.loads((folder / files["reference_profile"]).read_text(encoding="utf-8"))
    return joblib.load(folder / files["model"]), meta, profile


def to_json(rows: list[dict[str, Any]]) -> str:
    return json.dumps(rows, indent=0) + "\n"


def run() -> list[dict[str, Any]]:
    """Generate the rows and save them to deploy/demo_seed.json."""
    rows = generate(*load_artifacts())
    file = path("demo_seed")
    file.write_text(to_json(rows), encoding="utf-8")
    print(f"saved {len(rows)} rows ({sum(r['converted'] for r in rows)} converted): {file}")
    return rows


if __name__ == "__main__":
    run()
