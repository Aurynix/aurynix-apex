"""Simulate production traffic to check that drift monitoring works (step 4.4).

    python -m apex.monitoring.simulate   (or `make drift-demo`)

Two scenarios, each in its own fresh database (data/simulation.db), so the real
prediction log is never touched:

- stable:  leads sampled from Leads.csv, i.e. the same distribution as training
- drifted: the same leads after a "new campaign" changes the traffic:
    · 30% now come from a new source ("TikTok") through landing pages
    · leads spend about half as long on the website
    · 50% more leads skip the occupation question

Leads are scored by the real ScoringService (so they are logged like API traffic),
then the Monitor checks them. Expected: stable → ok, drifted → drift.

    python -m apex.monitoring.simulate outcomes   (or `make outcomes-demo`)

Performance check with real results: leads sampled with their real `Converted`
value are scored, and the true outcomes are sent back. Then "behavior changed":
half of the outcomes are shuffled, so the score no longer matches who buys.
Expected: real outcomes → ok, changed behavior → drift (retraining recommended).
Note: the final model was trained on all of Leads.csv, so the first scenario is
in-sample and slightly optimistic; it shows the mechanics, not new evidence.
"""

import sys

import numpy as np
import pandas as pd

from apex.api.database import connect
from apex.api.service import ScoringService
from apex.config import load_config, path
from apex.data.load import load_raw
from apex.models.predict import load_model
from apex.monitoring.monitor import create_monitor, summary


def stable_leads(n: int, seed: int = 0) -> pd.DataFrame:
    """`n` leads sampled (with replacement) from the raw file: the 7 input fields only."""
    fields = load_config()["serving"]["input_fields"]
    return load_raw()[fields].sample(n, replace=True, random_state=seed).reset_index(drop=True)


def drift(leads: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Apply the "new campaign" shift described in the module docstring."""
    rng = np.random.default_rng(seed)
    leads = leads.copy()
    n = len(leads)

    tiktok = rng.random(n) < 0.3
    leads.loc[tiktok, "Lead Source"] = "TikTok"
    leads.loc[tiktok, "Lead Origin"] = "Landing Page Submission"

    leads["Total Time Spent on Website"] = (leads["Total Time Spent on Website"] * 0.5).round()

    skip = rng.random(n) < 0.5
    leads.loc[skip, "What is your current occupation"] = None
    return leads


def score_and_monitor(leads: pd.DataFrame, converted: pd.Series | None = None) -> dict:
    """Score the leads in a fresh simulation database (optionally send their real
    outcomes), then run the monitor on them."""
    db_file = path("raw_dir").parent / "simulation.db"
    db_file.unlink(missing_ok=True)
    db = connect(db_file)

    model, meta = load_model()
    rows = leads.astype(object).where(leads.notna(), None).to_dict(orient="records")
    service = ScoringService(model, meta, db)
    ids = [p["prediction_id"] for p in service.score(rows)]
    if converted is not None:
        service.record_outcomes(dict(zip(ids, map(bool, converted), strict=True)))
    return create_monitor(db, model, meta).run(save=False)


def run(n: int = 2000) -> dict[str, dict]:
    base = stable_leads(n)
    reports = {}
    for name, leads in {"stable": base, "drifted": drift(base)}.items():
        reports[name] = score_and_monitor(leads)
        print(f"===== {name} traffic ({n:,} leads) =====")
        print(summary(reports[name]), "\n")
    return reports


def run_outcomes(n: int = 2000, seed: int = 0) -> dict[str, dict]:
    """Real outcomes vs. changed behavior: the performance check."""
    cfg = load_config()
    sample = load_raw().sample(n, replace=True, random_state=seed).reset_index(drop=True)
    leads, real = sample[cfg["serving"]["input_fields"]], sample[cfg["data"]["target"]]

    changed = real.copy()
    rng = np.random.default_rng(seed)
    half = rng.random(n) < 0.5
    changed[half] = rng.permutation(changed[half].to_numpy())

    reports = {}
    for name, converted in {"real outcomes": real, "behavior changed": changed}.items():
        reports[name] = score_and_monitor(leads, converted)
        print(f"===== {name} ({n:,} leads with outcomes) =====")
        print(summary(reports[name]), "\n")
    return reports


if __name__ == "__main__":
    run_outcomes() if sys.argv[1:] == ["outcomes"] else run()
