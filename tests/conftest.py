"""Shared fixtures: a small model trained on synthetic leads, its metadata and
reference profile, and in-memory databases. No saved artifacts are needed."""

import copy
import sqlite3

import numpy as np
import pandas as pd
import pytest

from apex.api.database import init_db
from apex.api.service import ScoringService
from apex.config import load_config
from apex.models.explain import Explainer
from apex.models.train import make_model
from apex.monitoring.monitor import Monitor
from apex.monitoring.reference import build_profile


def synthetic_leads(n: int, seed: int = 0) -> pd.DataFrame:
    """Raw leads with the 7 input fields; conversion depends on time on site."""
    rng = np.random.default_rng(seed)
    time = rng.integers(0, 2000, n)
    return pd.DataFrame(
        {
            "Lead Origin": rng.choice(["API", "Landing Page Submission", "Lead Add Form"], n),
            "Lead Source": rng.choice(["Google", "Direct Traffic"], n),
            "Do Not Email": rng.choice(["Yes", "No"], n),
            "TotalVisits": np.where(time > 0, 3.0, 0.0),
            "Total Time Spent on Website": time,
            "Specialization": rng.choice(["Finance Management", None], n),
            "What is your current occupation": rng.choice(["Working Professional", None], n),
        }
    )


def as_rows(leads: pd.DataFrame) -> list[dict]:
    return leads.astype(object).where(leads.notna(), None).to_dict(orient="records")


@pytest.fixture(scope="session")
def trained():
    """(model, meta, profile) fitted on 600 synthetic leads."""
    cfg = load_config()
    leads = synthetic_leads(600)
    y = (leads["Total Time Spent on Website"] > 1000).astype(int)
    model = make_model(cfg).fit(leads, y)
    scores = model.predict_proba(leads)[:, 1]
    thresholds = {
        "high": float(np.quantile(scores, 0.8)),
        "medium": float(np.quantile(scores, 0.5)),
    }
    segments = np.select(
        [scores >= thresholds["high"], scores >= thresholds["medium"]], ["High", "Medium"], "Low"
    )
    meta = {
        "model_version": "apex-test",
        "trained_at": "2026-10-06T00:00:00+00:00",
        "data": {"rows": len(leads)},
        "input_fields": cfg["serving"]["input_fields"],
        "segments": {"thresholds": thresholds},
        "performance": {"test": {"pr_auc": 0.9}},
        "explainer_means": dict(enumerate(Explainer.fit(model, leads).means)),
    }
    profile = build_profile(leads, model[0][:-1].transform(leads), scores, segments, cfg)
    return model, meta, profile


@pytest.fixture
def db():
    conn = init_db(sqlite3.connect(":memory:", check_same_thread=False))
    yield conn
    conn.close()


@pytest.fixture
def service(trained, db):
    model, meta, _ = trained
    return ScoringService(model, meta, db)


@pytest.fixture
def monitor(trained, db, tmp_path, monkeypatch):
    """Monitor with min_samples = 50; reports are written to a temp folder."""
    from apex.monitoring import monitor as monitor_module

    model, _, profile = trained
    cfg = copy.deepcopy(load_config())
    cfg["monitoring"]["min_samples"] = 50
    monkeypatch.setattr(monitor_module, "path", lambda key: tmp_path)
    return Monitor(model, profile, db, cfg)
