import sqlite3

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from apex.api.app import create_app
from apex.api.database import SCHEMA
from apex.api.schemas import Lead
from apex.api.service import ScoringService
from apex.config import load_config
from apex.models.explain import Explainer
from apex.models.train import make_model

LEAD = {
    "lead_origin": "Landing Page Submission",
    "lead_source": "Google",
    "do_not_email": False,
    "total_visits": 3,
    "time_on_website": 1500,
    "specialization": "Finance Management",
    "occupation": "Working Professional",
}


@pytest.fixture
def service():
    """A ScoringService with a small model trained on synthetic leads and an in-memory DB."""
    rng = np.random.default_rng(0)
    n = 300
    time = rng.integers(0, 2000, n)
    leads = pd.DataFrame(
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
    model = make_model().fit(leads, (time > 1000).astype(int))
    meta = {
        "model_version": "apex-test",
        "trained_at": "2026-10-06T00:00:00+00:00",
        "data": {"rows": n},
        "input_fields": load_config()["serving"]["input_fields"],
        "segments": {"thresholds": {"high": 0.75, "medium": 0.3}},
        "performance": {"test": {"pr_auc": 0.9}},
        "explainer_means": dict(enumerate(Explainer.fit(model, leads).means)),
    }
    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.execute(SCHEMA)
    return ScoringService(model, meta, db)


@pytest.fixture
def client(service):
    with TestClient(create_app(service)) as client:
        yield client


def logged(service) -> int:
    return service.db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": "apex-test"}


def test_model_info(client):
    info = client.get("/model/info").json()
    assert info["model_version"] == "apex-test"
    assert info["segment_thresholds"] == {"high": 0.75, "medium": 0.3}
    assert len(info["input_fields"]) == 7


def test_predict_single_matches_contract_and_is_logged(client, service):
    response = client.post("/predict/single", json=LEAD)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"score", "segment", "reasons", "model_version"}
    assert 0 <= body["score"] <= 1
    assert body["segment"] in {"high", "medium", "low"}
    assert set(body["reasons"]) == {"up", "down"}
    assert all(isinstance(r, str) and " = " in r for r in body["reasons"]["up"])
    assert logged(service) == 1


def test_predict_batch_keeps_order_and_logs_every_lead(client, service):
    slow = {**LEAD, "time_on_website": 30, "total_visits": 1}
    body = client.post("/predict/batch", json={"leads": [LEAD, slow, LEAD]}).json()
    assert body["count"] == 3
    scores = [p["score"] for p in body["predictions"]]
    assert scores[0] == scores[2] > scores[1]
    assert logged(service) == 3


def test_only_lead_origin_is_required(client):
    assert client.post("/predict/single", json={"lead_origin": "API"}).status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {**LEAD, "total_visits": -1},
        {**LEAD, "unknown_field": "x"},
        {k: v for k, v in LEAD.items() if k != "lead_origin"},
    ],
)
def test_invalid_leads_are_rejected(client, payload):
    assert client.post("/predict/single", json=payload).status_code == 422


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json={"leads": []}).status_code == 422


def test_lead_maps_to_the_model_input_fields():
    row = Lead(**LEAD).to_row()
    assert list(row) == load_config()["serving"]["input_fields"]
    assert row["Do Not Email"] == "No"


def test_no_time_on_site_means_zero_visits():
    assert Lead(lead_origin="Lead Add Form").to_row()["TotalVisits"] == 0
    assert Lead(lead_origin="API", time_on_website=600).to_row()["TotalVisits"] is None
