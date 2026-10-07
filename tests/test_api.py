import pytest
from fastapi.testclient import TestClient

from apex.api.app import create_app
from apex.api.schemas import Lead
from apex.config import load_config

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
def client(service, monitor):
    with TestClient(create_app(service, monitor)) as client:
        yield client


def logged(service) -> int:
    return service.db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": "apex-test"}


def test_model_info(client, service):
    info = client.get("/model/info").json()
    assert info["model_version"] == "apex-test"
    assert info["segment_thresholds"] == service.meta["segments"]["thresholds"]
    assert len(info["input_fields"]) == 7


def test_predict_single_matches_contract_and_is_logged(client, service):
    response = client.post("/predict/single", json=LEAD)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction_id", "score", "segment", "reasons", "model_version"}
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


def test_invalid_requests_are_logged_for_monitoring(client, service):
    client.post("/predict/single", json={**LEAD, "total_visits": -1})
    assert service.db.execute("SELECT COUNT(*) FROM rejected_requests").fetchone()[0] == 1


def test_monitoring_endpoints(client):
    assert client.get("/monitoring/latest").status_code == 404
    client.post("/predict/batch", json={"leads": [LEAD] * 10})
    report = client.post("/monitoring/run").json()
    assert report["status"] == "insufficient_data"  # 10 < 50
    assert client.get("/monitoring/latest").json()["id"] == report["id"]
    assert client.get("/monitoring/history").json()[0]["status"] == "insufficient_data"


def test_outcomes_are_saved_by_prediction_id(client, service):
    first, second = client.post("/predict/batch", json={"leads": [LEAD, LEAD]}).json()[
        "predictions"
    ]
    assert second["prediction_id"] == first["prediction_id"] + 1
    payload = {"outcomes": [{"prediction_id": first["prediction_id"], "converted": True}]}
    assert client.post("/outcomes", json=payload).json() == {"saved": 1}
    payload["outcomes"][0]["converted"] = False  # sending again replaces the outcome
    client.post("/outcomes", json=payload)
    assert service.db.execute("SELECT converted FROM outcomes").fetchall() == [(0,)]


def test_unknown_prediction_ids_save_nothing(client, service):
    pid = client.post("/predict/single", json=LEAD).json()["prediction_id"]
    payload = {
        "outcomes": [
            {"prediction_id": pid, "converted": True},
            {"prediction_id": 999, "converted": True},
        ]
    }
    response = client.post("/outcomes", json=payload)
    assert response.status_code == 404
    assert response.json()["detail"]["unknown"] == [999]
    assert service.db.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0] == 0


def test_api_reports_the_software_version(client):
    from importlib.metadata import version

    assert client.get("/openapi.json").json()["info"]["version"] == version("aurynix-apex")
