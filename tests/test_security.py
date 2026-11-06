from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from apex.api import app as app_module
from apex.api.app import create_app
from apex.api.database import DEMO_SOURCE, log_rejection, purge_visitor_data, save_drift_run
from apex.api.security import RateLimiter
from tests.conftest import as_rows, synthetic_leads

LEAD = {"lead_origin": "API", "total_visits": 1, "time_on_website": 30}
OUTCOMES = {"outcomes": [{"prediction_id": 1, "converted": True}]}
KEY = {"X-API-Key": "right-key"}


@pytest.fixture
def make_client(service, monitor, monkeypatch):
    """A client for an app configured by the given environment variables."""
    clients = []

    def build(**env):
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        client = TestClient(create_app(service, monitor))
        client.__enter__()
        clients.append(client)
        return client

    yield build
    for client in clients:
        client.__exit__(None, None, None)


# ---------- API key ----------
@pytest.mark.parametrize(("path", "body"), [("/outcomes", OUTCOMES), ("/monitoring/run", None)])
def test_protected_endpoints_need_the_right_key(make_client, path, body):
    client = make_client(API_KEY="right-key")
    assert client.post(path, json=body).status_code == 401
    assert client.post(path, json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post(path, json=body, headers=KEY).status_code != 401


def test_fake_outcomes_are_not_saved_without_the_key(make_client, service):
    client = make_client(API_KEY="right-key")
    prediction_id = client.post("/predict/single", json=LEAD).json()["prediction_id"]
    body = {"outcomes": [{"prediction_id": prediction_id, "converted": True}]}
    assert client.post("/outcomes", json=body).status_code == 401
    assert service.db.scalar("SELECT COUNT(*) FROM outcomes") == 0
    assert client.post("/outcomes", json=body, headers=KEY).json() == {"saved": 1}


def test_scoring_and_reading_stay_open_with_a_key_configured(make_client):
    client = make_client(API_KEY="right-key")
    assert client.post("/predict/single", json=LEAD).status_code == 200
    assert client.post("/predict/batch", json={"leads": [LEAD]}).status_code == 200
    assert client.get("/monitoring/history").status_code == 200
    assert client.get("/model/info").status_code == 200


def test_a_rejected_key_is_checked_before_the_body_and_logs_nothing(make_client, service):
    client = make_client(API_KEY="right-key")
    response = client.post(
        "/outcomes", content=b"not json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 401
    assert service.db.scalar("SELECT COUNT(*) FROM rejected_requests") == 0


def test_a_persistent_database_without_a_key_refuses_protected_requests(make_client, service):
    client = make_client(DATABASE_URL="postgresql://u:p@unreachable.invalid/db")
    assert client.post("/outcomes", json=OUTCOMES).status_code == 503
    assert client.post("/monitoring/run").status_code == 503
    assert client.post("/outcomes", json=OUTCOMES, headers=KEY).status_code == 503
    assert service.db.scalar("SELECT COUNT(*) FROM drift_runs") == 0


def test_without_any_settings_local_use_stays_open(make_client):
    client = make_client()
    assert client.post("/monitoring/run").status_code == 200


def test_the_key_is_shown_in_the_api_docs(make_client):
    schema = make_client().get("/openapi.json").json()
    assert schema["components"]["securitySchemes"]["APIKeyHeader"]["name"] == "X-API-Key"
    assert schema["paths"]["/outcomes"]["post"]["security"] == [{"APIKeyHeader": []}]
    assert "security" not in schema["paths"]["/predict/single"]["post"]


# ---------- Rate limit ----------
def test_rate_limit_answers_429_with_retry_after(make_client):
    client = make_client(RATE_LIMIT_PER_MINUTE="2")
    for _ in range(2):
        assert client.post("/predict/single", json=LEAD).status_code == 200
    response = client.post("/predict/single", json=LEAD)
    assert response.status_code == 429
    assert 1 <= int(response.headers["Retry-After"]) <= 60


def test_rate_limit_stops_invalid_requests_being_logged(make_client, service):
    client = make_client(RATE_LIMIT_PER_MINUTE="1")
    for _ in range(3):
        client.post("/predict/single", json={"total_visits": -1})
    assert service.db.scalar("SELECT COUNT(*) FROM rejected_requests") == 1


def test_rate_limit_is_per_client_behind_cloudflare(make_client):
    client = make_client(RATE_LIMIT_PER_MINUTE="1", TRUST_CF_CONNECTING_IP="true")
    first = {"CF-Connecting-IP": "203.0.113.1"}
    second = {"CF-Connecting-IP": "203.0.113.2"}
    assert client.post("/predict/single", json=LEAD, headers=first).status_code == 200
    assert client.post("/predict/single", json=LEAD, headers=first).status_code == 429
    assert client.post("/predict/single", json=LEAD, headers=second).status_code == 200


@pytest.mark.parametrize("header", ["CF-Connecting-IP", "X-Forwarded-For", "X-Real-IP"])
def test_address_headers_cannot_dodge_the_limit_when_not_trusted(make_client, header):
    client = make_client(RATE_LIMIT_PER_MINUTE="1")
    first = client.post("/predict/single", json=LEAD, headers={header: "203.0.113.1"})
    second = client.post("/predict/single", json=LEAD, headers={header: "203.0.113.2"})
    assert (first.status_code, second.status_code) == (200, 429)


def test_x_forwarded_for_cannot_dodge_the_limit_behind_cloudflare(make_client):
    client = make_client(RATE_LIMIT_PER_MINUTE="1", TRUST_CF_CONNECTING_IP="true")
    real = {"CF-Connecting-IP": "203.0.113.1"}
    client.post("/predict/single", json=LEAD, headers={**real, "X-Forwarded-For": "1.1.1.1"})
    response = client.post(
        "/predict/single", json=LEAD, headers={**real, "X-Forwarded-For": "2.2.2.2"}
    )
    assert response.status_code == 429


def test_a_valid_key_is_exempt_because_the_demo_shares_one_address(make_client):
    client = make_client(RATE_LIMIT_PER_MINUTE="1", API_KEY="right-key")
    for _ in range(3):
        assert client.post("/predict/single", json=LEAD, headers=KEY).status_code == 200


def test_wrong_keys_are_rate_limited_too(make_client):
    client = make_client(RATE_LIMIT_PER_MINUTE="1", API_KEY="right-key")
    wrong = {"X-API-Key": "guess"}
    assert client.post("/outcomes", json=OUTCOMES, headers=wrong).status_code == 401
    assert client.post("/outcomes", json=OUTCOMES, headers=wrong).status_code == 429


def test_health_docs_and_model_info_are_not_rate_limited(make_client):
    client = make_client(RATE_LIMIT_PER_MINUTE="1")
    for _ in range(5):
        assert client.get("/health").status_code == 200
        assert client.get("/docs").status_code == 200
        assert client.get("/model/info").status_code == 200


def test_monitoring_reads_are_rate_limited_because_they_wake_the_database(make_client):
    client = make_client(RATE_LIMIT_PER_MINUTE="1")
    assert client.get("/monitoring/history").status_code == 200
    assert client.get("/monitoring/latest").status_code == 429
    assert client.get("/health/db").status_code == 429


def test_rate_limiter_window_slides():
    now = [0.0]
    limiter = RateLimiter(2, window_seconds=60, clock=lambda: now[0])
    assert limiter.allow("a") and limiter.allow("a")
    assert not limiter.allow("a")
    assert limiter.retry_after("a") == 60
    now[0] = 60.0
    assert limiter.allow("a")


def test_rate_limiter_memory_is_bounded():
    now = [0.0]
    limiter = RateLimiter(1, window_seconds=60, clock=lambda: now[0], max_clients=100)
    for i in range(1000):
        limiter.allow(f"client-{i}")
    assert len(limiter.hits) <= 100


# ---------- Batch limit ----------
def test_batch_limit_applies_to_every_caller_and_is_published(make_client, service):
    client = make_client(MAX_BATCH_SIZE="2", API_KEY="right-key")
    assert client.get("/model/info").json()["max_batch_size"] == 2
    assert client.post("/predict/batch", json={"leads": [LEAD] * 2}).status_code == 200
    for headers in ({}, KEY):
        response = client.post("/predict/batch", json={"leads": [LEAD] * 3}, headers=headers)
        assert response.status_code == 413
    assert service.db.scalar("SELECT COUNT(*) FROM predictions") == 2


def test_default_batch_limit_is_unchanged(make_client):
    assert make_client().get("/model/info").json()["max_batch_size"] == 10_000


# ---------- Rejected requests ----------
def test_a_huge_invalid_request_cannot_fill_the_log(service):
    log_rejection(service.db, "/predict/batch", [{"input": "x" * 1_000_000}])
    assert service.db.scalar("SELECT LENGTH(errors) FROM rejected_requests") == 2_000


# ---------- Clean-up ----------
def _age(db, table, column, days, where="1 = 1"):
    old = (datetime.now(UTC) - timedelta(days=days)).isoformat(timespec="seconds")
    with db.transaction() as conn:
        conn.exec_driver_sql(f"UPDATE {table} SET {column} = '{old}' WHERE {where}")


def _score(service, n=1, seed=0):
    return [p["prediction_id"] for p in service.score(as_rows(synthetic_leads(n, seed=seed)))]


def _mark_demo(db, prediction_id):
    with db.transaction() as conn:
        conn.exec_driver_sql(
            f"UPDATE predictions SET source = '{DEMO_SOURCE}' WHERE id = {prediction_id}"
        )


def test_purge_removes_old_visitor_rows_and_their_outcomes_only(service):
    old, recent, demo = _score(service, 3)
    service.record_outcomes({old: True, recent: True, demo: True})
    _mark_demo(service.db, demo)
    _age(service.db, "predictions", "created_at", 31, f"id IN ({old}, {demo})")

    assert purge_visitor_data(service.db, retention_days=30) == 1
    assert service.db.rows("SELECT id FROM predictions ORDER BY id") == [(recent,), (demo,)]
    assert service.db.rows("SELECT prediction_id FROM outcomes ORDER BY 1") == [(recent,), (demo,)]


def test_purge_removes_old_logs_but_keeps_the_latest_report(service):
    report = {"created_at": "2020-01-01T00:00:00+00:00", "n_samples": 0, "status": "ok"}
    first = save_drift_run(service.db, report)
    latest = save_drift_run(service.db, report)
    log_rejection(service.db, "/predict/single", [])
    _age(service.db, "rejected_requests", "created_at", 31)

    purge_visitor_data(service.db, retention_days=30)
    assert first != latest
    assert service.db.rows("SELECT id FROM drift_runs") == [(latest,)]
    assert service.db.scalar("SELECT COUNT(*) FROM rejected_requests") == 0


def test_purge_caps_visitor_rows_and_keeps_the_newest(service):
    ids = _score(service, 10)
    service.record_outcomes({ids[0]: True, ids[-1]: False})
    _mark_demo(service.db, ids[1])

    assert purge_visitor_data(service.db, max_rows=3) == 6
    kept = {row[0] for row in service.db.rows("SELECT id FROM predictions")}
    assert kept == {ids[1], *ids[-3:]}
    assert service.db.rows("SELECT prediction_id FROM outcomes") == [(ids[-1],)]


def test_purge_is_off_by_default(make_client, service):
    ids = _score(service)
    _age(service.db, "predictions", "created_at", 400, f"id = {ids[0]}")
    make_client()
    assert service.db.scalar("SELECT COUNT(*) FROM predictions") == 1


def test_startup_purges_when_configured(make_client, service):
    ids = _score(service)
    _age(service.db, "predictions", "created_at", 31, f"id = {ids[0]}")
    make_client(VISITOR_RETENTION_DAYS="30")
    assert service.db.scalar("SELECT COUNT(*) FROM predictions") == 0


def test_housekeeping_runs_after_requests_at_most_once_per_interval(make_client, monkeypatch):
    client = make_client(MAX_VISITOR_ROWS="1")
    calls = []
    monkeypatch.setattr(app_module, "purge_visitor_data", lambda *args: calls.append(args) or 0)

    client.post("/predict/single", json=LEAD)
    assert calls == []  # just ran at startup

    client.app.state.next_housekeeping = 0
    client.post("/predict/single", json=LEAD)
    client.post("/predict/single", json=LEAD)
    assert len(calls) == 1


def test_a_housekeeping_failure_does_not_fail_the_request(make_client, monkeypatch):
    client = make_client(MAX_VISITOR_ROWS="1")

    def broken(*args):
        raise RuntimeError("database down")

    monkeypatch.setattr(app_module, "purge_visitor_data", broken)
    client.app.state.next_housekeeping = 0
    assert client.post("/predict/single", json=LEAD).status_code == 200


def test_render_yaml_protects_the_live_api():
    from pathlib import Path

    text = (Path(__file__).parents[1] / "render.yaml").read_text()
    for setting in (
        "API_KEY",
        "RATE_LIMIT_PER_MINUTE",
        "MAX_BATCH_SIZE",
        "VISITOR_RETENTION_DAYS",
        "MAX_VISITOR_ROWS",
        "TRUST_CF_CONNECTING_IP",
    ):
        assert f"key: {setting}" in text
    assert "generateValue: true" in text
    assert "envVarKey: API_KEY" in text
