import copy
import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from apex.api.app import create_app
from apex.api.database import read_outcome_leads
from apex.config import load_config, path
from apex.monitoring import seed
from apex.monitoring.monitor import Monitor
from tests.conftest import as_rows, synthetic_leads


def seed_rows(n: int = 600) -> list[dict]:
    leads = synthetic_leads(n, seed=5)
    return [
        {"lead": lead, "converted": int(time > 1000)}
        for lead, time in zip(as_rows(leads), leads["Total Time Spent on Website"], strict=True)
    ]


@pytest.fixture
def real_thresholds(trained, db, tmp_path, monkeypatch):
    """A Monitor with the real thresholds from config.json (200 predictions, 200 outcomes)."""
    from apex.monitoring import monitor as monitor_module

    model, meta, profile = trained
    monkeypatch.setattr(monitor_module, "path", lambda key: tmp_path)
    cfg = copy.deepcopy(load_config())
    return Monitor(model, profile, db, cfg, meta["performance"]["test"])


def test_seed_marks_its_rows_and_passes_the_real_monitoring_thresholds(service, real_thresholds):
    assert seed.seed_demo_data(service, real_thresholds, seed_rows()) == 600
    assert service.db.rows("SELECT DISTINCT source FROM predictions") == [("demo_seed",)]
    assert service.db.scalar("SELECT COUNT(*) FROM outcomes") == 600

    report = real_thresholds.history(limit=1)[0]
    assert report["n_samples"] >= real_thresholds.cfg["monitoring"]["min_samples"]
    assert report["performance"]["n_outcomes"] >= real_thresholds.cfg["monitoring"]["min_outcomes"]
    assert report["status"] != "insufficient_data"
    assert report["performance"]["status"] != "insufficient_data"


def test_seed_is_idempotent(service, real_thresholds):
    assert seed.seed_demo_data(service, real_thresholds, seed_rows()) == 600
    runs = len(real_thresholds.history())
    assert seed.seed_demo_data(service, real_thresholds, seed_rows()) == 0
    assert service.db.scalar("SELECT COUNT(*) FROM predictions") == 600
    assert service.db.scalar("SELECT COUNT(*) FROM outcomes") == 600
    assert len(real_thresholds.history()) == runs  # no extra monitoring run


def test_seed_does_not_run_when_real_predictions_exist(service, real_thresholds):
    service.score([seed_rows(1)[0]["lead"]])
    assert seed.seed_demo_data(service, real_thresholds, seed_rows()) == 0
    assert service.db.scalar("SELECT COUNT(*) FROM predictions") == 1
    assert real_thresholds.history() == []


def test_seed_scores_like_the_scoring_service(service, real_thresholds):
    rows = seed_rows()
    seed.seed_demo_data(service, real_thresholds, rows)
    seeded = service.db.rows("SELECT score, segment FROM predictions ORDER BY id LIMIT 50")
    scored = service.score([row["lead"] for row in rows[:50]])
    assert seeded == [(result["score"], result["segment"]) for result in scored]


def test_seed_timestamps_are_spread_over_the_monitoring_windows():
    moment = datetime(2026, 11, 6, tzinfo=UTC)
    created, recorded = seed.timestamps(600, moment)
    created = [datetime.fromisoformat(value) for value in created]
    recorded = [datetime.fromisoformat(value) for value in recorded]

    assert sum(moment - value < timedelta(days=7) for value in created) == 300
    assert all(moment - value < timedelta(days=30) for value in created)
    assert max(moment - value for value in created) > timedelta(days=28)
    assert len(set(created)) == 600
    assert all(c <= r <= moment for c, r in zip(created, recorded, strict=True))
    assert all(moment - value < timedelta(days=30) for value in recorded)


def test_seeded_outcomes_are_left_out_of_retraining(service, real_thresholds):
    seed.seed_demo_data(service, real_thresholds, seed_rows())
    assert read_outcome_leads(service.db) == []

    real = service.score([seed_rows(1)[0]["lead"]])[0]["prediction_id"]
    service.record_outcomes({real: True})
    assert len(read_outcome_leads(service.db)) == 1


@pytest.mark.parametrize("value", [None, "", "false", "no"])
def test_seed_is_off_unless_enabled(value, service, monitor, monkeypatch):
    if value is None:
        monkeypatch.delenv("SEED_DEMO_DATA", raising=False)
    else:
        monkeypatch.setenv("SEED_DEMO_DATA", value)
    with TestClient(create_app(service, monitor)):
        assert service.db.scalar("SELECT COUNT(*) FROM predictions") == 0


def test_startup_seeds_when_enabled_and_latest_has_a_report(service, monitor, monkeypatch):
    monkeypatch.setenv("SEED_DEMO_DATA", "true")
    monkeypatch.setattr(seed, "load_seed_rows", seed_rows)
    with TestClient(create_app(service, monitor)) as client:
        report = client.get("/monitoring/latest").json()
    assert report["n_samples"] == 300
    assert report["performance"]["n_outcomes"] == 600


def test_a_seed_failure_does_not_stop_the_api(service, monitor, monkeypatch):
    def broken():
        raise FileNotFoundError("no seed file")

    monkeypatch.setenv("SEED_DEMO_DATA", "true")
    monkeypatch.setattr(seed, "load_seed_rows", broken)
    with TestClient(create_app(service, monitor)) as client:
        assert client.get("/health").status_code == 200


def _make_old(db, days):
    """Move every demo row `days` into the past, as if the seed had run back then."""
    moment = datetime.now(UTC) - timedelta(days=days)
    created, recorded = seed.timestamps(600, moment)
    with db.transaction() as conn:
        for i, (c, r) in enumerate(zip(created, recorded, strict=True), start=1):
            conn.exec_driver_sql(f"UPDATE predictions SET created_at = '{c}' WHERE id = {i}")
            conn.exec_driver_sql(
                f"UPDATE outcomes SET recorded_at = '{r}' WHERE prediction_id = {i}"
            )


def test_refresh_keeps_monitoring_working_after_the_rows_age(service, real_thresholds):
    seed.seed_demo_data(service, real_thresholds, seed_rows())
    _make_old(service.db, days=60)
    assert real_thresholds.run()["status"] == "insufficient_data"

    assert seed.refresh_demo_data(service, real_thresholds) == 600
    report = real_thresholds.history(limit=1)[0]
    assert report["n_samples"] == 300
    assert report["performance"]["n_outcomes"] == 600
    assert report["status"] != "insufficient_data"


def test_refresh_changes_only_the_timestamps_of_demo_rows(service, real_thresholds):
    seed.seed_demo_data(service, real_thresholds, seed_rows())
    _make_old(service.db, days=60)
    visitor = service.score([seed_rows(1)[0]["lead"]])[0]["prediction_id"]
    service.record_outcomes({visitor: True})
    with service.db.transaction() as conn:
        conn.exec_driver_sql(f"UPDATE predictions SET created_at = '2020' WHERE id = {visitor}")
    columns = "id, model_version, lead, score, segment, source"
    before = service.db.rows(f"SELECT {columns} FROM predictions ORDER BY id")
    outcomes = service.db.rows("SELECT prediction_id, converted FROM outcomes ORDER BY 1")
    visitor_time = service.db.rows(
        f"SELECT p.created_at, o.recorded_at FROM predictions p JOIN outcomes o "
        f"ON o.prediction_id = p.id WHERE p.id = {visitor}"
    )

    assert seed.refresh_demo_data(service, real_thresholds) == 600
    assert service.db.rows(f"SELECT {columns} FROM predictions ORDER BY id") == before
    assert service.db.rows("SELECT prediction_id, converted FROM outcomes ORDER BY 1") == outcomes
    assert visitor_time == service.db.rows(
        f"SELECT p.created_at, o.recorded_at FROM predictions p JOIN outcomes o "
        f"ON o.prediction_id = p.id WHERE p.id = {visitor}"
    )


def test_fresh_rows_are_not_refreshed(service, real_thresholds):
    seed.seed_demo_data(service, real_thresholds, seed_rows())
    runs = len(real_thresholds.history())
    assert seed.refresh_demo_data(service, real_thresholds) == 0
    assert len(real_thresholds.history()) == runs


def test_refresh_does_nothing_without_demo_rows(service, real_thresholds):
    service.score([seed_rows(1)[0]["lead"]])
    assert seed.refresh_demo_data(service, real_thresholds) == 0
    assert real_thresholds.history() == []


@pytest.mark.parametrize("enabled", [False, True])
def test_startup_refreshes_only_when_seeding_is_enabled(enabled, service, monitor, monkeypatch):
    seed.seed_demo_data(service, monitor, seed_rows())
    _make_old(service.db, days=60)
    newest = "SELECT MAX(created_at) FROM predictions"
    stale = service.db.scalar(newest)

    if enabled:
        monkeypatch.setenv("SEED_DEMO_DATA", "true")
    else:
        monkeypatch.delenv("SEED_DEMO_DATA", raising=False)
    with TestClient(create_app(service, monitor)):
        assert (service.db.scalar(newest) != stale) is enabled


def test_committed_seed_file_has_only_model_fields_and_enough_rows():

    cfg = load_config()
    rows = json.loads(path("demo_seed").read_text(encoding="utf-8"))
    assert len(rows) // 2 >= cfg["monitoring"]["min_samples"]
    assert len(rows) >= cfg["monitoring"]["min_outcomes"]
    assert {r["converted"] for r in rows} == {0, 1}
    assert all(set(r) == {"lead", "converted"} for r in rows)
    assert all(set(r["lead"]) == set(cfg["serving"]["input_fields"]) for r in rows)
