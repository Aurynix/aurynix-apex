import json

import numpy as np
import pytest

from apex.monitoring.simulate import drift
from tests.conftest import as_rows, synthetic_leads


def test_stable_traffic_is_not_flagged(service, monitor):
    service.score(as_rows(synthetic_leads(400, seed=1)))
    report = monitor.run()
    assert report["status"] == "ok"
    assert report["prediction"]["status"] == "ok"


def test_shifted_traffic_is_flagged(service, monitor):
    service.score(as_rows(drift(synthetic_leads(400, seed=1))))
    report = monitor.run()
    assert report["status"] == "drift"
    assert report["features"]["Lead Source"]["status"] == "drift"
    assert report["features"]["Lead Source"]["unseen_share"] > 0.2
    assert report["features"]["Total Time Spent on Website"]["status"] == "drift"
    assert (
        report["data_quality"]["missing_rate"]["What is your current occupation"]["status"]
        == "warning"
    )


def _demo_rows(service, monitor, n=300):
    from apex.api.database import DEMO_SOURCE

    leads = synthetic_leads(n, seed=1)
    ids = [p["prediction_id"] for p in service.score(as_rows(leads))]
    service.record_outcomes(
        dict(zip(ids, map(bool, leads["Total Time Spent on Website"] > 1000), strict=True))
    )
    with service.db.transaction() as conn:
        conn.exec_driver_sql(f"UPDATE predictions SET source = '{DEMO_SOURCE}'")
    monitor.source = DEMO_SOURCE


def test_with_a_source_visitor_traffic_cannot_change_the_report(service, monitor):
    from apex.api.database import log_rejection

    _demo_rows(service, monitor)
    clean = monitor.run(save=False)
    assert clean["status"] == "ok"
    assert clean["data_source"] == "demo_seed"

    attack = drift(synthetic_leads(3000, seed=9))
    ids = [p["prediction_id"] for p in service.score(as_rows(attack))]
    service.record_outcomes(dict.fromkeys(ids, False))
    for _ in range(200):
        log_rejection(service.db, "/predict/single", [])

    after = monitor.run(save=False)
    for part in ("features", "prediction", "performance", "n_samples", "status"):
        assert after[part] == clean[part]
    assert after["data_quality"]["rejected_requests"] == 0
    assert after["retrain_recommended"] is False


def test_without_a_source_every_row_counts(service, monitor):
    _demo_rows(service, monitor)
    monitor.source = None
    service.score(as_rows(drift(synthetic_leads(3000, seed=9))))
    report = monitor.run(save=False)
    assert report["n_samples"] == 3300
    assert report["status"] == "drift"
    assert "data_source" not in report


@pytest.mark.parametrize(("value", "expected"), [("demo_seed", "demo_seed"), ("", None)])
def test_create_monitor_reads_the_source_setting(
    value, expected, trained, db, monitor, tmp_path, monkeypatch
):
    from apex.monitoring.monitor import create_monitor

    model, meta, profile = trained
    (tmp_path / "reference_profile.json").write_text(json.dumps(profile))
    monkeypatch.setenv("MONITORING_SOURCE", value)
    assert create_monitor(db, model, meta).source == expected


def test_too_few_predictions_is_insufficient_data(service, monitor):
    service.score(as_rows(synthetic_leads(10)))
    assert monitor.run()["status"] == "insufficient_data"


def test_runs_are_saved_and_listed(service, monitor):
    service.score(as_rows(synthetic_leads(100)))
    first = monitor.run()
    assert monitor.history()[0]["id"] == first["id"]


def _score_with_outcomes(service, leads, converted):
    ids = [p["prediction_id"] for p in service.score(as_rows(leads))]
    assert service.record_outcomes(dict(zip(ids, map(bool, converted), strict=True))) == []


def test_performance_ok_when_the_model_is_still_right(service, monitor):
    leads = synthetic_leads(300, seed=2)
    _score_with_outcomes(
        service, leads, leads["Total Time Spent on Website"] > 1000
    )  # same rule as training
    perf = monitor.run()["performance"]
    assert perf["status"] == "ok"
    assert perf["n_outcomes"] == 300
    assert perf["actual"]["pr_auc"] > 0.9


def test_performance_drop_recommends_retraining(service, monitor):
    leads = synthetic_leads(300, seed=2)
    random_results = (
        np.random.default_rng(0).random(300) < 0.4
    )  # behavior changed: score no longer predicts
    _score_with_outcomes(service, leads, random_results)
    report = monitor.run()
    assert report["performance"]["status"] == "drift"
    assert report["retrain_recommended"] is True


def test_performance_needs_enough_outcomes(service, monitor):
    leads = synthetic_leads(20, seed=3)
    _score_with_outcomes(service, leads, leads["Total Time Spent on Website"] > 1000)
    assert monitor.run()["performance"]["status"] == "insufficient_data"
