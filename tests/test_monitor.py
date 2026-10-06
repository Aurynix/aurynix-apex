import numpy as np

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
