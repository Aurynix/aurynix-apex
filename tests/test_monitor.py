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
