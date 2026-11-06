import io
import json
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "app"))
import demo  # noqa: E402


def test_leads_from_raw_kaggle_columns():
    raw = pd.DataFrame(
        {
            "Prospect ID": ["a"],
            "Lead Origin": ["API"],
            "Do Not Email": ["Yes"],
            "TotalVisits": [2.0],
            "Total Time Spent on Website": [120],
            "Specialization": ["Select"],
            "Magazine": ["No"],
        }
    )
    assert demo.leads_from_csv(raw) == [
        {"lead_origin": "API", "do_not_email": True, "total_visits": 2, "time_on_website": 120}
    ]


def test_leads_from_api_columns_keep_missing_fields_out():
    df = pd.DataFrame({"lead_origin": ["API", "Lead Add Form"], "total_visits": [3, None]})
    assert demo.leads_from_csv(df) == [
        {"lead_origin": "API", "total_visits": 3},
        {"lead_origin": "Lead Add Form"},
    ]


def test_csv_without_lead_origin_is_rejected():
    with pytest.raises(ValueError, match="lead_origin"):
        demo.leads_from_csv(pd.DataFrame({"x": [1]}))


def test_demo_shows_an_error_when_the_api_is_down(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")  # nothing listens here
    at = AppTest.from_file(str(Path(__file__).parents[1] / "app" / "demo.py")).run()
    assert not at.exception
    assert "Cannot reach the API" in at.error[0].value


@pytest.mark.parametrize(("key", "sent"), [("secret", "secret"), ("", None)])
def test_demo_sends_the_api_key_only_when_it_has_one(key, sent, monkeypatch):
    seen = {}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def urlopen(request, timeout):
        seen["key"] = request.get_header("X-api-key")
        return Response(b"{}")

    monkeypatch.setattr(demo, "API_KEY", key)
    monkeypatch.setattr(demo.urllib.request, "urlopen", urlopen)
    demo.call("POST", "/monitoring/run")
    assert seen["key"] == sent


NOW = datetime(2026, 11, 6, 22, 50, tzinfo=UTC)


@pytest.mark.parametrize(
    ("age", "words"),
    [
        (timedelta(seconds=30), "just now"),
        (timedelta(minutes=2), "2 min ago"),
        (timedelta(minutes=59), "59 min ago"),
        (timedelta(hours=1), "1 h ago"),
        (timedelta(hours=23), "23 h ago"),
        (timedelta(days=1), "1 day ago"),
        (timedelta(days=3), "3 days ago"),
    ],
)
def test_last_run_says_how_long_ago(age, words):
    created = (NOW - age).isoformat(timespec="seconds")
    assert demo.last_run(created, NOW).startswith(f"last run {words} (")


def test_last_run_matches_the_requested_format():
    assert (
        demo.last_run("2026-11-06T22:48:00+00:00", NOW)
        == "last run 2 min ago (7 Nov 2026, 00:48 Jerusalem time)"
    )


def test_last_run_converts_any_offset_to_jerusalem_time():
    assert demo.last_run("2026-11-06T17:48:00-05:00", NOW).endswith(
        "(7 Nov 2026, 00:48 Jerusalem time)"
    )


def test_last_run_follows_daylight_saving():
    assert demo.last_run("2026-07-01T22:48:00+00:00", NOW).endswith(
        "(2 Jul 2026, 01:48 Jerusalem time)"
    )


def test_last_run_handles_a_clock_ahead_of_the_report():
    assert demo.last_run("2026-11-06T22:51:00+00:00", NOW).startswith("last run just now")


def test_last_run_keeps_an_unreadable_timestamp_visible():
    assert demo.last_run("not a date", NOW) == "last run not a date"


def _render_report(monkeypatch, data_source):
    """Run the page against a fake API that has one monitoring report."""
    from streamlit.testing.v1 import AppTest

    report = {
        "status": "ok",
        "n_samples": 300,
        "created_at": "2026-11-06T22:48:00+00:00",
        "performance": {
            "status": "ok",
            "n_outcomes": 600,
            "window_days": 30,
            "actual": {"pr_auc": 0.8209, "precision_high": 0.9083},
            "expected": {"pr_auc": 0.7874, "precision_high": 0.8324},
        },
    }
    if data_source:
        report["data_source"] = data_source
    info = {
        "model_version": "apex-test",
        "trained_at": "2026-10-07T00:00:00+00:00",
        "training_rows": 100,
        "max_batch_size": 100,
        "test_metrics": {"pr_auc": 0.7874, "precision_top20": 0.83, "recall_top20": 0.43},
    }

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def urlopen(request, timeout):
        body = report if request.full_url.endswith("/monitoring/latest") else info
        return Response(json.dumps(body).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    at = AppTest.from_file(str(Path(__file__).parents[1] / "app" / "demo.py")).run()
    assert not at.exception
    return at


def _captions(at) -> str:
    return " | ".join(caption.value for caption in at.caption)


def _delta_colors(at) -> set[int]:
    return {m.proto.color for m in at.metric if "vs. test" in m.delta}


def test_report_uses_the_new_heading_and_a_readable_time(monkeypatch):
    at = _render_report(monkeypatch, None)
    assert [s.value for s in at.subheader][0] == "Performance on recorded outcomes"
    assert "(7 Nov 2026, 00:48 Jerusalem time)" in _captions(at)
    assert "2026-11-06T22:48" not in _captions(at)


def test_demo_data_gets_the_caption_and_neutral_deltas(monkeypatch):
    from streamlit.proto.Metric_pb2 import Metric

    at = _render_report(monkeypatch, "demo_seed")
    assert (
        "Synthetic demo outcomes are drawn from the model's own probabilities, so this is "
        "higher than the real held-out test result (0.787). It is not evidence of quality."
    ) in _captions(at)
    assert _delta_colors(at) == {Metric.MetricColor.GRAY}


def test_other_data_keeps_colored_deltas_and_no_demo_caption(monkeypatch):
    from streamlit.proto.Metric_pb2 import Metric

    at = _render_report(monkeypatch, None)
    assert "Synthetic demo outcomes" not in _captions(at)
    assert _delta_colors(at) == {Metric.MetricColor.GREEN}


def _raise_http(code: int):
    def urlopen(request, timeout):
        raise urllib.error.HTTPError("http://api", code, "error", {}, io.BytesIO(b"detail"))

    return urlopen


def test_starting_api_status_is_unavailable(monkeypatch):
    monkeypatch.setattr(demo.urllib.request, "urlopen", _raise_http(503))
    with pytest.raises(demo.ApiUnavailable, match="not ready"):
        demo.call("GET", "/model/info")


def test_other_api_errors_are_not_treated_as_starting(monkeypatch):
    monkeypatch.setattr(demo.urllib.request, "urlopen", _raise_http(404))
    with pytest.raises(RuntimeError, match="API error 404") as err:
        demo.call("GET", "/model/info")
    assert not isinstance(err.value, demo.ApiUnavailable)


def test_read_timeout_is_unavailable(monkeypatch):
    def urlopen(request, timeout):
        raise TimeoutError

    monkeypatch.setattr(demo.urllib.request, "urlopen", urlopen)
    with pytest.raises(demo.ApiUnavailable, match="timed out"):
        demo.call("GET", "/model/info")


def test_wait_for_api_retries_until_it_wakes(monkeypatch):
    answers = [demo.ApiUnavailable("asleep"), demo.ApiUnavailable("asleep"), {"ok": True}]

    def call(method, path):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(demo, "call", call)
    monkeypatch.setattr(demo.time, "sleep", lambda seconds: None)
    assert demo.wait_for_api("/model/info", wait_seconds=60) == {"ok": True}
    assert answers == []


def test_wait_for_api_gives_up_after_the_deadline(monkeypatch):
    def call(method, path):
        raise demo.ApiUnavailable("asleep")

    now = iter([0, 10, 21])  # start, after the first try, after the second try
    monkeypatch.setattr(demo, "call", call)
    monkeypatch.setattr(demo.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(demo.time, "monotonic", lambda: next(now))
    with pytest.raises(demo.ApiUnavailable):
        demo.wait_for_api("/model/info", wait_seconds=20)


def test_wait_for_api_does_not_retry_other_errors(monkeypatch):
    calls = []

    def call(method, path):
        calls.append(path)
        raise RuntimeError("API error 500")

    monkeypatch.setattr(demo, "call", call)
    with pytest.raises(RuntimeError):
        demo.wait_for_api("/model/info", wait_seconds=60)
    assert len(calls) == 1
