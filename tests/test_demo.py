import io
import sys
import urllib.error
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
