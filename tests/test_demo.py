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


def _render_report(monkeypatch, data_source, extra=None):
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
    report.update(extra or {})
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


def test_page_uses_the_logo_and_no_emoji_in_the_title(monkeypatch):
    at = _render_report(monkeypatch, None)
    assert at.title[0].value == "Aurynix Apex: lead scoring"


def test_logo_file_exists_and_is_a_static_svg():
    logo = Path(demo.LOGO)
    assert logo.parent.name == "app"  # the Dockerfiles copy app/ into the image
    text = logo.read_text(encoding="utf-8").lower()
    assert text.lstrip().startswith("<svg")
    assert "<script" not in text
    assert "http" not in text.replace("http://www.w3.org/2000/svg", "")


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


FULL_REPORT = {
    "prediction": {
        "segment_shares": {
            "current": {"high": 0.21, "low": 0.49, "medium": 0.30},
            "reference": {"high": 0.20, "low": 0.50, "medium": 0.30},
        }
    },
    "features": {
        "What is your current occupation": {"status": "ok", "psi": 0.0496},
        "TotalVisits": {"status": "ok", "psi": 0.0104},
        "time_per_visit": {"status": "ok", "psi": 0.0071},
    },
    "data_quality": {
        "status": "ok",
        "missing_rate": {
            "Specialization": {"reference": 0.3658, "current": 0.4, "status": "ok"},
            "Total Time Spent on Website": {"reference": 0.0, "current": 0.0, "status": "ok"},
        },
        "defaulted_share": 0.59,
        "rejected_requests": 0,
        "rejection_rate": 0.0,
    },
}


def test_drift_and_data_quality_tables_are_explained(monkeypatch):
    at = _render_report(monkeypatch, "demo_seed", FULL_REPORT)
    text = " ".join(m.value for m in at.markdown)
    assert "Population Stability Index" in text
    assert "below 0.10" in text and "0.25 or more" in text
    assert "share of missing values" in text and "10 points" in text


def test_explanations_follow_the_config_thresholds():
    mon = {"psi_warning": 0.2, "psi_drift": 0.4, "missing_rate_change_warning": 0.15}
    assert "below 0.20" in demo.psi_help(mon)
    assert "0.20 to 0.40" in demo.psi_help(mon)
    assert "0.40 or more" in demo.psi_help(mon)
    assert "15 points" in demo.data_quality_help(mon)


def test_monitoring_tables_use_clean_labels_and_percentages(monkeypatch):
    at = _render_report(monkeypatch, "demo_seed", FULL_REPORT)
    drift, quality = (table.value for table in at.dataframe)
    assert list(drift.index) == ["Occupation", "Website visits", "Time per visit"]
    assert list(quality.index) == ["Specialization", "Time on website"]
    assert quality.loc["Specialization", "reference"] == pytest.approx(36.58)
    assert quality.loc["Specialization", "current"] == pytest.approx(40.0)


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("What is your current occupation", "Occupation"),
        ("TotalVisits", "Website visits"),
        ("Total Time Spent on Website", "Time on website"),
        ("time_per_visit", "Time per visit"),
        ("has_web_activity", "Has web activity"),
        ("some_new_field", "Some new field"),
    ],
)
def test_field_labels(name, label):
    assert demo.field_label(name) == label


def test_segment_chart_is_ordered_and_starts_at_zero():
    spec = demo.segment_chart(FULL_REPORT["prediction"]["segment_shares"]).to_dict()
    x, y = spec["encoding"]["x"], spec["encoding"]["y"]
    assert x["sort"] == ["High", "Medium", "Low"]
    assert y["scale"]["domainMin"] == 0
    assert y["scale"]["zero"] is True


def test_ranked_csv_has_row_numbers_and_input_fields():
    df = pd.DataFrame({"Prospect ID": ["a", "b", "c"]})
    leads = [
        {"lead_origin": "API", "total_visits": 2},
        {"lead_origin": "Lead Add Form", "lead_source": "Google", "occupation": "Student"},
        {"lead_origin": "API", "time_on_website": 600},
    ]

    def scored(score, segment):
        return {"score": score, "segment": segment, "reasons": {"up": ["u"], "down": ["d"]}}

    result = [scored(0.2, "low"), scored(0.9, "high"), scored(0.5, "medium")]
    ranked = demo.ranked_leads(df, leads, result)
    assert list(ranked.columns) == [
        "row",
        "Prospect ID",
        "score",
        "segment",
        "lead_origin",
        "lead_source",
        "occupation",
        "total_visits",
        "time_on_website",
        "top reason up",
        "top reason down",
    ]
    assert list(ranked["row"]) == [2, 3, 1]
    assert list(ranked["Prospect ID"]) == ["b", "c", "a"]
    assert ranked.iloc[0]["lead_source"] == "Google"
    downloaded = pd.read_csv(io.StringIO(ranked.to_csv(index=False)))
    assert list(downloaded.columns) == list(ranked.columns)


def test_csv_time_is_minutes_on_screen_and_seconds_in_the_download():
    df = pd.DataFrame({"x": [1, 2]})
    leads = [
        {"lead_origin": "API", "time_on_website": 600},
        {"lead_origin": "Lead Add Form", "time_on_website": 95},
    ]
    result = [
        {"score": score, "segment": "low", "reasons": {"up": [], "down": []}}
        for score in (0.9, 0.1)
    ]
    ranked = demo.ranked_leads(df, leads, result)

    screen = demo.ranked_for_screen(ranked)
    assert "time_on_website" not in screen.columns
    assert list(screen["Time on website (min)"]) == [10.0, 1.6]

    downloaded = pd.read_csv(io.StringIO(ranked.to_csv(index=False)))
    assert list(downloaded["time_on_website"]) == [600, 95]
    # The download can be uploaded again and gives the API the same leads (in seconds).
    assert demo.leads_from_csv(downloaded) == leads


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
