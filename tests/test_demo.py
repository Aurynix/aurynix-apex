import sys
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
