import numpy as np
import pandas as pd
import pytest

from apex.data.features import add_features, build_pipeline


@pytest.fixture
def raw():
    """Raw leads as they arrive (no target), with missing and rare values."""
    n = 200
    rng = np.random.default_rng(0)
    visits = rng.choice([0.0, 2.0, 5.0, np.nan], n)
    return pd.DataFrame(
        {
            "Prospect ID": [f"id{i}" for i in range(n)],
            "Lead Origin": rng.choice(["API", "Landing Page Submission"], n),
            "Lead Source": ["Google"] * 150 + ["Direct Traffic"] * 49 + ["bing"],
            "Do Not Email": rng.choice(["Yes", "No"], n),
            "TotalVisits": visits,
            "Total Time Spent on Website": rng.integers(0, 2000, n),
            "Page Views Per Visit": np.where(visits > 0, 2.0, visits),
            "City": rng.choice(["Mumbai", "Select", None], n),
            "What matters most to you in choosing a course": "Better Career Prospects",
        }
    )


def test_add_features_time_per_visit_and_activity():
    df = pd.DataFrame({"TotalVisits": [0, 4], "Total Time Spent on Website": [0, 200]})
    out = add_features(df)
    assert out["time_per_visit"].tolist() == [0, 50]
    assert out["has_web_activity"].tolist() == [0, 1]


def test_add_features_drops_redundant_column(raw):
    assert "What matters most to you in choosing a course" not in add_features(raw).columns


def test_pipeline_outputs_numbers_only_with_no_missing(raw):
    out = build_pipeline().fit_transform(raw)
    assert out.shape[0] == len(raw)
    assert np.isfinite(out).all()


def test_pipeline_groups_rare_and_unseen_categories(raw):
    pipeline = build_pipeline().fit(raw)
    names = list(pipeline[-1].get_feature_names_out())
    assert "Lead Source_bing" not in names  # 1 of 200 rows: below the 1% share
    assert "Lead Source_infrequent_sklearn" in names

    new_lead = raw.iloc[[0]].assign(**{"Lead Source": "TikTok"})  # never seen in training
    row = pd.Series(pipeline.transform(new_lead)[0], index=names)
    assert row["Lead Source_infrequent_sklearn"] == 1


def test_pipeline_learns_from_training_rows_only(raw):
    """Scaling uses training statistics, so new rows are not re-centered on themselves."""
    train, new = raw.iloc[:150], raw.iloc[150:].assign(**{"Total Time Spent on Website": 5000})
    pipeline = build_pipeline().fit(train)
    names = list(pipeline[-1].get_feature_names_out())
    col = names.index("Total Time Spent on Website")
    assert pipeline.transform(new)[:, col].mean() > 3


def test_add_features_reduces_presence_only_columns():
    df = pd.DataFrame(
        {
            "TotalVisits": [1, 1],
            "Total Time Spent on Website": [5, 5],
            "Specialization": ["Finance Management", "Missing"],
        }
    )
    assert add_features(df)["Specialization"].tolist() == ["Given", "Missing"]


def test_add_features_building_blocks_from_config():
    """ratios, flags, bins, presence_only, drop: the bank-style and lead-style blocks."""
    config = {
        "features": {
            "presence_only": ["spec"],
            "ratios": {"per_visit": ["time", "visits"]},
            "flags": {"contacted_before": ["pdays", -1]},
            "bins": {"age": [25, 60]},
            "drop": ["age"],
        }
    }
    df = pd.DataFrame(
        {
            "spec": ["Finance", "Missing", "HR"],
            "time": [100, 0, 30],
            "visits": [4, 0, 3],
            "pdays": [-1, 10, -1],
            "age": [22, 40, 70],
        }
    )
    out = add_features(df, config)
    assert out["spec"].tolist() == ["Given", "Missing", "Given"]
    assert out["per_visit"].tolist() == [25, 0, 10]
    assert out["contacted_before"].tolist() == [0, 1, 0]
    assert out["age_group"].tolist() == ["≤ 25", "25–60", "> 60"]
    assert "age" not in out.columns
