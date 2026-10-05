import numpy as np
import pandas as pd
import pytest

from apex.config import load_config
from apex.monitoring.reference import build_profile, categorical_profile, numeric_profile


def test_numeric_profile_bins_cover_everything():
    profile = numeric_profile(pd.Series(np.arange(100.0)), n_bins=10)
    assert profile["edges"][0] == -np.inf and profile["edges"][-1] == np.inf
    assert len(profile["shares"]) == len(profile["edges"]) - 1
    assert sum(profile["shares"]) == pytest.approx(1.0)


def test_categorical_profile_shares():
    assert categorical_profile(pd.Series(["a", "a", "b", "c"])) == {"a": 0.5, "b": 0.25, "c": 0.25}


def test_build_profile_counts_placeholders_as_missing():
    raw = pd.DataFrame({"Specialization": ["Select", None, "Finance", "Finance"]})
    inputs = pd.DataFrame(
        {"Specialization": ["Missing", "Missing", "Given", "Given"], "x": [1, 2, 3, 4]}
    )
    profile = build_profile(
        raw, inputs, [0.1, 0.2, 0.8, 0.9], ["Low", "Low", "High", "High"], load_config()
    )
    assert profile["missing_rate"]["Specialization"] == 0.5
    assert set(profile["numeric"]) == {"x"}
    assert profile["categorical"]["Specialization"] == {"Missing": 0.5, "Given": 0.5}
    assert profile["segment_shares"] == {"Low": 0.5, "High": 0.5}
