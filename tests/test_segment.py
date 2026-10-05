import numpy as np
import pytest

from apex.models.segment import assign, segment_table, thresholds

SCORES = np.arange(1, 101) / 100  # 0.01 … 1.00


def test_thresholds_follow_capacity_shares():
    cut = thresholds(SCORES)  # config: high 20%, medium 30%
    segments = assign(SCORES, cut)
    assert (segments == "High").mean() == pytest.approx(0.2, abs=0.01)
    assert (segments == "Medium").mean() == pytest.approx(0.3, abs=0.01)
    assert (segments == "Low").mean() == pytest.approx(0.5, abs=0.01)


def test_assign_uses_fixed_thresholds_for_new_scores():
    cut = {"high": 0.75, "medium": 0.27}
    assert assign([0.9, 0.75, 0.5, 0.27, 0.1], cut).tolist() == [
        "High",
        "High",
        "Medium",
        "Medium",
        "Low",
    ]


def test_segment_table_shares_and_rates():
    y = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    segments = ["High", "High", "Medium", "Medium", "Low", "Low", "Low", "Low", "Low", "Low"]
    table = segment_table(y, segments)
    assert table.index.tolist() == ["High", "Medium", "Low"]
    assert table.loc["High", "rate"] == 1.0
    assert table.loc["High", "share_of_conversions"] == pytest.approx(2 / 3)
    assert table["share_of_leads"].sum() == pytest.approx(1.0)
