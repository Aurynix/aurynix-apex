import pandas as pd
import pytest

from apex.data.eda import conversion_by, cramers_v, numeric_bins


def test_conversion_by_groups_small_values_into_other():
    df = pd.DataFrame({"src": ["a"] * 4 + ["b"], "y": [1, 1, 0, 0, 1]})
    out = conversion_by(df, "src", "y", min_rows=2)
    assert out.loc["a", "rate"] == 0.5
    assert out.loc["Other", "leads"] == 1


def test_numeric_bins_puts_zeros_first():
    df = pd.DataFrame({"x": [0, 0, 1, 2, 3, 4], "y": [0, 0, 1, 1, 1, 1]})
    out = numeric_bins(df, "x", "y", q=2)
    assert out.index[0] == "0"
    assert out.loc["0", "rate"] == 0
    assert out["leads"].sum() == 6


def test_cramers_v_is_one_for_identical_and_zero_for_independent():
    a = pd.Series(["x", "y"] * 50)
    assert cramers_v(a, a) == pytest.approx(1)
    assert cramers_v(a, pd.Series(["p"] * 50 + ["q"] * 50)) == pytest.approx(0)
