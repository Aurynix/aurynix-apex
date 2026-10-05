import math

import pandas as pd
import pytest

from apex.config import load_config
from apex.monitoring.drift import bin_shares, category_shares, psi, status, worst


def test_psi_is_zero_for_identical_distributions():
    assert psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == pytest.approx(0)


def test_psi_known_value():
    # (0.6 − 0.5)·ln(0.6/0.5) + (0.4 − 0.5)·ln(0.4/0.5)
    expected = 0.1 * math.log(1.2) - 0.1 * math.log(0.8)
    assert psi([0.5, 0.5], [0.6, 0.4]) == pytest.approx(expected)


def test_psi_smooths_empty_bins():
    assert math.isfinite(psi([0.5, 0.5], [1.0, 0.0]))
    assert psi([0.5, 0.5], [1.0, 0.0]) > 1


def test_bin_shares_use_fixed_edges():
    edges = [-math.inf, 0, 10, math.inf]
    assert bin_shares(pd.Series([0, 0, 5, 50]), edges) == [0.5, 0.25, 0.25]


def test_category_shares_group_unseen_values():
    shares = category_shares(
        pd.Series(["Google", "TikTok", "Google", "Reference"]), ["Google", "Reference"], "__other__"
    )
    assert shares == {"Google": 0.5, "Reference": 0.25, "__other__": 0.25}


def test_status_thresholds_and_worst():
    cfg = load_config()  # warning 0.10, drift 0.25
    assert [status(v, cfg) for v in (0.05, 0.1, 0.3)] == ["ok", "warning", "drift"]
    assert worst(["ok", "drift", "warning"]) == "drift"
    assert worst([]) == "ok"
