import numpy as np
import pandas as pd
import pytest

from apex.data import split
from apex.data.split import load_splits, make_splits


@pytest.fixture
def leads():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"Prospect ID": [f"id{i}" for i in range(1000)], "Converted": rng.random(1000) < 0.385}
    ).astype({"Converted": int})


def test_make_splits_sizes_are_60_20_20(leads):
    counts = make_splits(leads).value_counts()
    assert counts.to_dict() == {"train": 600, "val": 200, "test": 200}


def test_make_splits_keeps_class_ratio(leads):
    rates = leads.groupby(make_splits(leads))["Converted"].mean()
    assert (rates - leads["Converted"].mean()).abs().max() < 0.01


def test_make_splits_is_reproducible(leads):
    pd.testing.assert_series_equal(make_splits(leads), make_splits(leads))


def test_load_splits_hides_test_unless_asked(leads, tmp_path, monkeypatch):
    file = tmp_path / "splits.csv"
    pd.DataFrame({"Prospect ID": leads["Prospect ID"], "split": make_splits(leads)}).to_csv(
        file, index=False
    )
    monkeypatch.setattr(split, "path", lambda key: file)

    assert set(load_splits(leads)) == {"train", "val"}
    parts = load_splits(leads, include_test=True)
    assert sum(len(p) for p in parts.values()) == len(leads)
    assert not set(parts["train"]["Prospect ID"]) & set(parts["test"]["Prospect ID"])


def test_load_splits_rejects_unknown_rows(leads, tmp_path, monkeypatch):
    file = tmp_path / "splits.csv"
    pd.DataFrame({"Prospect ID": ["id0"], "split": ["train"]}).to_csv(file, index=False)
    monkeypatch.setattr(split, "path", lambda key: file)
    with pytest.raises(ValueError, match="no saved split"):
        load_splits(leads)
