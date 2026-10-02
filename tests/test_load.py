import hashlib

import pandas as pd
import pytest

from apex.data.load import describe_raw, file_sha256, load_raw, raw_leads_path


@pytest.fixture
def leads_csv(tmp_path):
    file = tmp_path / "Leads.csv"
    pd.DataFrame(
        {
            "Prospect ID": ["a", "b", "c", "d"],
            "Lead Source": ["Google", "Select", "Olark Chat", "Google"],
            "TotalVisits": [5, 0, 2, None],
            "Converted": [1, 0, 0, 1],
        }
    ).to_csv(file, index=False)
    return file


def test_raw_leads_path_uses_config():
    assert raw_leads_path().parts[-3:] == ("data", "raw", "Leads.csv")


def test_load_raw_returns_data_unchanged(leads_csv):
    df = load_raw(leads_csv)
    assert df.shape == (4, 4)
    # Cleaning happens in clean.py; placeholders must still be present here.
    assert "Select" in df["Lead Source"].tolist()


def test_load_raw_missing_file_has_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        load_raw(tmp_path / "Leads.csv")


def test_load_raw_requires_target(tmp_path):
    file = tmp_path / "Leads.csv"
    pd.DataFrame({"Lead Source": ["Google"]}).to_csv(file, index=False)
    with pytest.raises(ValueError, match="Converted"):
        load_raw(file)


def test_file_sha256_matches_hashlib(leads_csv):
    assert file_sha256(leads_csv) == hashlib.sha256(leads_csv.read_bytes()).hexdigest()


def test_describe_raw(leads_csv):
    info = describe_raw(leads_csv)
    assert info["n_rows"] == 4
    assert info["n_columns"] == 4
    assert info["target_rate"] == 0.5
    assert info["columns"][0] == "Prospect ID"
