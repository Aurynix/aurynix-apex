import hashlib
from pathlib import Path

import pandas as pd
import pytest

from apex.data.load import describe_raw, file_sha256, load_raw, raw_data_path


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


def test_raw_data_path_uses_config():
    assert raw_data_path().parts[-3:] == ("data", "raw", "Leads.csv")


def test_load_raw_returns_data_unchanged(leads_csv):
    df = load_raw(leads_csv)
    assert df.shape == (4, 4)
    # Cleaning happens in clean.py; placeholders must still be present here.
    assert "Select" in df["Lead Source"].tolist()


def test_load_raw_missing_file_has_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="make data-download"):
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


def test_load_raw_with_bank_style_config(tmp_path, monkeypatch):
    """Semicolon CSV, yes/no target, and no id column (UCI Bank Marketing)."""
    import json

    from apex.data import load

    bank = json.loads(Path("config_bank.json").read_text())
    monkeypatch.setattr(load, "load_config", lambda: bank)
    file = tmp_path / "bank.csv"
    file.write_text('"age";"job";"y"\n30;"admin.";"no"\n41;"unknown";"yes"\n')

    df = load.load_raw(file)
    assert df.columns.tolist() == ["row_id", "age", "job", "y"]
    assert df["row_id"].tolist() == [1, 2]
    assert df["y"].tolist() == [0, 1]


def test_load_raw_rejects_unexpected_target_labels(tmp_path, monkeypatch):
    import json

    from apex.data import load

    bank = json.loads(Path("config_bank.json").read_text())
    monkeypatch.setattr(load, "load_config", lambda: bank)
    file = tmp_path / "bank.csv"
    file.write_text('"age";"y"\n30;"maybe"\n')
    with pytest.raises(ValueError, match="maybe"):
        load.load_raw(file)
