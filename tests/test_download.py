import sys
import types

import pytest

from apex.data import download


@pytest.fixture
def fake_kaggle(tmp_path, monkeypatch):
    """Fake kagglehub download dir and redirect data/raw/ to a temp folder."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "Lead Scoring.csv").write_text("Converted\n1\n")
    (cache / "Leads Data Dictionary.xlsx").write_bytes(b"xlsx")

    fake = types.SimpleNamespace(dataset_download=lambda slug: str(cache))
    monkeypatch.setitem(sys.modules, "kagglehub", fake)

    raw_dir = tmp_path / "raw"
    monkeypatch.setattr(download, "path", lambda key: raw_dir)
    return cache, raw_dir


def test_download_renames_files_to_config_names(fake_kaggle):
    _, raw_dir = fake_kaggle
    saved = download.download_raw()
    assert sorted(f.name for f in saved) == ["Leads Data Dictionary.xlsx", "Leads.csv"]
    assert (raw_dir / "Leads.csv").read_text() == "Converted\n1\n"


def test_download_fails_if_expected_file_missing(fake_kaggle):
    cache, _ = fake_kaggle
    (cache / "Lead Scoring.csv").unlink()
    with pytest.raises(FileNotFoundError, match="Lead Scoring.csv"):
        download.download_raw()


def test_download_from_url_finds_files_in_nested_zips(tmp_path, monkeypatch):
    """UCI-style zip: the data file sits inside a zip inside the zip."""
    import io
    import zipfile

    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("bank-full.csv", "y\nno\n")
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("bank.zip", inner.getvalue())

    config = {
        "source": {"url": "https://example.org/bank.zip", "files": {"bank-full.csv": "raw_data"}},
        "files": {"raw_data": "bank-full.csv"},
    }
    raw_dir = tmp_path / "raw"
    monkeypatch.setattr(download, "load_config", lambda: config)
    monkeypatch.setattr(download, "path", lambda key: raw_dir)
    monkeypatch.setattr(
        download.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(outer.getvalue())
    )

    saved = download.download_raw()
    assert saved == [raw_dir / "bank-full.csv"]
    assert saved[0].read_text() == "y\nno\n"
