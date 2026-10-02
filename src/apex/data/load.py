"""Load raw datasets from data/raw/.

Run `python -m apex.data.load` (or `make data-info`) to print the facts that
go into docs/data_dictionary.md: shape, file hash, and target balance.
"""

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from apex.config import load_config, path


def raw_leads_path() -> Path:
    """Return the configured path of the raw leads file."""
    return path("raw_dir") / load_config()["files"]["raw_leads"]


def load_raw(file: Path | None = None) -> pd.DataFrame:
    """Read the raw leads CSV as-is (no cleaning) and check the target exists."""
    file = file or raw_leads_path()
    if not file.exists():
        raise FileNotFoundError(
            f"{file} not found. Download Leads.csv from Kaggle "
            '("Lead Scoring X Education") and place it in data/raw/.'
        )

    df = pd.read_csv(file)

    target = load_config()["data"]["target"]
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not found in {file.name}.")
    return df


def file_sha256(file: Path) -> str:
    """Return the SHA-256 hash of a file, used to pin the exact dataset version."""
    digest = hashlib.sha256()
    with file.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_raw(file: Path | None = None) -> dict[str, Any]:
    """Summarize the raw file for the data dictionary."""
    file = file or raw_leads_path()
    df = load_raw(file)
    target = load_config()["data"]["target"]
    return {
        "file": file.name,
        "sha256": file_sha256(file),
        "size_mb": round(file.stat().st_size / 1e6, 2),
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "target_rate": round(float(df[target].mean()), 4),
        "columns": list(df.columns),
    }


if __name__ == "__main__":
    info = describe_raw()
    for key, value in info.items():
        if key == "columns":
            print(f"columns ({len(value)}):")
            for col in value:
                print(f"  - {col}")
        else:
            print(f"{key}: {value}")
