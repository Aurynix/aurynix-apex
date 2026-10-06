"""Load raw datasets from data/raw/.

Run `python -m apex.data.load` (or `make data-info`) to print the facts that
go into the data dictionary: shape, file hash, and target balance.

Dataset differences come from the config (`config.json`, or another file via
`APEX_CONFIG`): the CSV separator (`data.csv_sep`), how target labels map to
1/0 (`data.target_values`), and a row id when the file has none (`data.id_columns`).
"""

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from apex.config import load_config, path


def raw_data_path() -> Path:
    """Return the configured path of the raw data file."""
    return path("raw_dir") / load_config()["files"]["raw_data"]


def load_raw(file: Path | None = None) -> pd.DataFrame:
    """Read the raw CSV (no cleaning), map the target to 1/0, and add a row id if needed."""
    cfg = load_config()["data"]
    file = file or raw_data_path()
    if not file.exists():
        raise FileNotFoundError(f"{file} not found. Run `make data-download` first.")

    df = pd.read_csv(file, sep=cfg.get("csv_sep", ","))

    target = cfg["target"]
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not found in {file.name}.")
    if "target_values" in cfg:
        mapped = df[target].map(cfg["target_values"])
        if mapped.isna().any():
            unexpected = sorted(set(df[target]) - set(cfg["target_values"]))
            raise ValueError(f"Unexpected {target!r} values: {unexpected}")
        df[target] = mapped.astype(int)

    id_col = cfg["id_columns"][0] if cfg["id_columns"] else None
    if id_col and id_col not in df.columns:
        df.insert(0, id_col, range(1, len(df) + 1))  # stable while the file (hash) is the same
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
    file = file or raw_data_path()
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
