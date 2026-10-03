"""Clean raw leads.

Cleaning is **stateless**: every rule comes from config.json, nothing is learned
from the data. The same function can therefore run on the full training file
and on a single lead at prediction time without leaking information.
Anything that must be learned from data (outlier caps, rare-category grouping,
imputation) belongs in features.py and is fitted on the training split only.

Decisions behind each rule are documented in docs/data_quality.md.

Run `python -m apex.data.clean` (or `make preprocess`) to write
data/interim/leads_clean.parquet (name set in config.json → files).
"""

from typing import Any

import pandas as pd

from apex.config import load_config, path
from apex.data.load import load_raw

YES_NO = {"Yes": 1, "No": 0}


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Strip text values and collapse repeated inner whitespace."""
    df = df.copy()
    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].str.strip().str.replace(r"\s+", " ", regex=True)
    return df


def replace_placeholders(df: pd.DataFrame, placeholders: list[str]) -> pd.DataFrame:
    """Turn placeholder values such as "Select" (no option chosen) into missing."""
    df = df.copy()
    text_cols = df.select_dtypes(exclude="number").columns
    df[text_cols] = df[text_cols].mask(df[text_cols].isin(placeholders))
    return df


def apply_aliases(df: pd.DataFrame, aliases: dict[str, dict[str, str]]) -> pd.DataFrame:
    """Map spelling variants to one label, e.g. "google" → "Google"."""
    df = df.copy()
    for col, mapping in aliases.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)
    return df


def encode_binary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Encode Yes/No columns as 1/0; anything else is an error."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        unexpected = set(df[col].dropna().unique()) - set(YES_NO)
        if unexpected:
            raise ValueError(f"{col!r} has non Yes/No values: {sorted(unexpected)}")
        df[col] = df[col].map(YES_NO).astype("Int8")
    return df


def validate(df: pd.DataFrame, target: str | None, numeric_columns: list[str]) -> None:
    """Fail loudly on values that should be impossible."""
    if target is not None and target in df.columns:
        bad = set(df[target].dropna().unique()) - {0, 1}
        if bad or df[target].isna().any():
            raise ValueError(f"Target {target!r} must be 0/1 with no missing values.")
    for col in numeric_columns:
        if col in df.columns and (df[col] < 0).any():
            raise ValueError(f"{col!r} has negative values.")


def clean(df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Apply all cleaning rules from config.json → data.

    Works with or without the target column, so it can also clean new leads.
    """
    cfg = (config or load_config())["data"]

    df = normalize_text(df)
    df = replace_placeholders(df, cfg["missing_placeholders"])
    df = apply_aliases(df, cfg["category_aliases"])
    df = encode_binary(df, cfg["binary_columns"])

    to_drop = cfg["id_columns"] + cfg["drop_columns"] + cfg["leakage_columns"]
    df = df.drop(columns=[c for c in to_drop if c in df.columns])

    validate(df, cfg["target"], cfg["numeric_columns"])
    return df


def run() -> pd.DataFrame:
    """Load the raw file, clean it, and save it to data/interim/."""
    df = clean(load_raw())
    out = path("interim_dir") / load_config()["files"]["interim_leads"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"saved: {out} ({df.shape[0]} rows × {df.shape[1]} columns)")
    return df


if __name__ == "__main__":
    run()
