"""Clean raw leads.

Cleaning is **stateless**: every rule and every value (fill values, caps) is
fixed in config.json, nothing is learned at run time. The same function can
therefore run on the full training file and on a single lead at prediction time.

Steps, in order: normalize text → placeholders to missing → merge spellings →
Yes/No to 1/0 → drop columns → group rare values → validate → fill missing →
fix types → cap outliers. Each step is explained in docs/data_cleaning.md.

There is no saved copy of the cleaned data: cleaning takes under a second, so
every stage calls `clean(load_raw())` and always gets the current rules.
"""

from typing import Any

import pandas as pd

from apex.config import load_config

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


def group_other(df: pd.DataFrame, keep: dict[str, list[str]]) -> pd.DataFrame:
    """Keep the listed values of a column and turn every other value into "Other"."""
    df = df.copy()
    for col, values in keep.items():
        if col in df.columns:
            df[col] = df[col].where(df[col].isna() | df[col].isin(values), "Other")
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
    visits, views = "TotalVisits", "Page Views Per Visit"
    if {visits, views} <= set(df.columns) and ((df[visits] == 0) & (df[views] > 0)).any():
        raise ValueError("Page views recorded for a lead with 0 visits.")


def fill_missing(df: pd.DataFrame, numeric_fill: dict[str, float], label: str) -> pd.DataFrame:
    """Fill numbers with fixed values and text with a "Missing" category."""
    df = df.copy()
    df = df.fillna({col: v for col, v in numeric_fill.items() if col in df.columns})
    text_cols = df.select_dtypes(exclude="number").columns
    df[text_cols] = df[text_cols].fillna(label)
    return df


def cap_outliers(df: pd.DataFrame, caps: dict[str, float]) -> pd.DataFrame:
    """Clip extreme values to a fixed upper limit; rows are kept."""
    df = df.copy()
    for col, cap in caps.items():
        if col in df.columns:
            df[col] = df[col].clip(upper=cap)
    return df


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
    df = group_other(df, cfg["group_other"])

    validate(df, cfg["target"], cfg["numeric_columns"])

    df = fill_missing(df, cfg["numeric_fill"], cfg["missing_label"])
    for col in cfg["integer_columns"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    df = cap_outliers(df, cfg["caps"])
    return df
