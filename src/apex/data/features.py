"""Feature engineering and the preprocessing pipeline (step 2.1).

`build_pipeline()` returns one sklearn Pipeline that goes from raw lead rows
to model-ready numbers:

    clean  →  add_features  →  scale numbers + one-hot encode text

Only the last step learns from data (means, scales, which categories are
frequent), so the pipeline must be fitted on the training split only. Rare
categories (below `rare_category_min_share`) and categories never seen in
training share one "infrequent" column, so new leads never break the model.

Run `python -m apex.data.features` (or `make features`) to print the result.
Feature ideas and evidence: docs/eda.md; decisions: docs/features.md.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from apex.config import load_config
from apex.data.clean import clean
from apex.data.load import load_raw


def add_features(df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Add engineered features and drop redundant columns (stateless)."""
    cfg = (config or load_config())["features"]
    df = df.copy()
    visits = df["TotalVisits"]
    df["time_per_visit"] = (df["Total Time Spent on Website"] / visits.where(visits > 0)).fillna(0)
    df["has_web_activity"] = (visits > 0).astype(int)
    return df.drop(columns=[c for c in cfg["drop"] if c in df.columns])


def build_pipeline(config: dict[str, Any] | None = None) -> Pipeline:
    """Raw lead rows (without the target) → model-ready feature matrix."""
    cfg = config or load_config()
    encode = make_column_transformer(
        (StandardScaler(), make_column_selector(dtype_include=np.number)),
        (
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=cfg["data"]["rare_category_min_share"],
                sparse_output=False,
            ),
            make_column_selector(dtype_exclude=np.number),
        ),
        verbose_feature_names_out=False,
    )
    return make_pipeline(
        FunctionTransformer(clean, kw_args={"config": cfg}),
        FunctionTransformer(add_features, kw_args={"config": cfg}),
        encode,
    )


def run() -> pd.DataFrame:
    """Fit the pipeline on all raw leads and print the resulting features."""
    cfg = load_config()
    X = load_raw().drop(columns=[cfg["data"]["target"]])
    pipeline = build_pipeline(cfg)
    features = pd.DataFrame(pipeline.fit_transform(X), columns=pipeline[-1].get_feature_names_out())
    print(f"{len(X):,} leads: {X.shape[1]} raw columns → {features.shape[1]} features\n")
    print("\n".join(features.columns))
    return features


if __name__ == "__main__":
    run()
