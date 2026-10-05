"""Train / validation / test split (step 2.2).

The split is stratified on the target and seeded by `config.json → split`.
It is saved once as `Prospect ID → split` (data/splits.csv), so it stays fixed
even if a library changes how it shuffles.

Rule: the test set is not touched until the final evaluation (step 3.1).
`load_splits()` therefore returns train and validation only, unless
`include_test=True` is passed on purpose.

Run `python -m apex.data.split` (or `make split`) to create the file.
"""

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from apex.config import load_config, path
from apex.data.load import load_raw


def make_splits(df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.Series:
    """Label every row "train", "val", or "test", stratified on the target.

    `test_size` and `val_size` are shares of all rows (e.g. 0.2 + 0.2 → 60/20/20).
    """
    cfg = config or load_config()
    target, s = cfg["data"]["target"], cfg["split"]

    rest, test = train_test_split(
        df.index, test_size=s["test_size"], stratify=df[target], random_state=s["random_state"]
    )
    train, val = train_test_split(
        rest,
        test_size=s["val_size"] / (1 - s["test_size"]),
        stratify=df.loc[rest, target],
        random_state=s["random_state"],
    )
    labels = pd.Series("train", index=df.index, name="split")
    labels[val], labels[test] = "val", "test"
    return labels


def summary(df: pd.DataFrame, labels: pd.Series, target: str) -> pd.DataFrame:
    """Rows, share of rows, and conversion rate per split."""
    out = df.groupby(labels)[target].agg(rows="size", conversion_rate="mean")
    out.insert(1, "share", out["rows"] / len(df))
    return out.loc[["train", "val", "test"]].round(4)


def load_splits(
    df: pd.DataFrame | None = None, include_test: bool = False
) -> dict[str, pd.DataFrame]:
    """Return the saved splits as DataFrames: {"train", "val"} (+ "test" if asked)."""
    cfg = load_config()
    id_col = cfg["data"]["id_columns"][0]
    file = path("splits")
    if not file.exists():
        raise FileNotFoundError(f"{file} not found. Run `make split` first.")

    df = load_raw() if df is None else df
    labels = df[id_col].map(pd.read_csv(file).set_index(id_col)["split"])
    if labels.isna().any():
        raise ValueError(f"{labels.isna().sum()} rows have no saved split; re-run `make split`.")

    names = ["train", "val", "test"] if include_test else ["train", "val"]
    return {name: df[labels == name] for name in names}


def run() -> pd.Series:
    """Create the split for the raw leads, save it, and print class ratios."""
    cfg = load_config()
    df = load_raw()
    labels = make_splits(df, cfg)

    file = path("splits")
    id_col = cfg["data"]["id_columns"][0]
    pd.DataFrame({id_col: df[id_col], "split": labels}).to_csv(file, index=False)

    print(summary(df, labels, cfg["data"]["target"]).to_string())
    print(f"\nsaved: {file}")
    return labels


if __name__ == "__main__":
    run()
