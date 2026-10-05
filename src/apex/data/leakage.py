"""Leakage check: does a quick model score much higher with suspect columns?

A throwaway logistic regression is scored (5-fold CV, PR-AUC) on the cleaned
data with and without each group of suspect columns. A big jump when a group
is added is evidence that it carries information from after the sales contact.
The decisions are recorded in docs/data_dictionary.md and docs/decisions.md.

Run `python -m apex.data.leakage` (or `make leakage`) to print the comparison.
"""

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from apex.config import load_config
from apex.data.clean import clean
from apex.data.load import load_raw

# Columns suspected of being filled in after the sales contact (step 1.4).
SUSPECTS = {
    "Tags": ["Tags"],
    "Lead Quality": ["Lead Quality"],
    "Last Activity": ["Last Activity", "Last Notable Activity"],
    "Asymmetrique": [
        "Asymmetrique Activity Index",
        "Asymmetrique Profile Index",
        "Asymmetrique Activity Score",
        "Asymmetrique Profile Score",
    ],
    "Lead Profile": ["Lead Profile"],
}


def cv_pr_auc(df: pd.DataFrame, target: str, seed: int = 42) -> float:
    """Mean PR-AUC of a simple logistic regression over 5 CV folds."""
    X, y = df.drop(columns=[target]), df[target]
    text_cols = X.select_dtypes(exclude="number").columns
    num_cols = X.select_dtypes(include="number").columns

    model = make_pipeline(
        make_column_transformer(
            (
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="Missing"),
                    OneHotEncoder(handle_unknown="ignore", min_frequency=10),
                ),
                text_cols,
            ),
            (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
        ),
        LogisticRegression(max_iter=1000, random_state=seed),
    )
    return cross_val_score(model, X, y, cv=5, scoring="average_precision").mean()


def compare(df: pd.DataFrame, target: str, suspects: dict[str, list[str]]) -> pd.DataFrame:
    """PR-AUC of the base model (no suspects) and of base + each suspect group."""
    all_suspects = [c for cols in suspects.values() for c in cols]
    base = df.drop(columns=all_suspects)
    base_score = cv_pr_auc(base, target)

    rows = [{"columns": "base (no suspects)", "pr_auc": base_score, "gain": 0.0}]
    for name, cols in suspects.items():
        score = cv_pr_auc(base.join(df[cols]), target)
        rows.append({"columns": f"+ {name}", "pr_auc": score, "gain": score - base_score})
    return pd.DataFrame(rows).round(3)


def run() -> pd.DataFrame:
    """Clean the raw file but keep the suspects, then print the comparison."""
    cfg = load_config()
    target = cfg["data"]["target"]
    keep_suspects = {**cfg, "data": {**cfg["data"], "leakage_columns": []}}
    df = clean(load_raw(), keep_suspects)

    result = compare(df, target, SUSPECTS)
    print(f"PR-AUC of random guessing = conversion rate = {df[target].mean():.3f}")
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    run()
