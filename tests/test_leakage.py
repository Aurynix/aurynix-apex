import numpy as np
import pandas as pd

from apex.data.leakage import compare


def test_compare_flags_a_column_that_copies_the_target():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    df = pd.DataFrame(
        {
            "Converted": y,
            "noise": rng.normal(size=200),
            "source": rng.choice(["a", "b"], 200),
            "leaky": np.where(y == 1, "won", "lost"),
        }
    )

    result = compare(df, "Converted", {"leaky": ["leaky"]}).set_index("columns")

    assert result.loc["base (no suspects)", "pr_auc"] < 0.7
    assert result.loc["+ leaky", "pr_auc"] > 0.95
    assert result.loc["+ leaky", "gain"] > 0.3
