import mlflow
import numpy as np
import pandas as pd

from apex.models.train import fit_and_log, make_baselines, make_model


def _leads(n, seed):
    rng = np.random.default_rng(seed)
    time = rng.integers(0, 2000, n)
    return pd.DataFrame(
        {
            "Prospect ID": [f"{seed}-{i}" for i in range(n)],
            "Lead Origin": rng.choice(["API", "Landing Page Submission"], n),
            "TotalVisits": rng.integers(1, 6, n).astype(float),
            "Total Time Spent on Website": time,
            "Page Views Per Visit": 2.0,
            "Converted": (time > 1000).astype(int),
        }
    )


def test_make_model_uses_logistic_regression_with_params():
    model = make_model(C=0.5)
    assert type(model[-1]).__name__ == "LogisticRegression"
    assert model[-1].C == 0.5


def test_baselines_are_logged_and_lr_beats_no_skill(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test")
    parts = {"train": _leads(300, 0), "val": _leads(100, 1)}

    scores = {
        name: fit_and_log(name, model, parts, "Converted")["pr_auc"]
        for name, model in make_baselines().items()
    }
    assert scores["logistic_regression"] > scores["no_skill"] + 0.3
    assert len(mlflow.search_runs(experiment_names=["test"])) == 2
