import joblib
import numpy as np
import pandas as pd
import pytest

from apex.models import retrain
from apex.models.train import make_model


def leads(n, seed, rule):
    """Synthetic raw leads; `rule` decides who converts."""
    rng = np.random.default_rng(seed)
    time = rng.integers(1, 2000, n)
    df = pd.DataFrame(
        {
            "Prospect ID": [f"{seed}-{i}" for i in range(n)],
            "Lead Origin": rng.choice(["API", "Landing Page Submission"], n),
            "TotalVisits": rng.integers(1, 6, n).astype(float),
            "Total Time Spent on Website": time,
        }
    )
    df["Converted"] = rule(df).astype(int)
    return df


OLD = lambda d: d["Total Time Spent on Website"] > 1000  # noqa: E731
NEW = lambda d: d["Lead Origin"] == "API"  # behavior changed  # noqa: E731


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A saved champion trained on the OLD rule, in a temp models folder."""
    models = tmp_path / "models"
    models.mkdir()
    champion = make_model().fit(*split(leads(400, 0, OLD)))
    joblib.dump(champion, models / "model.pkl")
    (models / "model_meta.json").write_text('{"trained_at": "2026-10-01T00:00:00+00:00"}')

    paths = {"models_dir": models, "database": tmp_path / "none.db"}
    monkeypatch.setattr(retrain, "path", lambda key: paths[key])
    calls = []
    monkeypatch.setattr(retrain, "run_final", lambda: calls.append("final"))
    monkeypatch.setattr(retrain.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(retrain.mlflow, "set_experiment", lambda name: None)
    monkeypatch.setattr(retrain.mlflow, "start_run", lambda run_name: _NoRun())
    monkeypatch.setattr(retrain.mlflow, "log_params", lambda p: None)
    monkeypatch.setattr(retrain.mlflow, "log_metrics", lambda m: None)
    return models, calls, monkeypatch


class _NoRun:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def split(df):
    return df.drop(columns=["Converted"]), df["Converted"]


def test_new_behavior_promotes_the_challenger_and_archives_the_old_model(workspace):
    models, calls, monkeypatch = workspace
    monkeypatch.setattr(retrain, "load_raw", lambda: leads(600, 1, NEW))
    result = retrain.run()
    assert result["decision"] == "promoted"
    assert result["scores"]["challenger"]["pr_auc"] > result["scores"]["champion"]["pr_auc"]
    assert calls == ["final"]  # refitted on all data
    assert (models / "archive" / "2026-10-01T00-00-00+00-00" / "model.pkl").exists()


def test_a_model_that_is_not_better_enough_is_not_promoted(workspace):
    import copy

    models, calls, monkeypatch = workspace
    cfg = copy.deepcopy(retrain.load_config())
    cfg["retrain"]["min_gain"] = 0.05  # same behavior → equal scores, below the required gain
    monkeypatch.setattr(retrain, "load_config", lambda: cfg)
    monkeypatch.setattr(retrain, "load_raw", lambda: leads(600, 1, OLD))
    result = retrain.run()
    assert result["decision"] == "kept"
    assert calls == []
    assert not (models / "archive").exists()


def test_real_outcomes_are_used_when_there_are_enough(workspace):
    _, _, monkeypatch = workspace
    real = leads(300, 2, NEW)
    rows = [(r.drop("Converted").to_dict(), int(r["Converted"])) for _, r in real.iterrows()]
    monkeypatch.setattr(retrain, "load_raw", lambda: leads(600, 1, NEW))
    monkeypatch.setattr(retrain, "connect", lambda path: None)
    monkeypatch.setattr(retrain, "read_outcome_leads", lambda conn: rows)
    monkeypatch.setattr(type(retrain.path("database")), "exists", lambda self: True)
    _, held, source = retrain.holdout(retrain.load_config())
    assert source == "300 real outcomes"
    assert len(held) == 300
