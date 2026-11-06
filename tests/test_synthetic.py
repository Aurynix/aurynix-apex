"""The demo's synthetic leads: reproducible, and built without the original dataset."""

import copy

import pandas as pd
import pytest

from apex.config import load_config, path
from apex.monitoring import monitor as monitor_module
from apex.monitoring import seed as seed_module
from apex.monitoring import synthetic
from apex.monitoring.drift import category_shares, psi
from apex.monitoring.monitor import Monitor


@pytest.fixture(scope="module")
def artifacts():
    return synthetic.load_artifacts()


@pytest.fixture(scope="module")
def rows(artifacts):
    return synthetic.generate(*artifacts)


def test_committed_file_is_exactly_what_the_script_generates(rows):
    assert path("demo_seed").read_text(encoding="utf-8") == synthetic.to_json(rows)


def test_generation_does_not_need_the_original_data(artifacts, monkeypatch):
    from apex.data import load, split

    def fail(*args, **kwargs):
        raise AssertionError("must not read the original dataset")

    monkeypatch.setattr(load, "load_raw", fail)
    monkeypatch.setattr(split, "load_splits", fail)
    assert len(synthetic.generate(*artifacts)) == synthetic.SEED_ROWS


def test_generation_is_reproducible_and_depends_on_the_seed(artifacts, rows):
    assert synthetic.generate(*artifacts) == rows
    assert synthetic.generate(*artifacts, seed=synthetic.SEED + 1) != rows


def test_rows_have_only_the_model_fields_and_an_outcome(artifacts, rows):
    _, meta, _ = artifacts
    assert len(rows) == 600
    assert all(set(row) == {"lead", "converted"} for row in rows)
    assert all(list(row["lead"]) == meta["input_fields"] for row in rows)
    assert {row["converted"] for row in rows} == {0, 1}


def test_no_activity_means_no_visits_and_no_time(rows):
    for row in rows:
        visits, time_on_site = (
            row["lead"]["TotalVisits"],
            row["lead"]["Total Time Spent on Website"],
        )
        assert (visits == 0) == (time_on_site == 0)


def test_each_field_follows_its_training_distribution(artifacts, rows):
    model, meta, profile = artifacts
    leads = pd.DataFrame([row["lead"] for row in rows], columns=meta["input_fields"])
    inputs = model[0][:-1].transform(leads)
    for column, reference in profile["categorical"].items():
        current = category_shares(inputs[column], list(reference), "__other__")
        assert psi([*reference.values(), 0.0], list(current.values())) < 0.1, column


def test_outcomes_follow_the_model_probabilities(artifacts, rows):
    model, meta, _ = artifacts
    leads = pd.DataFrame([row["lead"] for row in rows], columns=meta["input_fields"])
    expected = model.predict_proba(leads)[:, 1].mean()
    actual = sum(row["converted"] for row in rows) / len(rows)
    assert abs(actual - expected) < 0.05


def test_the_real_monitoring_report_is_healthy_with_the_real_model(
    artifacts, rows, db, tmp_path, monkeypatch
):
    from apex.api.service import ScoringService

    model, meta, profile = artifacts
    monkeypatch.setattr(monitor_module, "path", lambda key: tmp_path)
    service = ScoringService(model, meta, db)
    monitor = Monitor(model, profile, db, copy.deepcopy(load_config()), meta["performance"]["test"])

    assert seed_module.seed_demo_data(service, monitor, rows) == 600
    report = monitor.history(limit=1)[0]
    assert report["status"] == "ok"
    assert report["retrain_recommended"] is False
    assert report["performance"]["status"] == "ok"
    assert report["n_samples"] == 300
