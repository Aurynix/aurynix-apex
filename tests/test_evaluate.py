import pytest

from apex.models.evaluate import capture_at, evaluate


def test_capture_at_perfect_ranking_finds_all_buyers_first():
    y = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    score = [0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    assert capture_at(y, score, 0.2) == 1.0


def test_capture_at_worst_ranking_finds_none():
    y = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    score = [0.1, 0.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    assert capture_at(y, score, 0.2) == 0.0


def test_evaluate_returns_all_metrics_and_lift():
    y = [1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    score = [0.9, 0.2, 0.8, 0.3, 0.1, 0.7, 0.2, 0.1, 0.4, 0.3]
    metrics = evaluate(y, score)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["capture_top20"] == pytest.approx(2 / 3, abs=1e-4)
    assert metrics["lift_top20"] == pytest.approx(metrics["capture_top20"] / 0.2, abs=1e-3)
    assert 0 <= metrics["brier"] <= 1
