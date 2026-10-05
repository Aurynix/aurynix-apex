import pytest

from apex.models.evaluate import evaluate, top_share

Y = [1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
SCORE = [0.9, 0.2, 0.8, 0.3, 0.1, 0.4, 0.6, 0.1, 0.4, 0.3]


def test_top_share_marks_the_highest_scores():
    assert top_share([0.1, 0.9, 0.5, 0.7], 0.5).tolist() == [0, 1, 0, 1]


def test_precision_and_recall_at_top_20():
    # top 2 of 10 leads are both buyers: precision 1, recall 2 of 3 buyers
    metrics = evaluate(Y, SCORE)
    assert metrics["precision_top20"] == 1.0
    assert metrics["recall_top20"] == pytest.approx(2 / 3, abs=1e-4)
    assert metrics["lift_top20"] == pytest.approx(metrics["recall_top20"] / 0.2, abs=1e-3)


def test_precision_and_recall_at_half():
    # scores >= 0.5: leads 0, 2, 6 → 2 buyers of 3 called, 2 of 3 buyers found
    metrics = evaluate(Y, SCORE)
    assert metrics["precision_at_0.5"] == pytest.approx(2 / 3, abs=1e-4)
    assert metrics["recall_at_0.5"] == pytest.approx(2 / 3, abs=1e-4)


def test_evaluate_ranking_metrics():
    metrics = evaluate([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert 0 <= metrics["brier"] <= 1


def test_plot_test_report_writes_a_figure(tmp_path):
    from apex.models.evaluate import plot_test_report

    file = tmp_path / "report.png"
    plot_test_report(Y, SCORE, file, "test")
    assert file.stat().st_size > 0


def test_calibration_table_and_ece_for_honest_scores():
    from apex.models.evaluate import calibration_table, expected_calibration_error

    # bin 0.2: predicted 0.25, 1 of 4 convert; bin 0.7: predicted 0.75, 3 of 4 convert
    y = [1, 0, 0, 0, 1, 1, 1, 0]
    score = [0.25] * 4 + [0.75] * 4
    table = calibration_table(y, score)
    assert table["actual"].tolist() == [0.25, 0.75]
    assert table["leads"].tolist() == [4, 4]
    assert expected_calibration_error(y, score) == pytest.approx(0.0)


def test_ece_measures_overconfidence():
    from apex.models.evaluate import expected_calibration_error

    # predicts 0.9 for everyone, half convert → gap 0.4
    assert expected_calibration_error([1, 0] * 5, [0.9] * 10) == pytest.approx(0.4)
