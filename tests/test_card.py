from apex.config import load_config
from apex.models.card import build_card

METRICS = {"pr_auc": 0.79, "roc_auc": 0.85, "ece": 0.03, "recall_top20": 0.43, "recall_top50": 0.85}
META = {
    "model_version": "apex-test",
    "trained_at": "2026-10-06T00:00:00+00:00",
    "data": {"sha256": "abc", "rows": 100, "raw_columns": 37, "conversion_rate": 0.385},
    "input_fields": ["Lead Origin"],
    "model_inputs": ["a", "b"],
    "hyperparameters": {"C": 1.0},
    "segments": {"thresholds": {"high": 0.75, "medium": 0.27}, "table": {}},
    "performance": {"test": METRICS, "out_of_fold_all": METRICS},
}


def test_card_has_all_sections_and_a_risk_rating():
    card = build_card(META, load_config())
    for section in ("model_details", "intended_use", "data", "features", "algorithm",
                    "performance", "risk_rating", "limitations", "monitoring"):  # fmt: skip
        assert section in card
    assert card["risk_rating"]["overall"] in {"Low", "Medium", "High"}
    assert all({"risk", "likelihood", "impact", "rating", "mitigation"} <= set(r)
               for r in card["risk_rating"]["risks"])  # fmt: skip


def test_card_checks_targets_from_the_metrics():
    targets = build_card(META, load_config())["performance"]["targets_met"]
    assert all(targets.values())
    weak = {
        **META,
        "performance": {"test": {**METRICS, "recall_top20": 0.3}, "out_of_fold_all": METRICS},
    }
    assert (
        build_card(weak, load_config())["performance"]["targets_met"]["recall_top20_at_least_40pct"]
        is False
    )
