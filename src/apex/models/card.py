"""Model card: a JSON summary of the model for people (step 3.5).

`model_meta.json` is for code (thresholds, feature means). The model card is
for people who need to decide whether to trust and use the model: what it
does, what data and algorithm it uses, how well it works, its risks and
limits. Numbers come from the training run (`meta`); the rest is fixed text
that matches docs/problem_framing.md, docs/models.md, and docs/decisions.md.
"""

from typing import Any


def build_card(meta: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """The model card as a dict, ready for `json.dump`."""
    perf, seg = meta["performance"], meta["segments"]
    test, oof = perf["test"], perf["out_of_fold_all"]
    return {
        "model_details": {
            "name": "Aurynix Apex lead scoring model",
            "version": meta["model_version"],
            "trained_at": meta["trained_at"],
            "owner": "Yazan Al-Sedih (Aurynix)",
            "license": "MIT",
            "task": "Binary classification used as a ranking: probability that a new lead converts",
            "artifacts": ["model.pkl", "model_meta.json", "reference_profile.json"],
            "documentation": [
                "docs/problem_framing.md",
                "docs/data_dictionary.md",
                "docs/data_cleaning.md",
                "docs/features.md",
                "docs/models.md",
                "docs/decisions.md",
            ],
        },
        "intended_use": {
            "primary_use": (
                "Rank new leads so the sales (SDR) team calls the most likely buyers first, "
                "and split them into High / Medium / Low priority by team capacity."
            ),
            "users": "SDR team and sales managers, through Aurynix Pulse (CRM) or the Apex API",
            "prediction_moment": "When the lead enters the queue, before any sales contact",
            "decision_support_only": True,
            "out_of_scope": [
                "Deciding whether a person gets a product, price, or service",
                "Credit, hiring, insurance, or any decision with legal or similar effect",
                "Scoring leads after sales contact (post-contact fields are excluded by design)",
                "Lead data from a different business or form without re-validation",
                "Predicting revenue, deal size, or time to conversion",
            ],
        },
        "data": {
            "source": "Kaggle: amritachatterjee09/lead-scoring-dataset (X Education), version 2",
            "license": "Listed as Unknown on Kaggle; used for research / portfolio only",
            "file_sha256": meta["data"]["sha256"],
            "rows": meta["data"]["rows"],
            "raw_columns": meta["data"]["raw_columns"],
            "target": f"{config['data']['target']} (1 = became a paying customer)",
            "conversion_rate": meta["data"]["conversion_rate"],
            "splits": "Stratified 60 / 20 / 20 train / validation / test, seed 42",
            "final_training_data": "All rows (train + validation + test) after evaluation",
            "cleaning": [
                '"Select" and "unknown" treated as missing',
                "Missing text filled with a 'Missing' category (missingness is informative)",
                "Missing visit counts filled with fixed medians; outliers capped (visits 30)",
                "Constant, near-constant, and ID columns dropped",
                "No rows removed",
            ],
            "leakage_columns_removed": config["data"]["leakage_columns"],
            "known_issues": [
                "Single snapshot per lead, no timestamps: cannot prove when website activity "
                "was recorded",
                "One company, one period: may not represent other businesses",
                "About 22% of leads look identical to another lead (low-information leads)",
            ],
        },
        "features": {
            "input_fields": meta["input_fields"],
            "engineered": ["time_per_visit", "has_web_activity"],
            "model_inputs": len(meta["model_inputs"]),
            "protected_attributes_used": False,
            "location_fields_used": False,
            "notes": (
                "Country and City were dropped (weak signal). Occupation is used and may "
                "correlate with age or employment status."
            ),
        },
        "algorithm": {
            "type": "Logistic Regression (scikit-learn)",
            "pipeline": [
                "clean (stateless rules from config.json)",
                "add_features (time_per_visit, has_web_activity, column selection)",
                "StandardScaler on numbers + OneHotEncoder on text (rare / unseen → infrequent)",
                "LogisticRegression",
            ],
            "hyperparameters": meta["hyperparameters"],
            "selection": (
                "Tied with LightGBM / XGBoost on 5-fold CV (PR-AUC 0.818); chosen for "
                "simplicity, explainability, and honest probabilities (ADR-003, ADR-004)"
            ),
            "calibration": "Raw probabilities, no extra calibration (ADR-005)",
            "explainability": (
                "Exact linear SHAP: weight × (value − average value), grouped into readable "
                "reasons per lead"
            ),
            "segmentation": {
                "method": "Capacity-based: top 20% High, next 30% Medium, rest Low",
                "thresholds": seg["thresholds"],
            },
        },
        "performance": {
            "test_set": {
                "description": (
                    "Held-out 20% (1,848 leads), used once; model fitted on train + validation"
                ),
                **test,
            },
            "out_of_fold_all_rows": {
                "description": "5-fold out-of-fold predictions on all rows (final model setup)",
                **oof,
            },
            "segments_out_of_fold": seg["table"],
            "no_skill_baseline": {"pr_auc": meta["data"]["conversion_rate"], "roc_auc": 0.5},
            "expected_pr_auc_on_new_leads": "about 0.80 ± 0.03",
            "targets_met": {
                "recall_top20_at_least_40pct": test["recall_top20"] >= 0.40,
                "recall_top50_at_least_80pct": test["recall_top50"] >= 0.80,
                "ece_at_most_0.05": oof["ece"] <= 0.05,
                "no_leakage_alarm_roc_auc_below_0.95": test["roc_auc"] < 0.95,
            },
        },
        "risk_rating": {
            "overall": "Low",
            "rationale": (
                "Decision support for sales prioritization only: a person decides whom to "
                "call, Low leads still receive automated emails, no protected or location "
                "attributes are used, and no one is denied a product or service. Model risk "
                "(data representativeness, drift) is the main concern and is monitored."
            ),
            "regulatory_note": (
                "Likely a minimal-risk use case (marketing / sales prioritization). This is "
                "an engineering assessment, not a legal one."
            ),
            "risks": [
                {
                    "risk": "Data not representative (one company, one period)",
                    "likelihood": "High",
                    "impact": "Medium",
                    "rating": "Medium",
                    "mitigation": "Re-validate on each client's data; drift monitoring (PSI)",
                },
                {
                    "risk": "Hidden leakage: Lead Add Form leads (92.5% convert) may be "
                    "added by sales after a good call",
                    "likelihood": "Medium",
                    "impact": "Medium",
                    "rating": "Medium",
                    "mitigation": "Confirm the Lead Add Form process with the business; "
                    "watch its share and conversion in production",
                },
                {
                    "risk": "Data and score drift (new channels, forms, campaigns)",
                    "likelihood": "Medium",
                    "impact": "Medium",
                    "rating": "Medium",
                    "mitigation": "PSI on features and scores; warning at 0.10, drift at 0.25",
                },
                {
                    "risk": "Feedback loop: only High leads get called, so labels favor them",
                    "likelihood": "Medium",
                    "impact": "Low",
                    "rating": "Low",
                    "mitigation": "Keep a small random-contact sample when outcomes are logged",
                },
                {
                    "risk": "Calibration bias: 0.5–0.6 scores convert ~10 pts more, "
                    "0.8–0.9 ~10 pts less than shown",
                    "likelihood": "High",
                    "impact": "Low",
                    "rating": "Low",
                    "mitigation": "Segments use rank, not probability; switch to isotonic "
                    "if exact percentages matter",
                },
                {
                    "risk": "Indirect bias through occupation (e.g. students, unemployed "
                    "ranked lower)",
                    "likelihood": "Medium",
                    "impact": "Low",
                    "rating": "Low",
                    "mitigation": "Human decides; Low leads still nurtured; review "
                    "segment mix by occupation",
                },
            ],
        },
        "limitations": [
            "Trained on 9,240 leads from one online-education company",
            "Website activity may include visits after first contact (no timestamps)",
            "Random split, not time-based: future performance may be lower if behavior changes",
            "About 15% of buyers fall in the Low segment",
            "Probabilities are approximate in the 0.5–0.9 range (see risks)",
        ],
        "monitoring": {
            "reference_profile": "reference_profile.json",
            "metric": "Population Stability Index per feature and on the score",
            "psi_warning": config["monitoring"]["psi_warning"],
            "psi_drift": config["monitoring"]["psi_drift"],
            "retrain_when": [
                "Score or key feature PSI ≥ 0.25",
                "New lead sources or forms appear",
                "Real conversion outcomes show a drop in precision of the High segment",
            ],
        },
    }
