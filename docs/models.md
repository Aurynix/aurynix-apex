# Models

> Stages 7+ (from step 2.3). Code: [`src/apex/models/`](../src/apex/models/). Runs are tracked in MLflow (`make mlflow-ui` → http://localhost:5000, experiment `apex-lead-scoring`).

**Model:** Logistic Regression on the feature pipeline from [features.md](features.md) ([ADR-003](decisions.md)).

---

## Metrics

From [problem_framing.md](problem_framing.md) section 5, in [`evaluate.py`](../src/apex/models/evaluate.py):

| Metric | Meaning | No-skill value |
|---|---|---|
| **PR-AUC** (primary) | How well buyers are ranked above non-buyers, focused on the buyers | = conversion rate (0.385) |
| ROC-AUC | Ranking quality overall | 0.5 |
| Brier | Mean squared error of the probabilities (lower is better) | p(1 − p) ≈ 0.237 |
| **Precision top 20%** | Of the leads called (the 20% highest-scored, the **High** segment), the share who buy | = conversion rate (38.5%) |
| **Recall top 20%** | Of **all** buyers, the share found in the top 20% | 20% |
| Lift top 20% | Recall ÷ 20%: how many times better than calling at random | 1.0 |
| Precision / recall / lift top 50% | Same for High + Medium | 38.5% / 50% / 1.0 |
| Precision / recall at 0.5 | Same, but "called" = probability ≥ 0.5 (standard classifier view) | — |

**Why top k%, not only 0.5?** The sales team calls leads from the top of the list until it runs out of time, so the cut-off is a share of leads, not a probability. The 0.5 view is shown for comparison.

## Step 2.3: Baselines

```bash
make split       # once, if data/splits.csv does not exist
make baselines   # fit on train (5,544), score on validation (1,848), log to MLflow
```

**Ranking quality**

| Model | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| No skill (predicts 38.5% for everyone) | 0.385 | 0.500 | 0.237 |
| **Logistic Regression** (default settings) | **0.840** | **0.887** | **0.130** |

**Precision and recall**

| Model | Cut-off | Leads called | Precision | Recall | Lift |
|---|---|---|---|---|---|
| No skill | top 20% | 370 | 36.8% | 19.1% | 0.96 |
| No skill | top 50% | 924 | 40.2% | 52.1% | 1.04 |
| **Logistic Regression** | **top 20%** | 370 | **88.9%** | **46.2%** | **2.31** |
| **Logistic Regression** | **top 50%** | 924 | **67.3%** | **87.4%** | **1.75** |
| **Logistic Regression** | probability ≥ 0.5 | 617 | 81.2% | 70.4% | — |

How to read it for Logistic Regression:
- **Top 20%:** about **9 in 10 calls reach a buyer** (vs. fewer than 4 in 10 at random), and these calls find **46% of all buyers**.
- **Top 50%:** 2 in 3 calls reach a buyer, and they find **87% of all buyers**.
- **Probability ≥ 0.5:** 81% of the flagged leads buy, and they are 70% of all buyers. The cut-off changes the trade-off: call fewer leads → higher precision, lower recall.

Against the targets in [problem_framing.md](problem_framing.md):

| Target | Required | Logistic Regression | |
|---|---|---|---|
| Recall in top 20% | ≥ 40% (ceiling ≈ 52%) | 46.2% | ✅ |
| Recall in top 50% | ≥ 80% | 87.4% | ✅ |
| PR-AUC | far above no skill (0.385) | 0.840 | ✅ |
| Brier | below no skill (0.237) | 0.130 | ✅ |
| Leakage alarm | ROC-AUC < 0.95 | 0.887 | ✅ no alarm |

The top 20% can hold at most 20 / 38.5 ≈ 52% of buyers, so 46.2% is about 89% of the best possible result.

The no-skill scores are all equal, so its ranking is arbitrary; its precision and recall values are what random calling gives, up to noise.

**These numbers are the bar to beat.** The test set is untouched; its single use is step 3.1.

## Step 2.4: Cross-validation and feature selection

One validation score moves by about ±0.01 PR-AUC depending on which leads land in it, so choices are made on **5-fold stratified CV of the train split** (mean ± std), and the validation split confirms them.

```bash
make cv   # all features vs. selected features, logged to MLflow
```

Each feature idea from [eda.md](eda.md), tested alone (CV PR-AUC; all features = 0.8176 ± 0.011):

| Change | CV PR-AUC | Effect |
|---|---|---|
| Drop `Page Views Per Visit` | 0.8171 | −0.0005 (noise) |
| Drop `City` | 0.8173 | −0.0003 (noise) |
| Drop `Country` | 0.8172 | −0.0004 (noise) |
| Drop free-book flag | 0.8177 | +0.0001 (noise) |
| `Specialization` → Given / Missing | 0.8186 | +0.0010 (noise) |
| Drop `Specialization` entirely | 0.8165 | −0.0011: kept as Given / Missing instead |

**Rule:** when a simpler version scores the same (difference well inside ± 0.011), the simpler version wins. All five simplifications together:

| Features | Count | CV PR-AUC | CV ROC-AUC | CV Brier | CV precision top 20% | CV recall top 20% |
|---|---|---|---|---|---|---|
| All | 51 | 0.8176 ± 0.0108 | 0.8679 | 0.1402 | 86.3% | 44.8% |
| **Selected** | **24** | **0.8176 ± 0.0109** | **0.8691** | **0.1404** | **85.7%** | **44.5%** |

Same quality with half the features, and the API needs 4 fewer input fields. Details: [features.md](features.md).

**Validation with the selected features** (`make baselines`):

| Model | PR-AUC | ROC-AUC | Brier | Precision top 20% | Recall top 20% | Recall top 50% |
|---|---|---|---|---|---|---|
| Logistic Regression, all features (step 2.3) | 0.840 | 0.887 | 0.130 | 88.9% | 46.2% | 87.4% |
| **Logistic Regression, selected features** | **0.839** | **0.887** | **0.130** | **88.7%** | **46.1%** | **87.4%** |
