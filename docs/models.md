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
| Capture top 20% | Share of all buyers in the 20% highest-scored leads (the **High** segment) | 20% |
| Lift top 20% | Capture ÷ 20%: how many times better than calling at random | 1.0 |
| Capture / lift top 50% | Same for High + Medium | 50% / 1.0 |

## Step 2.3: Baselines

```bash
make split       # once, if data/splits.csv does not exist
make baselines   # fit on train (5,544), score on validation (1,848), log to MLflow
```

| Model | PR-AUC | ROC-AUC | Brier | Capture top 20% | Lift top 20% | Capture top 50% |
|---|---|---|---|---|---|---|
| No skill (predicts 38.5% for everyone) | 0.385 | 0.500 | 0.237 | 19.1% | 0.96 | 52.1% |
| **Logistic Regression** (default settings) | **0.840** | **0.887** | **0.130** | **46.2%** | **2.31** | **87.4%** |

Against the targets in [problem_framing.md](problem_framing.md):

| Target | Required | Logistic Regression | |
|---|---|---|---|
| Capture in top 20% | ≥ 40% (ceiling ≈ 52%) | 46.2% | ✅ |
| Capture in top 50% | ≥ 80% | 87.4% | ✅ |
| PR-AUC | far above no skill (0.385) | 0.840 | ✅ |
| Brier | below no skill (0.237) | 0.130 | ✅ |
| Leakage alarm | ROC-AUC < 0.95 | 0.887 | ✅ no alarm |

The top 20% can hold at most 20 / 38.5 ≈ 52% of buyers, so 46.2% is about 89% of the best possible result.

The no-skill scores are all equal, so its ranking is arbitrary; its capture values (19.1%, 52.1%) are what random calling gives, up to noise.

**These numbers are the bar to beat.** The test set is untouched; its single use is step 3.1.
