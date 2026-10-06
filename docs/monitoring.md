# Drift Monitoring

> Stage 14 (step 4.4). Code: [`src/apex/monitoring/`](../src/apex/monitoring/). Settings: `config.json → monitoring`.

```bash
make monitor      # check the last 7 days of API predictions (data/apex.db)
make drift-demo   # simulate stable vs. drifted traffic and show the result
```

A model trained on last year's leads can quietly get worse when the **kind of leads** changes: a new campaign, a new form, a new source. Monitoring compares the leads scored recently with the **reference profile** saved at training time (`models/reference_profile.json`, step 3.5).

Drift means **the data changed**, not that the model is wrong. It is the signal to investigate, and to retrain if the change is real.

---

## The four checks

### 1. Feature drift: did the inputs change?

For each model input (time on website, visits, time per visit, occupation, lead origin, lead source, specialization, email opt-out), compare its distribution now with training using the **PSI**.

- Numeric inputs: the bin edges are the training quantiles, fixed in the profile and reused every time.
- Categorical inputs: the share of each training category; values never seen in training (e.g. a new source `TikTok`) are grouped as `__other__` and reported as `unseen_share`.

### 2. Prediction drift: did the scores change?

- **Score PSI**: the distribution of predicted probabilities now vs. training.
- **Segment shares**: High / Medium / Low now vs. training (20 / 30 / 50). For example, High 47% / Medium 38% / Low 15% would mean the model suddenly sees far more "good" leads, which must be explained before the sales team trusts it.

### 3. Data quality: is the input healthy?

| Check | Warning when |
|---|---|
| Missing rate per field (including `"Select"`) | it rises by ≥ 10 points vs. training |
| Unseen categories per field | ≥ 5% of leads |
| Leads that relied on defaults (at least one field missing) | reported for context (about half of the training leads miss at least one field) |
| Invalid requests rejected by the API (422) | ≥ 5% of requests |

Invalid requests are logged by the API in `rejected_requests`, so this check sees problems the model never sees.

### 4. Performance: is the model still right?

Drift only shows that the data changed. When the real result of a lead is known (converted or not), send it to `POST /outcomes` with the lead's `prediction_id` (returned by every prediction). The monitor then compares real results with the test results saved with the model (`model_meta.json`):

| Measure | Expected (test) | Warning | Drift → retraining recommended |
|---|---|---|---|
| PR-AUC on leads with outcomes | 0.787 | drop ≥ 0.05 | drop ≥ 0.10 |
| Precision of the High segment (share that converted) | 83% | drop ≥ 5 points | drop ≥ 10 points |

- Uses outcomes **recorded in the last 30 days** (conversions arrive later than predictions); needs **200** outcomes with both results present.
- Sending an outcome again for the same prediction replaces it. Unknown ids are rejected (404) and nothing is saved.
- `retrain_recommended` is `true` when performance or the score distribution is in drift.

**Limit (feedback loop):** outcomes mostly arrive for leads that were called, and most called leads are High. Measured performance therefore leans toward the model's own choices. Keeping a small random sample of Medium / Low leads that are also called would make it unbiased ([ADR-008](decisions.md)).

## PSI

**Population Stability Index**: how different two distributions are over the same bins.

```
PSI = Σ (now_share − training_share) × ln(now_share / training_share)
```

| PSI | Status | Action |
|---|---|---|
| < 0.10 | 🟢 ok | None |
| 0.10 – 0.25 | 🟡 warning | Investigate |
| ≥ 0.25 | 🔴 drift | Find the cause; consider retraining |

Empty bins get a tiny share (0.0001) so the formula never divides by zero.

**Overall status** = the worst of all checks. Drift and data quality need **200** predictions in the window, performance needs **200** outcomes; with neither, the run is `insufficient_data`.

## Simulation: does it catch real drift?

`make drift-demo` ([`simulate.py`](../src/apex/monitoring/simulate.py)) samples 2,000 real leads from `Leads.csv`, scores them with the real `ScoringService` (logged exactly like API traffic, in a separate `data/simulation.db`), and runs the monitor. Then it does the same after a **"new campaign" shift**: 30% of leads come from a new source (TikTok) through landing pages, leads spend half as long on the site, and 50% more skip the occupation question.

| | Stable traffic | Drifted traffic |
|---|---|---|
| **Overall** | 🟢 **ok** | 🔴 **drift** |
| Lead Source PSI | 0.008 | **2.537** (30% unseen) |
| Time on website PSI | 0.003 | **1.305** |
| Occupation PSI | 0.002 | **0.569** |
| Time per visit PSI | 0.001 | **0.318** |
| Other inputs PSI | ≤ 0.003 | ≤ 0.098 |
| Score PSI | 0.017 | **0.490** |
| High / Medium / Low | 19% / 32% / 49% | **7% / 16% / 77%** |
| Occupation missing | 29% (training 29%) | **66%** |

Stable data is not flagged; the shift is flagged on exactly the inputs that changed, and the segment shares show the business impact (High leads drop from 20% to 7%).

## Simulation: does it catch a model that went wrong?

`make outcomes-demo` samples 2,000 real leads **with their real outcome**, scores them, and sends the outcomes back. Then it repeats with **changed customer behavior**: half of the outcomes are shuffled, so the score no longer matches who buys, while the inputs stay exactly the same.

| | Real outcomes | Behavior changed |
|---|---|---|
| Feature / prediction drift, data quality | 🟢 ok | 🟢 ok (inputs did not change) |
| PR-AUC (expected 0.787) | 0.828 | **0.571** |
| High precision (expected 83%) | 86% | **64%** |
| Performance | 🟢 ok | 🔴 drift |
| `retrain_recommended` | false | **true** |

Only the outcomes catch this case. (The first scenario is in-sample, since the final model was trained on all of `Leads.csv`, so 0.828 is slightly optimistic; the demo shows the mechanics.)

## Where results go

| Place | Content |
|---|---|
| SQLite `drift_runs` (`data/apex.db`) | One row per run: time, sample size, status, score PSI, full report (JSON) |
| `reports/monitoring/latest.json` | The latest report, for people |
| `GET /monitoring/latest`, `/monitoring/history` | The same, through the API |

API:

| Method | Path | Description |
|---|---|---|
| `POST` | `/outcomes` | Send real results: `{"outcomes": [{"prediction_id": 1042, "converted": true}]}` |
| `POST` | `/monitoring/run?window_days=7` | Run the checks now and return the report |
| `GET` | `/monitoring/latest` | The latest report (404 if none yet) |
| `GET` | `/monitoring/history?limit=30` | Status, sample size, score PSI, segment shares per run |

## Report (shortened)

```json
{
  "created_at": "2026-10-06T00:00:00+00:00",
  "window_days": 7,
  "n_samples": 2000,
  "status": "drift",
  "features": {
    "Lead Source": {"type": "categorical", "psi": 2.537, "status": "drift", "unseen_share": 0.30},
    "Total Time Spent on Website": {"type": "numeric", "psi": 1.305, "status": "drift"}
  },
  "prediction": {
    "score_psi": 0.49, "segment_psi": 0.357, "mean_score": 0.206,
    "segment_shares": {"reference": {"high": 0.2, "medium": 0.3, "low": 0.5},
                       "current": {"high": 0.07, "medium": 0.16, "low": 0.77}},
    "status": "drift"
  },
  "data_quality": {
    "missing_rate": {"What is your current occupation": {"reference": 0.29, "current": 0.66, "status": "warning"}},
    "unseen_category_share": {"Lead Source": 0.30},
    "defaulted_share": 0.77,
    "rejected_requests": 0, "rejection_rate": 0.0,
    "status": "warning"
  }
}
```

## Not done (on purpose)

- **Evidently HTML reports**: the own PSI code, the JSON report, and the API cover the need with no extra dependency.
- **Random-contact sample** against the feedback loop: documented, not built (needs a sales process change, not code).
