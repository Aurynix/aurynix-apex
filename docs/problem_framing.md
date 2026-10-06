# Problem Framing

> Stage 1 of the [ML pipeline](../README.md#ml-pipeline). This document defines **what** Apex predicts, **when**, **for whom**, and **how success is judged**. Later stages must stay consistent with it; any change is recorded in [`decisions.md`](decisions.md).

---

## 1. Business problem

A company receives more leads than its sales development reps (SDRs) can contact. Leads are currently worked in arrival order or with hand-made CRM point rules. As a result:

- SDR time goes to low-intent leads.
- High-intent leads wait and may buy elsewhere.
- Nobody can tell when the rules stop working.

**Goal:** rank incoming leads by how likely they are to convert, so the limited SDR capacity goes to the leads most likely to buy first.

## 2. ML task

| Item | Definition |
|---|---|
| **Task type** | Binary classification, used as a **ranking** (probability of conversion) |
| **Unit of prediction** | One lead (one row of `Leads.csv`) |
| **Target** | `Converted` (1 = became a paying customer, 0 = did not) |
| **Model output** | Calibrated probability in [0, 1], then a priority segment (High / Medium / Low) and top reasons |
| **Consumer** | SDR team, through Aurynix Pulse (CRM view) or the API |

The ranking matters more than the exact 0/1 decision, because the team contacts leads from the top of the list until it runs out of capacity. The probability must also be **calibrated** so that "0.70" can be shown to a sales rep and mean what it says.

## 3. Prediction moment

> **A lead is scored when it enters the SDR queue: after the lead form is submitted, before any sales contact.**

Allowed features are those known at that moment:
- How the lead arrived (origin, source, campaign).
- What the lead told us on the form (country, occupation, specialization, preferences).
- Website and engagement activity recorded **up to** that moment.

Not allowed: anything created **by or after** sales contact (sales tags, rep quality judgments, call outcomes, activity triggered by the rep).

**Known limitation:** the dataset is a single snapshot per lead with no timestamps. We cannot prove when activity counts such as `TotalVisits` or `Total Time Spent on Website` were recorded. Step 1.4 (leakage audit) decides column by column, and every uncertain column is recorded as a risk.

## 4. How predictions are used

Segments are defined by **capacity**, not fixed probability cut-offs (shares come from `config.json → segmentation`):

| Segment | Share of leads (initial) | Action |
|---|---|---|
| 🟢 High | Top 20% | Call the same day |
| 🟡 Medium | Next 30% | Follow up within the week |
| 🔴 Low | Remaining 50% | Automated email nurturing; no SDR time |

If the team's capacity changes, the shares change in config and the model stays the same.

## 5. Success criteria

### 5.1 Business criteria (primary)

The main question: **what share of all conversions lands in the High segment?**

| Metric | Random ordering | Target (provisional) |
|---|---|---|
| Share of conversions captured by top 20% | 20% | **≥ 40%** (lift ≥ 2.0) |
| Share of conversions captured by top 50% (High + Medium) | 50% | **≥ 80%** |
| Conversion rate in High vs. Low | equal | High ≥ 3× Low |

**Ceiling check:** the conversion rate in this dataset is expected to be roughly 38% (confirmed in step 1.2). The top 20% of leads therefore cannot hold more than about 20 / 38 ≈ **52%** of all conversions, so the highest possible lift at 20% is about 2.6. Targets are set relative to that ceiling. They are provisional and will be revisited once the baseline exists (step 2.3).

### 5.2 ML criteria

| Metric | Requirement |
|---|---|
| **PR-AUC** (primary) | Higher than the Logistic Regression baseline on validation; far above the no-skill value (= conversion rate) |
| **ROC-AUC** | Reported for ranking quality |
| **Brier score** | Lower than the no-skill score `p(1 − p)` (≈ 0.24 for p ≈ 0.38) |
| **Calibration** | Reliability curve close to the diagonal; expected calibration error ≤ 0.05 |
| **Stability** | Standard deviation of PR-AUC across CV folds is small compared to the gap over the baseline |

### 5.3 Guardrails

- **Leakage alarm:** ROC-AUC ≥ 0.95, or one feature dominating the model, triggers a leakage investigation before the result is accepted.
- **Test set used once:** final numbers come from a single evaluation on the held-out test set (step 3.1).
- **Explainable:** every score comes with its top reasons, and the reasons must make sense to a sales rep.

## 6. Baselines

| Baseline | Purpose |
|---|---|
| Majority class / random ordering | The "no model" floor; lift = 1.0 |
| Logistic Regression (same features) | The simple, explainable model every complex model must beat |

## 7. Scope

**In scope**
- Scoring new leads from the X Education dataset schema.
- Segmentation, per-lead explanations, API serving, and drift monitoring.

**Out of scope (for now)**
- Revenue or deal-size prediction (only *whether* the lead converts).
- Time-to-conversion and next-best-action recommendations.
- Live feedback from real outcomes: built as `POST /outcomes` + the performance check ([monitoring.md](monitoring.md)).
- Fairness auditing beyond checking that no protected attributes are used as features.

## 8. Assumptions

1. Historical labels are correct: `Converted = 1` means the lead really became a customer.
2. Future leads come from a similar process (same form, same channels). If they don't, drift monitoring should detect it.
3. SDR capacity can be expressed as a share of incoming leads.
4. Recorded website activity mostly comes before sales contact. Step 1.4 checks this per column.

## 9. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **Target leakage** (`Tags`, `Lead Quality`, `Last Activity`, `Last Notable Activity`, score columns) | Inflated metrics; model useless in production | Leakage audit (1.4), with-vs-without experiment, and the leakage alarm in 5.3 |
| **Hidden missing values** (`"Select"`) | Biased features; misleading EDA | Treated as missing in `clean.py` (`config.json → data.missing_placeholders`) |
| **Snapshot data without timestamps** | Cannot prove when activity happened | Conservative feature choice; documented limitation |
| **Dataset not representative** (one company, one period) | Results may not carry over to Aurynix Pulse clients | Re-validate on Bank Marketing data; drift monitoring |
| **Class imbalance** (moderate here, severe in Bank Marketing) | Accuracy looks good while ranking is poor | PR-AUC as the primary metric; class weights tested in tuning (step 2.5) |
| **Feedback loop** (only High leads get called, so mostly they convert) | Future labels become biased toward the model's own choices | Outcomes are tracked; keeping a small random-contact sample of Medium / Low leads is recommended ([ADR-008](decisions.md)) |

## 10. Open questions

- [x] Confirm the conversion rate and row count (step 1.2): 9,240 leads, 38.54% converted ([data_dictionary.md](data_dictionary.md)).
- [x] Which activity columns pass the prediction-moment test (step 1.4)? Website visits and time on site are kept (with a documented risk); `Last Activity`, `Last Notable Activity`, `Tags`, `Lead Quality`, `Lead Profile`, and the `Asymmetrique` scores are removed ([ADR-001](decisions.md)).
- [x] Do the provisional business targets in 5.1 hold up? Yes, all met on the held-out test set: 43.3% of conversions in the top 20% (target ≥ 40%), 85.4% in the top 50% (≥ 80%), High converts 7.1× more than Low (≥ 3×) ([models.md](models.md) steps 3.1 and 3.3).
