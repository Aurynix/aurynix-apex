# Architecture Decision Records

Short records of decisions that shape the model and the system. Newest last.

| ADR | Decision | Step |
|---|---|---|
| [001](#adr-001-drop-post-contact-columns-leakage) | Drop 9 post-contact columns (leakage) | 1.4 |
| [002](#adr-002-fill-missing-values-and-cap-outliers-with-fixed-values-in-cleaning) | Fill missing values and cap outliers with fixed values in cleaning | data cleaning |
| [003](#adr-003-logistic-regression-as-the-model) | Logistic Regression as the model | 2.3 |
| [004](#adr-004-logistic-regression-settings-c--1-no-class-weights) | C = 1, no class weights | 2.5 |
| [005](#adr-005-no-extra-calibration-keep-raw-logistic-regression-probabilities) | No extra calibration | 3.2 |
| [006](#adr-006-api-thin-routers-one-scoring-service-no-training-in-the-api) | API: thin routers, one scoring service, no training in the API | 4.1 |
| [007](#adr-007-drift-monitoring-with-own-psi-code-and-three-checks) | Drift monitoring with own PSI code and three checks | 4.4 |

Other key choices are recorded where they were made: capacity-based segments ([problem_framing.md](problem_framing.md) 4), stratified random split ([splits.md](splits.md)), feature selection ([models.md](models.md) step 2.4).

---

## ADR-001: Drop post-contact columns (leakage)

- **Date:** 2026-10-05 · **Step:** 1.4 · **Status:** Accepted

**Context.** Apex scores a lead when it is created, before any sales contact ([problem_framing.md](problem_framing.md)). The dataset is a snapshot taken later, so some columns describe what happened *after* the contact. A model trained on them looks great offline and fails in production.

**Decision.** Drop 9 columns via `config.json → data.leakage_columns`:
`Tags`, `Lead Quality`, `Last Activity`, `Last Notable Activity`, `Lead Profile`, and the 4 `Asymmetrique *` columns.

**Why.**
- `Tags`, `Lead Quality`: the publisher's dictionary says they are the lead's current status and the employee's judgment. A quick model gains +0.148 and +0.061 PR-AUC from them.
- `Last Activity`, `Last Notable Activity`: state at export time, including sales actions such as `SMS Sent`.
- `Lead Profile`, `Asymmetrique *`: assigned labels and scores with unknown timing. Dropped to be safe: the cost is a little signal (≤ +0.022 PR-AUC), the benefit is honest offline numbers.

**Consequences.** Base PR-AUC of a quick model drops from 0.95 (with `Tags`) to 0.805, which is the honest number to beat. Kept but watched: `Lead Origin = Lead Add Form`, website activity, occupation. Details and per-column audit: [data_dictionary.md](data_dictionary.md) section 2.

---

## ADR-002: Fill missing values and cap outliers with fixed values in cleaning

- **Date:** 2026-10-05 · **Step:** data cleaning · **Status:** Accepted

**Context.** Missing values and outliers can be handled in two places: in `clean.py` with fixed rules, or in the model pipeline with values learned on the training split only. The second is stricter but adds moving parts.

**Decision.** Do it in `clean.py` with values fixed in `config.json → data`: text missing → `"Missing"`; `TotalVisits` / `Page Views Per Visit` missing → 3 / 2 (medians); caps 30 / 15; `Country` → India / Other. Count-based rare grouping (e.g. small `Lead Source` values) stays in the model pipeline (step 2.1).

**Why.** Keep it simple: one stateless function cleans the training file and a single API lead the same way, and the cleaned data has no missing values. The leak is tiny: two medians and two round caps read once from 9,240 rows.

**Consequences.** Changing a fill value or cap is a config change and needs a retrain. Details: [data_cleaning.md](data_cleaning.md).

---

## ADR-003: Logistic Regression as the model

- **Date:** 2026-10-06 · **Step:** 2.3 · **Status:** Accepted

**Context.** Gradient boosting (LightGBM, XGBoost) often wins on tabular data, but it is harder to explain and its probabilities usually need calibration. A quick 5-fold CV on the **train split only**, default settings, same 51 features:

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.818 ± 0.010 | 0.868 |
| HistGradientBoosting | 0.818 ± 0.011 | 0.868 |
| LightGBM | 0.818 ± 0.012 | 0.867 |
| XGBoost | 0.812 ± 0.012 | 0.862 |

**Decision.** Use Logistic Regression.

**Why.** Same accuracy, and simpler: one weight per feature that a sales rep can understand, probabilities that are close to calibrated without extra steps, fast training and serving. The data is small (5,544 training leads) and the main signals add up independently (Lead Add Form, occupation, time on site), which is what a linear model captures.

**Consequences.** Model work focuses on Logistic Regression settings (regularization strength `C`, class weights) rather than comparing model families. Validation baseline: PR-AUC 0.840, 46.2% of buyers in the top 20% ([models.md](models.md)).

---

## ADR-004: Logistic Regression settings: C = 1, no class weights

- **Date:** 2026-10-06 · **Step:** 2.5 · **Status:** Accepted

**Context.** Grid search with 5-fold CV on the train split: `C` ∈ {0.01, 0.03, 0.1, 0.3, 1, 3, 10} × `class_weight` ∈ {none, balanced}. Imbalance is moderate (38.5% buyers).

**Decision.** `C = 1.0`, `class_weight = none`, stored in `config.json → model`.

**Why.**
- `C` from 0.3 to 10 gives the same PR-AUC (0.8165–0.8177, std ≈ 0.011); `C = 1` sits in the middle of that plateau and has the best ROC-AUC. Strong regularization (`C` ≤ 0.1) clearly hurts.
- `balanced` class weights leave the ranking unchanged (0.8171 vs. 0.8176) but inflate probabilities (validation mean prediction 0.461 vs. actual 0.385; Brier 0.136 vs. 0.130). Segments are cut by share of leads, not by probability, so class weights bring no benefit, and calibrated probabilities are needed (step 3.2).

**Consequences.** The tuned model is the default Logistic Regression: validation PR-AUC 0.839. Resampling (e.g. SMOTE) is not tried: if reweighting the classes does not change the ranking, resampling, which has a similar effect, is unlikely to.

---

## ADR-005: No extra calibration: keep raw Logistic Regression probabilities

- **Date:** 2026-10-06 · **Step:** 3.2 · **Status:** Accepted

**Context.** Probabilities are shown to sales reps, so they should be honest. Checked on out-of-fold predictions over train + validation (7,392 leads), without the test split: raw model vs. Platt scaling vs. isotonic regression.

| | ECE | Brier | PR-AUC |
|---|---|---|---|
| Raw | 0.031 | 0.1377 | 0.824 |
| Platt | 0.032 | 0.1378 | 0.824 |
| Isotonic | 0.006 | 0.1362 | 0.821 |

**Decision.** Keep the raw Logistic Regression probabilities.

**Why.** The raw model already meets the target (ECE ≤ 0.05) and its average prediction matches the actual rate. Isotonic is better calibrated but adds 5 models and a step function, makes explanations indirect, and lowers PR-AUC slightly, for a Brier gain of 0.0015. Platt adds nothing to a model that is already a sigmoid.

**Consequences.** Known bias: 0.5–0.6 scores convert ~10 points more than predicted, 0.8–0.9 scores ~10 points less. Segments (step 3.3) use ranks, so they are not affected. If exact percentages become important in the product, switch to isotonic (one line in `make_model`).

---

## ADR-006: API: thin routers, one scoring service, no training in the API

- **Date:** 2026-10-06 · **Step:** 4.1 (4.3 skipped) · **Status:** Accepted

**Context.** The API must score leads, segment them, explain them, and log them. Training could also be exposed as an endpoint (`/pipeline/train`).

**Decision.**
- Routers only validate requests (Pydantic) and call one `ScoringService`, which does score → segment → explain → log.
- At startup the API loads the saved artifacts once (`model.pkl`, `model_meta.json`). It **never trains**; training stays a separate workflow (`make train`). No training endpoint.
- Reasons are part of every prediction; there is no separate explain endpoint.

**Why.** One place for the logic (API, demo, and simulation use the same service), small routers, easy tests without a server, and a predictable API: its behavior changes only when a new model is trained and deployed on purpose. A training endpoint would add background jobs, status tracking, and model swapping for no current user need.

**Consequences.** Retraining = `make train` + restart the API (in Docker, `models/` is mounted, so no rebuild). Revisit a training endpoint if retraining must be triggered from the product.

---

## ADR-007: Drift monitoring with own PSI code and three checks

- **Date:** 2026-10-06 · **Step:** 4.4 · **Status:** Accepted

**Context.** There are no conversion outcomes in production yet, so performance cannot be measured directly. Tools such as Evidently can produce drift reports.

**Decision.** Compare recent predictions with the training reference profile in three checks: **feature drift** (PSI per model input, fixed training bins), **prediction drift** (score PSI and High / Medium / Low shares), and **data quality** (missing rates, unseen categories, leads relying on defaults, rejected requests). PSI is computed by a few lines of own code; results go to SQLite and a JSON report. No Evidently.

**Why.** PSI is simple, explainable, and enough for these inputs. Own code avoids a large dependency and keeps the numbers in the API and the demo. Evidence: the simulation flags a "new campaign" shift (Lead Source PSI 2.54, High share 20% → 7%) and does not flag stable traffic (all PSI ≤ 0.02) ([monitoring.md](monitoring.md)).

**Consequences.** Drift says the data changed, not that the model is wrong. When outcomes arrive, add an `outcomes` table and track live precision of the High segment.

