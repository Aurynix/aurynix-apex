# Aurynix Apex — Build Steps

The working plan for building Apex, one step at a time. Each step ends with a short summary and a check-in before moving on.

**How to use this file**
- Work top to bottom; a step starts only when the previous one is done.
- Each step lists its **goal**, **tasks**, **files**, and **done when** (the exit criteria).
- Tick the box when a step is done and add a short note under **Log** at the bottom.
- Results and metrics go into the README only after the step that produces them is complete.

Legend: `[x]` done · `[ ]` to do

### Git workflow

`main` only changes through pull requests. Each step gets its own branch from the latest `main`:

```bash
git switch main && git pull                     # start from the latest main
git switch -c feature/1.1-problem-framing       # one branch per step
# ... work, make lint, make test, commit ...
git push -u origin feature/1.1-problem-framing
gh pr create --base main                        # review and merge on GitHub
```

Branch prefixes: `feature/<step>-<name>` for build steps, `docs/…` for documentation-only changes, `fix/…` for bug fixes, `chore/…` for tooling.
The step's checkbox and **Log** entry are updated in the same branch.

---

## Phase 0 — Project Setup

### 0.1 README ✅
- [x] Project README with problem, design, pipeline, API, monitoring, and roadmap.
- **Files:** `README.md`

### 0.2 .gitignore ✅
- [x] Ignore data, model artifacts, databases, MLflow runs, secrets, caches, virtual environments.
- **Files:** `.gitignore`

### 0.3 Project structure ✅
- [x] Folders, package stubs, `config.json` + loader, `pyproject.toml` (with dependencies), Makefile, Docker, LICENSE.
- **Files:** everything under `src/`, `docs/`, `data/`, `models/`, `reports/`, plus root config files.

### 0.4 Build plan ✅
- [x] This file.
- **Files:** `BUILD_STEPS.md`

### 0.5 Environment & first commit ✅
- [x] Switch to uv: dependencies in `pyproject.toml`, exact versions pinned in `uv.lock`.
- [x] `make venv && make install`
- [x] Verify imports: `pandas`, `sklearn`, `xgboost`, `lightgbm`, `shap`, `mlflow`, `fastapi`.
- [x] Add a first test (`tests/test_config.py`) so `make test` and `make lint` pass.
- [x] First commit on `main`; push to `github.com/Aurynix/aurynix-apex`.
- **Done when:** `make lint` and `make test` pass on a clean checkout.

---

## Phase 1 — Understand & Explore (Week 1)

### 1.1 Problem framing — *Stage 1* ✅
- [x] Define the target (`Converted`) and the **prediction moment** (lead creation, before any sales contact).
- [x] Define the unit of prediction (one lead), the consumer (SDR team via Aurynix Pulse), and the action taken per segment.
- [x] Define business success criteria (e.g. top 20% of leads capture ≥ X% of conversions) and ML success criteria (PR-AUC above baseline).
- [x] List assumptions and risks (leakage, imbalance, dataset representativeness).
- **Files:** `docs/problem_framing.md`
- **Done when:** someone outside the project could read it and know exactly what is predicted, when, and how success is judged.

### 1.2 Data collection — *Stage 2* ✅
- [x] Download the dataset from Kaggle into `data/raw/` (`make data-download`, via `kagglehub`; saved as `Leads.csv`).
- [x] Implement `load.py`: read raw CSV using paths from `config.json`, with tests; `make data-info` prints shape, hash, and target rate.
- [x] Record source, download date, row/column counts, and file hash in `docs/data_dictionary.md` (9,240 × 37, conversion rate 38.54%).
- **Files:** `src/apex/data/load.py`, `src/apex/data/download.py`, `docs/data_dictionary.md`
- **Done when:** `load_raw()` returns the DataFrame and the source is documented.

### 1.3 Data quality checks — *Stage 3 (part 1)* ✅
- [x] Missing values per column, including hidden nulls (`"Select"`).
- [x] Duplicates (rows and IDs), invalid values, numeric outliers (`TotalVisits`, `Page Views Per Visit`, time on site).
- [x] Class balance of `Converted`.
- [x] Implement `clean.py`: replace placeholders with NaN, fix types, drop ID columns, standardize category labels.
- [x] Unit tests for the cleaning functions.
- **Files:** `src/apex/data/clean.py`, `tests/test_clean.py`, `docs/data_quality.md`, `reports/figures/dq_*.png`
- **Done when:** cleaning is a tested function (not notebook code) and every issue found has a documented decision.

### 1.4 Leakage audit — *Stage 4* ✅
- [x] Classify **every** column: available at lead creation? yes / no / uncertain, with reasoning.
- [x] Verify the candidates: `Tags`, `Lead Quality`, `Last Activity`, `Last Notable Activity`, score/index columns.
- [x] Put confirmed leakage columns in `config.json → data.leakage_columns`; `clean.py` drops them.
- [x] Quick check: train a throwaway model with and without suspect columns and compare (a big jump is evidence of leakage).
- **Files:** `docs/data_dictionary.md`, `config.json`, `docs/decisions.md` (ADR), `src/apex/data/leakage.py`, `tests/test_leakage.py`
- **Done when:** every column has a status and the leakage decision is recorded as an ADR.

### 1.5 EDA report — *Stage 3 (part 2)* ✅
- [x] Univariate distributions; conversion rate by each categorical feature; numeric features vs. target.
- [x] Correlations and redundant features.
- [x] Save key figures to `reports/figures/`.
- [x] Write findings and the resulting feature ideas.
- **Files:** `docs/eda.md`, `src/apex/data/eda.py`, `tests/test_eda.py`, `reports/figures/eda_*.png`
- **Done when:** there is a clear list of findings that drive feature engineering.

---

## Phase 2 — Build Models (Week 2)

### 2.1 Preprocessing & feature pipeline — *Stage 5* ✅
- [x] `sklearn` `Pipeline` + `ColumnTransformer`: scale numerics, one-hot categoricals (missing values are already filled in `clean.py`).
- [x] Rare-category grouping using `rare_category_min_share` (`OneHotEncoder` infrequent column); unseen categories go to the same column at inference.
- [x] Engineered features from EDA findings.
- [x] Tests: pipeline fits/transforms, handles missing and unseen categories.
- **Files:** `src/apex/data/features.py`, `tests/test_features.py`, `docs/features.md`, `config.json`, `Makefile`
- **Done when:** one pipeline object goes from raw rows to model-ready features, and it is fitted on training data only.

### 2.2 Stratified splits — *Stage 6* ✅
- [x] Train / validation / test split, stratified on target, seeded by `random_state`.
- [x] Save split indices so they are reproducible.
- [x] Rule: the test set is not touched until step 3.1.
- **Files:** `src/apex/data/split.py`, `tests/test_split.py`, `docs/splits.md`, `config.json`, `Makefile`
- **Done when:** splits are fixed, reproducible, and class ratios match across splits.

### 2.3 Baselines — *Stage 7* ✅
- [x] Majority-class baseline.
- [x] Logistic Regression on the full pipeline.
- [x] Implement `evaluate.py`: PR-AUC, ROC-AUC, Brier, lift/gain at top-k.
- [x] Set up MLflow tracking (`make mlflow-ui`).
- **Files:** `src/apex/models/evaluate.py`, `src/apex/models/train.py`, `tests/test_evaluate.py`, `tests/test_train.py`, `docs/models.md`, `docs/decisions.md` (ADR-003), `Makefile`
- **Done when:** baseline metrics are logged in MLflow and become the bar to beat.

### 2.4 Cross-validation & feature selection — *Stage 8* ✅
> Changed after ADR-003 (Logistic Regression is the model): instead of comparing model families, this step makes the evaluation stable and simplifies the features.
- [x] Stratified 5-fold CV on the train split in `cv.py`; `make cv`; mean ± std logged in MLflow.
- [x] Test the EDA feature ideas (drop `Page Views Per Visit`, `City`, `Country`, free-book flag; `Specialization` → present / missing); keep a simplification when CV PR-AUC does not drop.
- [x] Record the result in `docs/models.md` and `docs/features.md`.
- **Files:** `src/apex/models/cv.py`, `tests/test_cv.py`, `src/apex/data/features.py`, `config.json`, `docs/models.md`, `docs/features.md`
- **Done when:** the selected features are justified by CV mean ± std and logged in MLflow.

### 2.5 Tuning — *Stage 9* ✅
- [x] Grid search over Logistic Regression settings with CV on the train split: regularization strength `C`, `class_weight` (none vs. balanced).
- [x] Confirm the best settings on the validation split.
- [x] Record the choice in `docs/decisions.md`.
- **Files:** `src/apex/models/cv.py`, `src/apex/models/train.py`, `config.json`, `tests/test_cv.py`, `docs/models.md`, `docs/decisions.md` (ADR-004), `Makefile`
- **Done when:** the tuned model is at least as good as the baseline on validation, and the choice is justified in an ADR.

---

## Phase 3 — Evaluate & Explain (Week 3)

### 3.1 Final test evaluation — *Stage 10 (part 1)* ✅
- [x] Retrain the chosen model on train + validation.
- [x] Evaluate on the test set **once**.
- [x] PR curve, ROC curve, cumulative gain and lift charts.
- **Files:** `src/apex/models/evaluate.py`, `src/apex/models/train.py`, `tests/test_evaluate.py`, `reports/figures/test_evaluation.png`, `docs/models.md`, `README.md`, `Makefile`
- **Done when:** final metrics are recorded and the README results section is filled in.

### 3.2 Probability calibration — *Stage 10 (part 2)* ✅
- [x] Calibration curve + Brier score.
- [x] If needed, calibrate (isotonic / Platt) using held-out data only.
- **Files:** `src/apex/models/train.py`, `src/apex/models/evaluate.py`, `tests/test_evaluate.py`, `reports/figures/calibration.png`, `docs/models.md`, `docs/decisions.md` (ADR-005), `Makefile`
- **Done when:** a predicted 0.70 means roughly 70% actually convert (within reason).

### 3.3 Segmentation — *Stage 11* ✅
- [x] Implement `segment.py`: capacity-based cut-offs from `config.json → segmentation`.
- [x] Derive score thresholds from out-of-fold scores on train + validation (stored in `model_meta.json` in step 3.5).
- [x] Segment table: number of leads, share of conversions captured, actual conversion rate.
- **Files:** `src/apex/models/segment.py`, `tests/test_segment.py`, `src/apex/models/train.py`, `docs/models.md`, `README.md`, `Makefile`
- **Done when:** segments come out of config and the segment table shows that High captures far more conversions than its share of leads.

### 3.4 Explainability — *Stage 12* ✅
- [x] Global SHAP: summary plot and feature-importance ranking.
- [x] Per-lead SHAP: top positive and negative contributors, mapped back to readable feature names.
- [x] Sanity check: explanations agree with the EDA findings.
- **Files:** `src/apex/models/explain.py`, `tests/test_explain.py`, `reports/figures/explain_importance.png`, `docs/models.md`, `README.md`, `Makefile`
- **Done when:** `explain_one(lead)` returns readable top reasons.

### 3.5 Training artifacts ✅
- [x] `make train` saves `models/model.pkl`, `models/model_meta.json` (version, metrics, thresholds, training date, feature list).
- [x] Implement `reference.py`: build `models/reference_profile.json` (bin edges, category shares, missing rates, score distribution, segment shares).
- [x] `predict.py` + `make predict` for offline batch scoring.
- [x] Model card `models/model_card.json` (details, intended use, data, features, algorithm, performance, risk rating, limitations, monitoring); committed to git.
- **Files:** `src/apex/models/train.py`, `src/apex/models/predict.py`, `src/apex/models/card.py`, `src/apex/monitoring/reference.py`, `tests/test_reference.py`, `tests/test_card.py`, `tests/test_predict.py`, `config.json`, `Makefile`, `models/model_card.json`
- **Done when:** `make pipeline` runs end to end from `Leads.csv` to all three artifacts.

---

## Phase 4 — Ship & Monitor (Week 4)

### 4.1 API serving — *Stage 13* ✅
> Design: routers only validate; one `ScoringService` does score → segment → explain → log. The API loads saved artifacts at startup and never trains. Reasons are part of every prediction, so there is no separate `/predict/explain` endpoint.
- [x] `app.py` (app, lifespan, `/health`), `schemas.py` (validation only), `dependencies.py` (load model once, inject service), `service.py` (all logic).
- [x] Endpoints: `GET /health`, `GET /model/info`, `POST /predict/single`, `POST /predict/batch`.
- [x] `database.py`: `predictions` table; every scored lead is logged.
- [x] `make run` / `make run-prod`.
- [x] API tests with FastAPI `TestClient`.
- **Files:** `src/apex/api/`, `tests/test_api.py`, `docs/api.md`, `README.md`, `Makefile`
- **Done when:** `/docs` loads, `/model/info` returns the model's metadata, predictions match the contract in `docs/api.md`, and every prediction is logged.

### 4.2 Prediction endpoints & logging ✅
- [x] Merged into 4.1 (single + batch endpoints, logging, API tests). CSV upload for batch scoring stays with `make predict` for now.

### 4.3 Training job endpoint ⏭️ skipped
- [x] Skipped on purpose (KISS): training stays a separate workflow (`make train`), and the API only loads saved artifacts. Revisit if retraining must be triggered from the product.

### 4.4 Drift monitoring — *Stage 14* ✅
- [x] `drift.py`: PSI for numeric (fixed bins) and categorical features (unseen → `__other__`), with smoothing for empty bins; unit tests with known values.
- [x] `monitor.py`: `Monitor.run()` with window and `min_samples` check; three checks: **feature drift**, **prediction drift** (score PSI + High / Medium / Low shares), **data quality** (missing rates, unseen categories, leads relying on defaults, rejected requests); statuses.
- [x] SQLite `drift_runs` (full report as JSON) and `rejected_requests` (logged by the API); `make monitor`; `reports/monitoring/latest.json`.
- [x] Monitoring endpoints: `POST /monitoring/run`, `GET /monitoring/latest`, `GET /monitoring/history` (per-feature history is inside each report).
- [x] Simulation (`make drift-demo`): stable traffic → ok, shifted traffic → drift.
- [x] Evidently HTML report: skipped (own PSI + JSON report cover it, no extra dependency).
- **Files:** `src/apex/monitoring/`, `src/apex/api/` (database, app, dependencies, routers/monitoring.py), `tests/test_drift.py`, `tests/test_monitor.py`, `tests/test_api.py`, `tests/conftest.py`, `docs/monitoring.md`, `config.json`, `Makefile`
- **Done when:** a simulated shift is flagged, and stable data is not.

### 4.5 Docker & Streamlit demo ✅
- [x] `make docker-build && make docker-up` serves the API (health check, `models/` mounted read-only) and the demo (starts when the API is healthy); ports from `API_PORT` / `DEMO_PORT`.
- [x] `app/demo.py`: an API client with three tabs: score a lead from a form, upload a CSV (API fields or raw Kaggle columns) → ranked leads with reasons, drift status.
- [x] Docker image size: uv cache kept out of the image with a build cache mount.
- **Files:** `Dockerfile`, `docker-compose.yml`, `.env.example`, `app/demo.py`, `tests/test_demo.py`, `docs/deployment.md`, `Makefile`, `README.md`
- **Done when:** a fresh machine can run the API and demo from the README instructions.

### 4.6 Final documentation ✅
- [x] `docs/api.md`, `docs/deployment.md`, `docs/monitoring.md`; `docs/decisions.md` completed (index + ADR-006 API design, ADR-007 monitoring).
- [x] README: status badge and note, architecture diagram, pipeline table linked to docs, leakage audit results, results / figures / segment table, roadmap ticked.
- [x] README "Design Decisions": all six questions answered with evidence and ADR links.
- [x] `problem_framing.md` open questions closed; unused `todo` placeholder removed from the Makefile; all relative doc links checked.
- **Files:** `README.md`, `docs/decisions.md`, `docs/problem_framing.md`, `BUILD_STEPS.md`, `Makefile`
- **Done when:** the README's "Design Decisions" questions are all answered with evidence.

---

## Phase 5 — Later

- [x] `outcomes` table and live performance tracking: `prediction_id` in every prediction, `POST /outcomes`, performance check (real PR-AUC and High precision vs. test, `retrain_recommended`), `make outcomes-demo` (ADR-008).
- [ ] Run the pipeline on the Bank Marketing dataset (UCI 222) and confirm `duration` is caught as leakage: in progress as Phase 6.
- [ ] Integration with Aurynix Pulse.
- [x] CI/CD (GitHub Actions: lint + test on every push and PR; Docker build on PRs, push to GHCR on `main`) — `.github/workflows/ci-cd.yml`.
- [x] Remove unused dependencies (`xgboost`, `lightgbm`, `optuna`, `evidently`, `seaborn`, `pyarrow`, `python-multipart`; `shap` moved to dev): smaller installs and Docker image.

---

## Phase 6 — Second dataset: Bank Marketing

Run the same pipeline on UCI Bank Marketing (id 222) to show what is general, and check that the leakage audit catches `duration`. Scope: pipeline and report, no second API. Report: [`docs/bank_marketing.md`](docs/bank_marketing.md). Every command runs with `APEX_CONFIG=config_bank.json`.

### B.1 Data collection & quality ✅
- [x] `config_bank.json` with its own paths (`models/bank/`, `reports/figures/bank/`, `data/splits_bank.csv`), so the lead model is never overwritten.
- [x] Downloader: zip from a URL, including zips inside zips; loader: `;` separator, `yes`/`no` target → 1/0, `row_id` when the file has no id.
- [x] Same `clean()` through config only (`"unknown"` → missing, yes/no → 1/0 in any letter case, caps); data quality documented.
- **Files:** `config_bank.json`, `src/apex/data/download.py`, `src/apex/data/load.py`, `src/apex/data/clean.py`, tests, `docs/bank_marketing.md`
- **Done when:** `APEX_CONFIG=config_bank.json make data-download data-info` works and `clean(load_raw())` gives clean data with no new cleaning code.

### B.2 Leakage audit ✅
- [x] Suspect groups moved to config (`data.leakage_suspects`) so `make leakage` works for any dataset.
- [x] Classify every column against the prediction moment (before the call); with/without check in time-ordered and shuffled folds; `duration` stands out (+0.155 / +0.167 PR-AUC).
- [x] Removed `duration`, `campaign`, `day`, `month`, `contact`; kept client profile and previous-campaign history (base PR-AUC 0.34 vs. 0.117 random).
- [x] Finding for B.4: the file is ordered by date and conversion rises from 3% to 47%; report random and time splits.
- **Files:** `config.json`, `config_bank.json`, `src/apex/data/leakage.py`, `tests/test_config.py`, `docs/bank_marketing.md`

### B.3 Features ✅
- [x] `add_features()` is config-driven (`presence_only`, `ratios`, `flags`, `bins`, `drop`); lead features moved to config with identical results (test PR-AUC 0.7874).
- [x] Explanations are config-driven (`explain.labels`, `explain.groups`); yes/no fields read "yes" / "no"; figure folders are created when missing.
- [x] Bank ideas tested with CV on the train split: `age_group` instead of `age` (+0.013 PR-AUC) kept; "contacted before" flag and balance bands rejected (no gain beyond noise).
- **Files:** `src/apex/data/features.py`, `src/apex/models/explain.py`, `src/apex/models/predict.py`, `src/apex/models/train.py`, `config.json`, `config_bank.json`, tests, `docs/bank_marketing.md`, `docs/features.md`, `reports/figures/bank/explain_importance.png`

### B.4 Model, evaluation, segments ✅
- [x] MLflow experiment per dataset (`project.experiment`); model card optional (`project.model_card`, off for the bank); `make time-split` for date-ordered data (`split.time_ordered`).
- [x] Split, baselines (val PR-AUC 0.376 vs. 0.117), tuning (`C = 1`, no class weights: `balanced` doubles the Brier score), test once (PR-AUC 0.359, top 20% recall 49%), calibration (raw ECE 0.005), segments (High 28.6% vs. Low 5.8%), final model in `models/bank/`.
- [x] Time split: ranking holds (ROC-AUC 0.726 → 0.707), probabilities do not (ECE 0.163; conversion 6.7% → 31.6%).
- **Files:** `src/apex/models/train.py`, `src/apex/models/cv.py`, `src/apex/models/segment.py`, `config.json`, `config_bank.json`, `Makefile`, `docs/bank_marketing.md`, `reports/figures/bank/`

### B.5 Report
- [ ] Complete `docs/bank_marketing.md`; README section comparing both datasets.


## Log

| Date | Step | Notes |
|---|---|---|
| 2026-10-02 | 0.1–0.4 | README, .gitignore, project structure, and build plan created. |
| 2026-10-02 | 0.5 | uv + uv.lock (Python 3.11), dependencies installed, lint and 5 config tests passing, first commit pushed. |
| 2026-10-02 | — | Git workflow: one branch per step from `main`, merged via pull request. |
| 2026-10-02 | 1.1 | Problem framing: prediction moment, capacity-based segments, business + ML success criteria, risks. |
| 2026-10-02 | 1.2 | Kaggle download (`make data-download`), loader, source record: 9,240 × 37, conversion rate 38.54%. |
| 2026-10-02 | 1.3 | Data quality: `"Select"` hides up to 54.6% missing per column; missingness is informative (kept); 12 constant/near-constant columns dropped; stateless `clean.py` → 9,240 × 23; leakage suspects flagged for 1.4. |
| 2026-10-05 | — | Notebooks removed from the repo; `notebooks/` is now git-ignored local scratch for testing. All code lives in `src/` and `tests/`. |
| 2026-10-05 | 1.4 | Leakage audit: 9 post-contact columns dropped (`Tags` alone adds +0.148 PR-AUC); quick-model base PR-AUC 0.804 vs. 0.385 random; `make leakage`; ADR-001; cleaned data → 9,240 × 14. |
| 2026-10-05 | — | Full cleaning: missing values filled (`"Missing"`, fixed medians), `How did you hear…` dropped (78.5% empty), outliers capped (30 / 15), `Country` → India / Other, integer types, logic check; `docs/data_cleaning.md`, ADR-002; cleaned data → 9,240 × 13, 0 nulls. |
| 2026-10-05 | — | Removed `data/interim/` and `data/processed/`: cleaning runs in memory with `clean(load_raw())`; `make preprocess` removed. |
| 2026-10-05 | 1.5 | EDA (`make eda`): time on site is the strongest numeric (14% → 69%), visits/page views flat; Lead Add Form 92.5%, Working Professional 92%, occupation missing 14%; "what matters most" duplicates occupation missingness; 7 feature ideas for 2.1 in `docs/eda.md`. |
| 2026-10-05 | 2.1 | Feature pipeline (`make features`): clean → `time_per_visit`, `has_web_activity`, drop "what matters most" → scale + one-hot (rare/unseen → infrequent); 36 raw columns → 51 features; quick CV PR-AUC 0.810 → 0.816. |
| 2026-10-05 | 2.2 | Stratified 60/20/20 split (`make split`) saved as `data/splits.csv`; conversion 38.5% in every split; `load_splits()` hides test unless `include_test=True`; 22% of val/test leads have a look-alike in train (kept, documented). |
| 2026-10-06 | 2.3 | Baselines (`make baselines`, MLflow on SQLite): no skill PR-AUC 0.385 vs. Logistic Regression **0.840** on validation; top 20% captures 46.2% of buyers (target ≥ 40%), top 50% 87.4%; LR chosen as the model (ADR-003). |
| 2026-10-06 | 2.4 | Plan changed for Logistic Regression (ADR-003). 5-fold CV on train (`make cv`): selected features (24) give the same PR-AUC as all features (51), 0.8176 ± 0.011; dropped `Page Views Per Visit`, `City`, `Country`, free-book flag; `Specialization` → Given / Missing. Validation PR-AUC 0.839. |
| 2026-10-06 | 2.5 | Tuning (`make tune`, 14 setups): `C` 0.3–10 is a plateau (CV PR-AUC ≈ 0.817); `balanced` class weights do not improve ranking and inflate probabilities (val mean 0.461 vs. 0.385). Kept `C = 1`, no class weights (ADR-004, `config.json → model`); validation PR-AUC 0.839. |
| 2026-10-06 | 3.1 | Test set used once (`make evaluate`, fitted on train + val): PR-AUC 0.787 (95% CI 0.756–0.816), ROC-AUC 0.855, Brier 0.149; top 20% precision 83.2% / recall 43.3%, top 50% recall 85.4%; all business targets met. Lower than validation (0.839): sampling variation, same lead mix. README results filled in. |
| 2026-10-06 | 3.2 | Calibration (`make calibration`, out-of-fold on train + val): raw LR ECE 0.031 (target ≤ 0.05), mean prediction 0.386 vs. 0.385 actual; isotonic ECE 0.006 but 5 models and indirect explanations. Kept raw probabilities (ADR-005). |
| 2026-10-06 | 3.3 | Segmentation (`make segments`): thresholds High ≥ 0.751, Medium ≥ 0.270 from out-of-fold scores; test: High 84.1% conversion (43% of buyers), Low 11.8% (7.1×); High + Medium hold 84.6% of buyers. |
| 2026-10-06 | 3.4 | Explainability (`make explain`): exact linear SHAP (weight × (value − mean)), matches `shap.LinearExplainer`; one-hot and the 4 website inputs grouped into readable reasons; top drivers: website activity, occupation, lead origin, specialization; agrees with EDA; `Explainer.explain_one(lead)` returns probability + top reasons up / down. |
| 2026-10-06 | 3.5 | `make train` fits the final model on all 9,240 leads and saves `model.pkl`, `model_meta.json`, `reference_profile.json`, `model_card.json` (risk rating Low); thresholds from out-of-fold scores on all rows (High ≥ 0.746, Medium ≥ 0.275); `make predict` → `data/predictions.csv`; `make pipeline` runs end to end. |
| 2026-10-06 | 4.1 (+4.2) | API serving (`make run`): `/health`, `/model/info`, `/predict/single`, `/predict/batch`; thin routers + `ScoringService` (score → segment → explain → log to SQLite); model loaded once at startup, never trained in the API; fixed: missing visits with 0 time on site → 0 visits (as in training). 80 tests. |
| 2026-10-06 | 4.3 | Skipped: training stays `make train`; the API never trains. |
| 2026-10-06 | 4.4 | Drift monitoring (`make monitor`, `/monitoring/*`): feature drift (PSI), prediction drift (score PSI + segment shares), data quality (missing, unseen, defaults, rejected requests). Simulation (`make drift-demo`): stable → ok (all PSI ≤ 0.02); new-campaign shift → drift (Lead Source 2.54, time on site 1.31, High 20% → 7%). Tests no longer need saved artifacts (shared synthetic fixtures). 92 tests. |
| 2026-10-06 | 4.5 | Docker (API + Streamlit demo, one image, `models/` mounted read-only, API health check, configurable ports) and `app/demo.py` (API client: score a lead, score a CSV, monitoring). Image 6.27 GB → 3.51 GB by keeping the uv cache out of it. Checked with the real stack in Docker. |
| 2026-10-06 | 4.6 | Final documentation: README answers every design question with evidence; ADR-006 (API design), ADR-007 (monitoring); open questions closed. **v0.1.0 complete.** |
| 2026-10-06 | 5 | CI/CD: GitHub Actions runs lint + 96 tests on every push / PR (no data or secrets needed), builds the Docker image on PRs, and pushes it to `ghcr.io/aurynix/aurynix-apex` on `main`. |
| 2026-10-06 | 5 | Removed unused dependencies (xgboost, lightgbm, optuna, evidently, seaborn, pyarrow, python-multipart; shap → dev): Docker image 3.51 GB → 1.49 GB, local env 1.2 GB; every make target and the Docker stack re-checked. |
| 2026-10-06 | 5 | Outcomes: `prediction_id` + `POST /outcomes` + performance check. Simulation: behavior change with unchanged inputs → drift checks green, real PR-AUC 0.787 → 0.571, High precision 83% → 64% → retraining recommended. 101 tests. |
| 2026-10-07 | B.1 | Bank Marketing data: UCI download (nested zip), `config_bank.json` with separate paths, loader handles `;`, yes/no target, missing id; same `clean()` through config only → 45,211 × 17, 0 nulls, 11.7% positive. |
| 2026-10-07 | B.2 | Bank leakage audit: `duration` caught by the with/without check (+0.16 PR-AUC, biggest gain); also removed `campaign`, `day`, `month`, `contact` (not known before the call / time-period markers). Base 0.34 PR-AUC vs. 0.117 random. File is date-ordered (conversion 3% → 47%). |
| 2026-10-07 | B.3 | Config-driven features and explanations (lead results identical). Bank: `age_group` replaces `age` (CV PR-AUC 0.344 → 0.356); top drivers: housing loan, previous campaigns (success → 65%), marital status, age. |
| 2026-10-07 | B.4 | Bank model: test PR-AUC 0.359 (3.1× random), top 20% recall 49% / precision 29%, ECE 0.009; High 28.6% vs. Low 5.8%. Time split: ranking holds (ROC-AUC 0.707), probabilities do not (ECE 0.163). |
