# Aurynix Apex — Build Steps

The working plan for building Apex, one step at a time. Each step ends with a short summary and a check-in before moving on.

**How to use this file**
- Work top to bottom; a step starts only when the previous one is done.
- Each step lists its **goal**, **tasks**, **files**, and **done when** (the exit criteria).
- Tick the box when a step is done and add a short note under **Log** at the bottom.
- Results and metrics go into the README only after the step that produces them is complete.

Legend: `[x]` done · `[ ]` to do

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

### 1.1 Problem framing — *Stage 1*
- [ ] Define the target (`Converted`) and the **prediction moment** (lead creation, before any sales contact).
- [ ] Define the unit of prediction (one lead), the consumer (SDR team via Aurynix Pulse), and the action taken per segment.
- [ ] Define business success criteria (e.g. top 20% of leads capture ≥ X% of conversions) and ML success criteria (PR-AUC above baseline).
- [ ] List assumptions and risks (leakage, imbalance, dataset representativeness).
- **Files:** `docs/problem_framing.md`
- **Done when:** someone outside the project could read it and know exactly what is predicted, when, and how success is judged.

### 1.2 Data collection — *Stage 2*
- [ ] Download `Leads.csv` from Kaggle into `data/raw/`.
- [ ] Implement `load.py`: read raw CSV using paths from `config.json`.
- [ ] Record source, download date, row/column counts, and file hash in `docs/data_dictionary.md`.
- **Files:** `src/apex/data/load.py`, `docs/data_dictionary.md`
- **Done when:** `load_raw()` returns the DataFrame and the source is documented.

### 1.3 Data quality checks — *Stage 3 (part 1)*
- [ ] Missing values per column, including hidden nulls (`"Select"`).
- [ ] Duplicates (rows and IDs), invalid values, numeric outliers (`TotalVisits`, `Page Views Per Visit`, time on site).
- [ ] Class balance of `Converted`.
- [ ] Implement `clean.py`: replace placeholders with NaN, fix types, drop ID columns, standardize category labels.
- [ ] Unit tests for the cleaning functions.
- **Files:** `notebooks/01_eda.ipynb`, `src/apex/data/clean.py`, `tests/test_clean.py`
- **Done when:** cleaning is a tested function (not notebook code) and every issue found has a documented decision.

### 1.4 Leakage audit — *Stage 4*
- [ ] Classify **every** column: available at lead creation? yes / no / uncertain, with reasoning.
- [ ] Verify the candidates: `Tags`, `Lead Quality`, `Last Activity`, `Last Notable Activity`, score/index columns.
- [ ] Put confirmed leakage columns in `config.json → data.leakage_columns`; `clean.py` drops them.
- [ ] Quick check: train a throwaway model with and without suspect columns and compare (a big jump is evidence of leakage).
- **Files:** `docs/data_dictionary.md`, `config.json`, `docs/decisions.md` (ADR)
- **Done when:** every column has a status and the leakage decision is recorded as an ADR.

### 1.5 EDA report — *Stage 3 (part 2)*
- [ ] Univariate distributions; conversion rate by each categorical feature; numeric features vs. target.
- [ ] Correlations and redundant features.
- [ ] Save key figures to `reports/figures/`.
- [ ] Write findings and the resulting feature ideas.
- **Files:** `notebooks/01_eda.ipynb`, `reports/figures/`
- **Done when:** there is a clear list of findings that drive feature engineering.

---

## Phase 2 — Build Models (Week 2)

### 2.1 Preprocessing & feature pipeline — *Stage 5*
- [ ] `sklearn` `Pipeline` + `ColumnTransformer`: impute, scale numerics, one-hot categoricals.
- [ ] Rare-category grouping (`__other__`) using `rare_category_min_share`; unknown categories handled at inference.
- [ ] Engineered features from EDA findings.
- [ ] `make preprocess` writes cleaned data to `data/interim/` / `data/processed/`.
- [ ] Tests: pipeline fits/transforms, handles missing and unseen categories.
- **Files:** `src/apex/data/features.py`, `tests/test_features.py`, `Makefile`
- **Done when:** one pipeline object goes from raw rows to model-ready features, and it is fitted on training data only.

### 2.2 Stratified splits — *Stage 6*
- [ ] Train / validation / test split, stratified on target, seeded by `random_state`.
- [ ] Save split indices so they are reproducible.
- [ ] Rule: the test set is not touched until step 3.1.
- **Files:** `src/apex/models/train.py` (or a `split` helper)
- **Done when:** splits are fixed, reproducible, and class ratios match across splits.

### 2.3 Baselines — *Stage 7*
- [ ] Majority-class baseline.
- [ ] Logistic Regression on the full pipeline.
- [ ] Implement `evaluate.py`: PR-AUC, ROC-AUC, Brier, lift/gain at top-k.
- [ ] Set up MLflow tracking (`make mlflow-ui`).
- **Files:** `src/apex/models/evaluate.py`, `src/apex/models/train.py`, `notebooks/02_modeling.ipynb`
- **Done when:** baseline metrics are logged in MLflow and become the bar to beat.

### 2.4 Model comparison — *Stage 8*
- [ ] Random Forest, XGBoost, LightGBM with sensible defaults.
- [ ] Stratified K-fold CV in `cv.py`; `make cv`.
- [ ] Comparison table (mean ± std per metric) from MLflow.
- **Files:** `src/apex/models/cv.py`, `notebooks/02_modeling.ipynb`
- **Done when:** every candidate is compared on the same folds and logged in MLflow.

### 2.5 Imbalance handling & tuning — *Stage 9*
- [ ] Compare class weights vs. resampling (inside CV folds only).
- [ ] Optuna tuning of the top 1–2 models, optimizing CV PR-AUC.
- [ ] Record the choice in `docs/decisions.md`.
- **Files:** `src/apex/models/train.py`, `src/apex/models/cv.py`, `docs/decisions.md`
- **Done when:** a tuned model beats the baselines on validation, and the choice is justified in an ADR.

---

## Phase 3 — Evaluate & Explain (Week 3)

### 3.1 Final test evaluation — *Stage 10 (part 1)*
- [ ] Retrain the chosen model on train + validation.
- [ ] Evaluate on the test set **once**.
- [ ] PR curve, ROC curve, cumulative gain and lift charts.
- **Files:** `src/apex/models/evaluate.py`, `reports/figures/`
- **Done when:** final metrics are recorded and the README results section is filled in.

### 3.2 Probability calibration — *Stage 10 (part 2)*
- [ ] Calibration curve + Brier score.
- [ ] If needed, calibrate (isotonic / Platt) using held-out data only.
- **Files:** `src/apex/models/train.py`, `reports/figures/`
- **Done when:** a predicted 0.70 means roughly 70% actually convert (within reason).

### 3.3 Segmentation — *Stage 11*
- [ ] Implement `segment.py`: capacity-based cut-offs from `config.json → segmentation`.
- [ ] Derive score thresholds on validation scores; store them in `model_meta.json`.
- [ ] Segment table: number of leads, share of conversions captured, actual conversion rate.
- **Files:** `src/apex/models/segment.py`, `tests/test_segment.py`
- **Done when:** segments come out of config and the segment table shows that High captures far more conversions than its share of leads.

### 3.4 Explainability — *Stage 12*
- [ ] Global SHAP: summary plot and feature-importance ranking.
- [ ] Per-lead SHAP: top positive and negative contributors, mapped back to readable feature names.
- [ ] Sanity check: explanations agree with the EDA findings.
- **Files:** `src/apex/models/explain.py`, `notebooks/03_explainability.ipynb`, `reports/figures/`
- **Done when:** `explain_one(lead)` returns readable top reasons.

### 3.5 Training artifacts
- [ ] `make train` saves `models/model.pkl`, `models/model_meta.json` (version, metrics, thresholds, training date, feature list).
- [ ] Implement `reference.py`: build `models/reference_profile.json` (bin edges, category shares, missing rates, score distribution, segment shares).
- [ ] `predict.py` + `make predict` for offline batch scoring.
- **Files:** `src/apex/models/train.py`, `src/apex/models/predict.py`, `src/apex/monitoring/reference.py`
- **Done when:** `make pipeline` runs end to end from `Leads.csv` to all three artifacts.

---

## Phase 4 — Ship & Monitor (Week 4)

### 4.1 API foundation — *Stage 13 (part 1)*
- [ ] `app.py`, `schemas.py`, `dependencies.py` (load model + SHAP explainer once).
- [ ] Routers: `/health`, `/model/info`.
- [ ] `make run` / `make run-prod`.
- **Files:** `src/apex/api/`
- **Done when:** `/docs` loads and `/model/info` returns the model's metadata.

### 4.2 Prediction endpoints & logging — *Stage 13 (part 2)*
- [ ] `/predict/single`, `/predict/batch` (JSON list + CSV upload), `/predict/explain`.
- [ ] `database.py`: `predictions` table; every scored lead is logged.
- [ ] API tests with FastAPI `TestClient`.
- **Files:** `src/apex/api/routers/predict.py`, `src/apex/api/database.py`, `tests/test_api.py`
- **Done when:** the response matches the README contract and every prediction is logged.

### 4.3 Training job endpoint
- [ ] `/pipeline/train` starts a background job; `/pipeline/jobs/{id}` reports status.
- **Files:** `src/apex/api/routers/pipeline.py`

### 4.4 Drift monitoring — *Stage 14*
- [ ] `drift.py`: PSI for numeric (fixed bins) and categorical features, with smoothing for empty bins; unit tests with known values.
- [ ] `monitor.py`: `run_monitoring()` — window, `min_samples` check, feature drift + score drift, statuses.
- [ ] `drift_runs` and `feature_drift` tables; `make monitor`.
- [ ] Monitoring endpoints: `/monitoring/run`, `/latest`, `/history`, `/features/{name}`.
- [ ] Optional Evidently HTML report to `reports/monitoring/`.
- [ ] Simulation: feed shifted leads and confirm drift is detected.
- **Files:** `src/apex/monitoring/`, `src/apex/api/routers/monitoring.py`, `tests/test_drift.py`
- **Done when:** a simulated shift is flagged, and stable data is not.

### 4.5 Docker & Streamlit demo
- [ ] `make docker-build && make docker-up` serves the API with mounted models.
- [ ] `app/demo.py`: score a lead from a form, upload a CSV, view ranked leads, reasons, and drift status.
- **Files:** `Dockerfile`, `docker-compose.yml`, `app/demo.py`
- **Done when:** a fresh machine can run the API and demo from the README instructions.

### 4.6 Final documentation
- [ ] `docs/api.md`, `docs/deployment.md`, complete `docs/decisions.md`.
- [ ] README: fill in results, figures, segment table; tick roadmap boxes; update status badge.
- **Done when:** the README's "Design Decisions" questions are all answered with evidence.

---

## Phase 5 — Later

- [ ] `outcomes` table and live performance tracking (live PR-AUC, retraining trigger).
- [ ] Run the pipeline on the Bank Marketing dataset (UCI 222) and confirm `duration` is caught as leakage.
- [ ] Integration with Aurynix Pulse.
- [ ] CI (GitHub Actions: lint + test on every push).

---

## Log

| Date | Step | Notes |
|---|---|---|
| 2026-10-02 | 0.1–0.4 | README, .gitignore, project structure, and build plan created. |
| 2026-10-02 | 0.5 | uv + uv.lock (Python 3.11), dependencies installed, lint and 5 config tests passing, first commit pushed. |
