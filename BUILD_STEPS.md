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
- **Files:** `src/apex/models/explain.py`, `reports/figures/`
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
