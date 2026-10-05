# Aurynix Apex — Lead Propensity Engine

> Surface the apex of your pipeline: predict which leads will convert, so sales teams call the right people first.

![Status](https://img.shields.io/badge/status-in%20progress-orange)
![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![License](https://img.shields.io/badge/license-MIT-green)

**Aurynix Apex** is an end-to-end machine learning system that:

1. Scores every lead with a **probability of conversion**.
2. Groups leads into **High / Medium / Low** priority segments based on sales-team capacity.
3. Explains **why** each lead received its score.
4. Monitors itself in production for **data drift** and **score drift**, and signals when retraining is needed.

It is the scoring engine behind **[Aurynix Pulse](#how-it-fits-into-aurynix-pulse)**, an AI lead qualification platform.

> 🚧 **Under active development.** This README describes the design and roadmap. Results and metrics will be added only after each stage is completed and validated.

---

## Table of Contents

- [The Problem](#the-problem)
- [The Solution](#the-solution)
- [How It Fits Into Aurynix Pulse](#how-it-fits-into-aurynix-pulse)
- [Dataset](#dataset)
- [System Architecture](#system-architecture)
- [ML Pipeline](#ml-pipeline)
- [Data Leakage Policy](#data-leakage-policy)
- [Evaluation Strategy](#evaluation-strategy)
- [Lead Segmentation](#lead-segmentation)
- [Explainability](#explainability)
- [API](#api)
- [Monitoring: Data Drift & Score Drift](#monitoring-data-drift--score-drift)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Make Targets](#make-targets)
- [Configuration](#configuration)
- [Roadmap](#roadmap)
- [Design Decisions](#design-decisions)
- [Author](#author)
- [License](#license)

---

## The Problem

A company receives hundreds or thousands of leads every month from ads, web forms, and sign-ups. Its sales team (SDRs) can only contact a fraction of them.

Without data-driven prioritization:

- Time is wasted on leads who were "just asking."
- High-intent buyers wait too long and go to a competitor.
- Manual rule-based scoring in CRMs (e.g. *"+10 points for visiting the pricing page"*) relies on guesswork, not evidence.
- Nobody notices when the type of incoming leads changes and the scoring rules quietly stop working.

## The Solution

Aurynix Apex learns from historical leads whose outcome is known, then predicts the outcome of new ones.

For every lead it returns:

| Output | Example |
|---|---|
| Conversion probability | `0.82` |
| Priority segment | `High` |
| Top reasons | Visited the website 6 times · Came from Google Ads · Spent 25 minutes on site |
| Model version | `apex-v1.0.0` |

**Example ranking:**

| Lead | Source | Visits | Time on site | Score | Segment |
|---|---|---|---|---|---|
| Lead A | Google Ads | 6 | 25 min | 0.82 | 🟢 High |
| Lead B | Organic search | 3 | 8 min | 0.41 | 🟡 Medium |
| Lead C | Facebook | 1 | 2 min | 0.09 | 🔴 Low |

*(Illustrative example, not model output.)*

## How It Fits Into Aurynix Pulse

```
 WhatsApp ─┐
 Web Form ─┼──▶  Aurynix Pulse  ──▶  Aurynix Apex  ──▶  Ranked leads + reasons
 CSV       ─┘    (collects &          (scores,            for the sales team
                  unifies leads)       explains,
                                       monitors)
                        ▲                                      │
                        └──── sales outcomes & corrections ◀───┘
                              (become new training labels)
```

| Product | Role |
|---|---|
| **Aurynix Nexus** | AI chat and dashboard platform |
| **Aurynix Pulse** | Collects leads from every channel into one unified profile |
| **Aurynix Apex** | Scores, ranks, and explains leads, and monitors model health |

## Dataset

### Primary: Lead Scoring Dataset (X Education)
- Source: Kaggle, [`amritachatterjee09/lead-scoring-dataset`](https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset) (saved as `data/raw/Leads.csv` by `make data-download`)
- About 9,000 real leads from an online education company
- Features: lead origin and source, website activity, demographics, engagement
- Target: `Converted` (1 = converted, 0 = did not convert)

**Known data issues to handle:**
- The placeholder value `"Select"` means the user did not choose an option. It is treated as missing.
- Several columns may be filled in by sales **after** contact (see [Data Leakage Policy](#data-leakage-policy)).

### Secondary (planned): Bank Marketing Dataset (UCI, id 222)
- Used to prove the pipeline is reusable on a second, highly imbalanced dataset.
- Contains a well-known leakage feature (`duration`, the call length, only known after the call).

> Raw data is never committed to this repository.

## System Architecture

```
                         ┌──────────────────── Training ────────────────────┐
  data/raw/Leads.csv ──▶ │ clean ─▶ features ─▶ train ─▶ evaluate ─▶ save   │
                         └──────────────────────────────────────┬───────────┘
                                                                │
                              models/model.pkl                  │
                              models/model_meta.json   ◀────────┤
                              models/reference_profile.json ◀───┘
                                       │
                                       ▼
  Client / Pulse ──▶  FastAPI  ──▶  predict + segment + explain
                         │
                         ▼
                 SQLite: predictions ──▶ monitor.py ──▶ drift_runs
                                                    └──▶ feature_drift
                                                    └──▶ reports/monitoring/*.html
```

## ML Pipeline

| # | Stage | Key Activities | Output |
|---|---|---|---|
| 1 | **Problem Framing** | Define target, prediction moment, business success criteria | `docs/problem_framing.md` |
| 2 | **Data Collection** | Download and document raw datasets | `data/raw/` |
| 3 | **Data Quality & EDA** | Missing values, hidden nulls (`"Select"`), duplicates, outliers, class balance, feature–target relationships | EDA & Data Quality Report |
| 4 | **Leakage Audit** | Classify every column as available or not available at lead creation | `docs/data_dictionary.md` |
| 5 | **Preprocessing & Feature Engineering** | `sklearn` `Pipeline` + `ColumnTransformer`, rare-category grouping, engineered features | `src/apex/data/` |
| 6 | **Data Splitting** | Stratified train / validation / test; test set used once | Fixed, seeded splits |
| 7 | **Baseline** | Majority-class baseline and Logistic Regression | Baseline metrics |
| 8 | **Cross-validation & Feature Selection** | Stratified 5-fold CV; keep the simplest feature set that scores the same; tracked in MLflow | Selected features |
| 9 | **Tuning** | Grid search over Logistic Regression `C` and class weights, with cross-validation | Tuned model |
| 10 | **Evaluation & Calibration** | PR-AUC, ROC-AUC, lift/gain, calibration curve, Brier score | Validation Report |
| 11 | **Segmentation** | Map probabilities to High / Medium / Low using sales capacity | Segmentation Methodology |
| 12 | **Explainability** | Global and per-lead SHAP explanations | Explainability Report |
| 13 | **Serving** | FastAPI service, prediction logging, Docker, Streamlit demo | Live API & demo |
| 14 | **Monitoring & Retraining** | Data drift and score drift (PSI), alerts, retraining entry point | Monitoring tables & reports |

## Data Leakage Policy

Every feature must pass one test:

> **Is this information available at the moment the lead is created, before any sales contact?**

If not, it is removed and documented. Candidate leakage columns to verify:

| Column | Risk | Status |
|---|---|---|
| `Tags` | Assigned by sales after contact | To be verified |
| `Lead Quality` | Sales rep's judgment after contact | To be verified |
| `Last Activity` | May include post-contact activity | To be verified |
| `Last Notable Activity` | May include post-contact activity | To be verified |

A model that suddenly scores 95%+ is treated as a leakage warning, not a success.

## Evaluation Strategy

Conversion data is typically imbalanced, so accuracy alone is misleading.

| Metric | Why it matters |
|---|---|
| **PR-AUC** (primary) | Focuses on finding the positive class (converters) |
| **ROC-AUC** | Overall ranking quality |
| **Lift / Cumulative Gain** | Business view: *"the top 20% of scored leads capture X% of conversions"* |
| **Calibration curve & Brier score** | A predicted 0.70 should mean roughly 70% actually convert |
| **Precision / Recall at threshold** | Tied to how many leads the team can actually contact |

**Rules:**
- Every model must beat the baseline to be considered.
- The test set is evaluated exactly once, at the end.
- All experiments are logged in MLflow (parameters, metrics, artifacts).

## Lead Segmentation

Segments are based on **sales-team capacity**, not arbitrary probability cut-offs.

Example: if the team can contact 20% of leads, the top 20% of scores are **High**.

| Segment | Action |
|---|---|
| 🟢 High | Contact first, same day |
| 🟡 Medium | Follow up within the week |
| 🔴 Low | Automated nurturing (email sequences) |

The final report will include a table showing, for each segment: number of leads, share of all conversions captured, and actual conversion rate.

## Explainability

- **Global:** SHAP summary plots showing which features drive conversion overall.
- **Per lead:** the top positive and negative contributors behind each individual score, returned by the API so sales reps can trust and act on the score.

## API

Built with FastAPI and split into routers.

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/model/info` | Model version, metrics, training date |
| `POST` | `/predict/single` | Score one lead |
| `POST` | `/predict/batch` | Score a CSV or a list of leads |
| `POST` | `/predict/explain` | SHAP explanation for one lead |
| `POST` | `/pipeline/train` | Start a training job |
| `GET` | `/pipeline/jobs/{id}` | Check job status |
| `POST` | `/monitoring/run` | Run drift monitoring now |
| `GET` | `/monitoring/latest` | Latest drift result |
| `GET` | `/monitoring/history` | Score drift over time |
| `GET` | `/monitoring/features/{name}` | Drift history for one feature |

Interactive docs at `http://localhost:8000/docs` once the API is running.

**Planned response for `/predict/single`:**

```json
{
  "probability": 0.82,
  "segment": "High",
  "model_version": "apex-v1.0.0",
  "top_reasons": [
    { "feature": "TotalVisits", "impact": 0.21 },
    { "feature": "Lead Source=Google", "impact": 0.14 },
    { "feature": "Total Time Spent on Website", "impact": 0.11 }
  ]
}
```

## Monitoring: Data Drift & Score Drift

A model trained on last year's leads can silently degrade when the type of incoming leads changes. Apex tracks this by comparing **recent leads** against a **reference profile** saved at training time.

### Reference profile

Saved by `train.py` to `models/reference_profile.json`:

- **Numeric features:** bin edges (from training quantiles) and the share of data in each bin
- **Categorical features:** the share of each known category
- **Missing rate** per feature
- **Score distribution** and High / Medium / Low shares on the validation set
- **Model version**

Bin edges are fixed at training time and reused for all future comparisons.

### Metric: Population Stability Index (PSI)

| PSI | Status | Action |
|---|---|---|
| < 0.10 | 🟢 Stable | None |
| 0.10 – 0.25 | 🟡 Warning | Investigate |
| > 0.25 | 🔴 Drift | Consider retraining |

- **Data drift:** PSI per feature, plus changes in missing rate and the share of unseen categories (grouped as `__other__`).
- **Score drift:** PSI of the predicted probability distribution, plus changes in mean score and High-segment share.
- A minimum of **200 predictions** is required per run; otherwise the run is marked `insufficient_data`.

### Storage (SQLite)

| Table | One row per | Key columns |
|---|---|---|
| `predictions` | Scored lead | timestamp, model_version, features (JSON), probability, segment |
| `drift_runs` | Monitoring run | window, n_samples, **score_psi**, mean score (ref/cur), High share (ref/cur), n_features_drifted, status |
| `feature_drift` | Feature per run | feature, type, **psi**, missing rate (ref/cur), unseen_share, status |
| `outcomes` *(planned)* | Known result | prediction_id, converted, recorded_at |

### Running it

```bash
make monitor            # analyze the last 7 days of predictions
```

Optional Evidently HTML reports are saved to `reports/monitoring/`. Core PSI values are computed by the project's own code and stored in SQLite.

### Next step: real performance

Drift shows that the **data** changed, not that the model is **wrong**. Once conversion outcomes arrive (from Aurynix Pulse), they are stored in `outcomes` and used to compute live PR-AUC and trigger retraining.

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| Data | pandas, NumPy, PyArrow |
| Modeling | scikit-learn (Logistic Regression) |
| Tuning | scikit-learn grid search |
| Experiment Tracking | MLflow |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn |
| API | FastAPI, Uvicorn, Pydantic |
| Persistence | SQLite |
| Monitoring | Custom PSI, Evidently (reports) |
| Demo | Streamlit |
| Container | Docker, Docker Compose |
| Environment | uv (`pyproject.toml` + `uv.lock`) |
| Code Quality | Ruff, pytest |

## Project Structure

```
aurynix-apex/
├── data/                          # git-ignored
│   └── raw/                       # Leads.csv (the only copy; cleaned in memory)
├── docs/
│   ├── problem_framing.md
│   ├── data_dictionary.md         # sources & leakage audit
│   ├── data_quality.md            # data quality findings & decisions
│   ├── data_cleaning.md           # cleaning steps & why
│   ├── eda.md                     # EDA findings & feature ideas
│   ├── features.md                # feature pipeline & decisions
│   ├── splits.md                  # train / val / test split
│   ├── models.md                  # metrics & model results
│   ├── decisions.md               # architecture decision records
│   ├── api.md
│   └── deployment.md
├── notebooks/                     # local scratch for testing (git-ignored)
├── src/
│   └── apex/
│       ├── config.py              # loads config.json
│       ├── data/
│       │   ├── load.py
│       │   ├── clean.py
│       │   └── features.py
│       ├── models/
│       │   ├── train.py
│       │   ├── cv.py
│       │   ├── evaluate.py
│       │   ├── segment.py
│       │   ├── explain.py
│       │   └── predict.py
│       ├── monitoring/
│       │   ├── reference.py       # build reference_profile.json
│       │   ├── drift.py           # PSI functions & thresholds
│       │   └── monitor.py         # run_monitoring()
│       └── api/
│           ├── app.py
│           ├── schemas.py
│           ├── dependencies.py    # model & SHAP explainer singletons
│           ├── database.py        # SQLite tables
│           └── routers/
│               ├── health.py
│               ├── model.py
│               ├── predict.py
│               ├── pipeline.py
│               └── monitoring.py
├── app/
│   └── demo.py                    # Streamlit demo
├── models/                        # git-ignored artifacts
├── reports/
│   ├── figures/
│   └── monitoring/
├── tests/
├── config.json
├── .env.example
├── Makefile
├── pyproject.toml                 # package metadata & dependencies
├── uv.lock                        # pinned dependency versions
├── Dockerfile
├── .dockerignore
├── docker-compose.yml
├── BUILD_STEPS.md                 # step-by-step build plan
├── LICENSE
└── README.md
```

## Getting Started

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.11 automatically if needed).

```bash
git clone https://github.com/Aurynix/aurynix-apex.git
cd aurynix-apex

make venv                          # create .venv with Python 3.11
make install                       # install locked dependencies from uv.lock
source .venv/bin/activate          # optional; make targets use `uv run`
```

Download the dataset from Kaggle into `data/raw/` (no Kaggle login needed for this public dataset):

```bash
make data-download   # → data/raw/Leads.csv + data/raw/Leads Data Dictionary.xlsx
make data-info       # print shape, SHA-256, and conversion rate
```

```bash
make pipeline      # clean + train
make run           # start the API → http://localhost:8000/docs
make demo          # start the Streamlit demo
```

> Commands become functional as each stage is implemented.

## Make Targets

Run `make help` for the full list.

| Group | Command | Description |
|---|---|---|
| Environment | `make venv` / `make install` / `make lock` | Create venv / install locked dependencies / update `uv.lock` |
| Data | `make data-download` | Download the Kaggle dataset into `data/raw/` |
| | `make data-info` | Print shape, hash, and target rate of the raw data |
| | `make leakage` | Compare a quick model with and without leakage suspects |
| | `make eda` | Print EDA tables and save figures to `reports/figures/` |
| | `make features` | Fit the feature pipeline and list the features |
| | `make split` | Create the fixed train / validation / test split |
| | `make baselines` | Fit no-skill + Logistic Regression, log to MLflow |
| | `make train` | Train model, save artifacts and reference profile |
| | `make cv` | 5-fold CV: all vs. selected features, logged to MLflow |
| | `make predict` | Offline batch scoring |
| | `make pipeline` | Clean + train |
| Serving | `make run` / `make run-prod` | Start API (dev / prod) |
| | `make demo` | Start Streamlit demo |
| Monitoring | `make monitor` | Run drift monitoring |
| | `make mlflow-ui` | Open MLflow at `http://localhost:5000` |
| Docker | `make docker-build` / `make docker-up` / `make docker-down` | Build / start / stop |
| Quality | `make lint` / `make format` / `make test` / `make clean` | Ruff, pytest, cleanup |

## Configuration

All paths, filenames, thresholds, and constants live in `config.json` and are loaded by `src/apex/config.py`. No hardcoded paths anywhere in the codebase.

Key settings: data paths, `target = "Converted"`, `random_state = 42`, segment capacity shares, PSI thresholds, monitoring window, and minimum sample size.

## Roadmap

### Week 1: Understand & Explore
- [x] Project setup (structure, environment, tooling)
- [x] Problem framing document
- [x] Download and inspect the dataset
- [x] Data quality checks and cleaning
- [x] Leakage audit and data dictionary
- [x] EDA report

### Week 2: Build Models
- [x] Preprocessing and feature pipeline
- [x] Stratified splits
- [x] Baselines (majority class, Logistic Regression)
- [x] Cross-validation and feature selection
- [ ] Tuning (`C`, class weights)

### Week 3: Evaluate & Explain
- [ ] Final test-set evaluation
- [ ] Probability calibration
- [ ] High / Medium / Low segmentation
- [ ] SHAP explainability

### Week 4: Ship & Monitor
- [ ] FastAPI service with prediction logging
- [ ] Reference profile at training time
- [ ] Data drift and score drift monitoring
- [ ] Docker and Streamlit demo
- [ ] Final documentation

### Later
- [ ] `outcomes` table and live performance tracking
- [ ] Apply the pipeline to the Bank Marketing dataset
- [ ] Integration with Aurynix Pulse

## Design Decisions

Documented in `docs/decisions.md` as the project progresses. This section will answer:

- Why the final model was chosen over the alternatives
- How data leakage was detected and prevented
- How class imbalance was handled
- How the model is shown to improve lead prioritization
- How scores are used operationally by sales teams
- How the model is monitored and when it is retrained

## Author

**Yazan Al-Sedih**

Part of the **Aurynix** product family: Nexus · Pulse · Apex

## License

MIT
