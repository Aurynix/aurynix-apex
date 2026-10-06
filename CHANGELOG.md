# Changelog

Software versions of Aurynix Apex. The **model version** (`apex-v0.1.0`, in `config.json`) is separate: it changes only when the model itself changes (new data or new settings).

## v0.2.0 (2026-10-07)

**Added**
- **Real outcomes and performance monitoring**: every prediction returns a `prediction_id`; `POST /outcomes` records whether a lead converted; monitoring compares real PR-AUC and High-segment precision with the test results and sets `retrain_recommended` ([ADR-008](docs/decisions.md)). `make outcomes-demo`.
- **Second dataset: Bank Marketing** (UCI, 45,211 clients): the same pipeline with configuration only (`APEX_CONFIG=config_bank.json`). The leakage check catches `duration` by itself; test PR-AUC 0.359 (3.1× random); `make time-split` shows probabilities drift over time ([docs/bank_marketing.md](docs/bank_marketing.md), [ADR-009](docs/decisions.md)).
- **CI/CD** (GitHub Actions): lint and tests on every push and pull request; Docker image built on pull requests and pushed to `ghcr.io/aurynix/aurynix-apex` on `main` (`latest`, `sha-…`) and on version tags (`0.2.0`).
- Config-driven building blocks: download from Kaggle or a URL (nested zips), CSV format and target labels, leakage suspects, features (`presence_only`, `ratios`, `flags`, `bins`, `drop`), explanation labels and groups, MLflow experiment, optional model card.
- `CHANGELOG.md`; the API reports the software version from `pyproject.toml`.

**Changed**
- Removed unused dependencies (`xgboost`, `lightgbm`, `optuna`, `evidently`, `seaborn`, `pyarrow`, `python-multipart`; `shap` moved to dev): Docker image **3.51 GB → 1.49 GB**.
- Prediction responses include `prediction_id`.
- Config keys: `files.raw_leads` → `files.raw_data`, `source.kaggle_files` → `source.files`.

**Unchanged**
- The lead model: same data, settings, and results (test PR-AUC 0.787), version `apex-v0.1.0`.

## v0.1.0 (2026-10-06)

First complete version: lead scoring from data to monitoring ([BUILD_STEPS.md](BUILD_STEPS.md) phases 0–4).
- Data quality, cleaning, and a leakage audit (9 post-contact columns removed).
- Logistic Regression on 24 features from 7 lead fields; test PR-AUC 0.787, top 20% of leads hold 43% of buyers; calibrated probabilities; High / Medium / Low segments; per-lead reasons; model card.
- FastAPI service (`/predict/single`, `/predict/batch`, `/model/info`), prediction log, drift monitoring (PSI), Streamlit demo, Docker.
