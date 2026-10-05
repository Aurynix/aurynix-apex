.DEFAULT_GOAL := help
.PHONY: help venv install lock data-download data-info leakage eda features split baselines evaluate calibration segments explain train cv tune predict pipeline run run-prod demo \
        monitor drift-demo mlflow-ui docker-build docker-up docker-down lint format test clean

# All commands run inside the uv-managed .venv (requires https://docs.astral.sh/uv/)
RUN := uv run
export MLFLOW_DISABLE_AGENT_HINT := 1

# Placeholder for targets whose stage is not implemented yet.
todo = @echo "⏳ '$@' is not implemented yet (see BUILD_STEPS.md)."

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ---------- Environment ----------
venv: ## Create a Python 3.11 virtual environment in .venv
	@test -d .venv && echo ".venv already exists" || uv venv --python 3.11

install: ## Install locked dependencies (incl. dev) and the apex package
	uv sync --frozen

lock: ## Re-resolve dependencies and update uv.lock
	uv lock

# ---------- Pipeline ----------
data-download: ## Download the Kaggle dataset into data/raw/
	$(RUN) python -m apex.data.download

data-info: ## Print shape, hash, and target rate of data/raw/Leads.csv
	$(RUN) python -m apex.data.load

leakage: ## Compare a quick model with and without leakage suspects
	$(RUN) python -m apex.data.leakage

eda: ## Print EDA tables and save figures to reports/figures/
	$(RUN) python -m apex.data.eda

features: ## Fit the feature pipeline on all leads and list the features
	$(RUN) python -m apex.data.features

split: ## Create the fixed train / validation / test split (data/splits.csv)
	$(RUN) python -m apex.data.split

baselines: ## Fit no-skill + Logistic Regression baselines, log to MLflow
	$(RUN) python -m apex.models.train baselines

evaluate: ## Step 3.1: retrain on train + val, score the test set ONCE, save figure
	$(RUN) python -m apex.models.train test

calibration: ## Step 3.2: out-of-fold calibration check (raw / Platt / isotonic), save figure
	$(RUN) python -m apex.models.train calibration

segments: ## Step 3.3: derive High / Medium / Low thresholds and print segment tables
	$(RUN) python -m apex.models.segment

explain: ## Step 3.4: global feature importance, figure, and example per-lead reasons
	$(RUN) python -m apex.models.explain

train: ## Fit the final model on all leads; save model, meta, reference profile, model card
	$(RUN) python -m apex.models.train final

cv: ## 5-fold CV on the train split: all vs. selected features, log to MLflow
	$(RUN) python -m apex.models.cv features

tune: ## Grid search over Logistic Regression C and class_weight (5-fold CV), log to MLflow
	$(RUN) python -m apex.models.cv tune

predict: ## Score data/raw/Leads.csv with the saved model → data/predictions.csv
	$(RUN) python -m apex.models.predict

pipeline: split train predict ## Leads.csv → split → final model + artifacts → predictions

# ---------- Serving ----------
run: ## Start the API (dev, auto-reload) → http://localhost:8000/docs (needs `make train`)
	$(RUN) uvicorn apex.api.app:app --reload --port 8000

run-prod: ## Start the API (prod, 2 workers)
	$(RUN) uvicorn apex.api.app:app --host 0.0.0.0 --port 8000 --workers 2

demo: ## Start the Streamlit demo
	$(todo)

# ---------- Monitoring ----------
monitor: ## Drift monitoring on the last 7 days of API predictions (data/apex.db)
	$(RUN) python -m apex.monitoring.monitor

drift-demo: ## Simulate stable vs. drifted traffic and show that the monitor flags only the drift
	$(RUN) python -m apex.monitoring.simulate

mlflow-ui: ## Open MLflow at http://localhost:5000
	$(RUN) mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# ---------- Docker ----------
docker-build: ## Build the Docker image
	docker compose build

docker-up: ## Start containers
	docker compose up -d

docker-down: ## Stop containers
	docker compose down

# ---------- Quality ----------
lint: ## Lint with Ruff
	$(RUN) ruff check src tests app
	$(RUN) ruff format --check src tests app

format: ## Format with Ruff
	$(RUN) ruff format src tests app
	$(RUN) ruff check --fix src tests app

test: ## Run tests
	$(RUN) pytest

clean: ## Remove caches and build files
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
