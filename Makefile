.DEFAULT_GOAL := help
.PHONY: help venv install lock data-download data-info leakage eda features split baselines train cv predict pipeline run run-prod demo \
        monitor mlflow-ui docker-build docker-up docker-down lint format test clean

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

train: ## Train model, save artifacts and reference profile
	$(todo)

cv: ## Cross-validation
	$(todo)

predict: ## Offline batch scoring
	$(todo)

pipeline: train ## Full pipeline (cleaning runs inside train)

# ---------- Serving ----------
run: ## Start the API (dev, auto-reload)
	$(todo)

run-prod: ## Start the API (prod)
	$(todo)

demo: ## Start the Streamlit demo
	$(todo)

# ---------- Monitoring ----------
monitor: ## Run drift monitoring
	$(todo)

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
