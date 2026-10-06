# Deployment

> Stage 13 (step 4.5). Files: [`Dockerfile`](../Dockerfile), [`docker-compose.yml`](../docker-compose.yml), [`app/demo.py`](../app/demo.py).

Two ways to run Apex: locally with `make`, or in Docker. Both need a trained model first.

---

## 1. Train once (on the host)

```bash
make install         # dependencies from uv.lock
make data-download   # Leads.csv from Kaggle
make pipeline        # split → final model → models/ (+ data/predictions.csv)
```

The API and the demo **never train**: they load `models/model.pkl`, `models/model_meta.json`, and `models/reference_profile.json`.

## 2a. Run locally

```bash
make run    # API  → http://localhost:8000/docs
make demo   # demo → http://localhost:8501   (in a second terminal)
```

Port taken (e.g. another project on 8000)? Use another one:

```bash
make run API_PORT=8020
make demo API_PORT=8020 DEMO_PORT=8521
```

## 2b. Run in Docker

```bash
make docker-build   # one image for the API and the demo
make docker-up      # API → http://localhost:8000/docs, demo → http://localhost:8501
make docker-down
```

Ports come from `.env` (copy `.env.example`) or the environment: `API_PORT=8020 DEMO_PORT=8521 make docker-up`.

| Service | Container | Runs | Port |
|---|---|---|---|
| `api` | `apex-api` | `uvicorn apex.api.app:app` | `${API_PORT:-8000}` → 8000 |
| `demo` | `apex-demo` | `streamlit run app/demo.py`, talks to `http://api:8000` | `${DEMO_PORT:-8501}` → 8501 |

- **The model is not inside the image.** `./models` is mounted read-only, so a new `make train` on the host is picked up by restarting the API (`docker compose restart api`), with no rebuild.
- `./data` is mounted for the SQLite log (`data/apex.db`), `./reports` for monitoring reports.
- The API has a health check (`/health`); the demo starts only when the API is healthy.
- `make docker-up` stops early with a clear message if `models/model.pkl` does not exist.

## The demo

`app/demo.py` is a Streamlit client of the API (it never loads the model itself):

| Tab | What it does |
|---|---|
| **Score a lead** | Form → probability, High / Medium / Low, reasons up and down |
| **Score a CSV** | Upload leads (API field names, or the raw Kaggle columns such as `Leads.csv`) → ranked list with segment and top reasons; download as CSV |
| **Monitoring** | Run monitoring; status, "retraining recommended", performance on real outcomes, segment shares vs. training, feature PSI, data quality |

The sidebar shows the loaded model version, training date, and test results. If the API is not reachable, the demo says so instead of failing.

## CI/CD

[`.github/workflows/ci-cd.yml`](../.github/workflows/ci-cd.yml) runs on every push to `main` and every pull request:

| Job | When | What |
|---|---|---|
| **Lint and test** | always | `uv sync --frozen` (exact versions from `uv.lock`), `make lint`, `make test` |
| **Docker image** | after tests pass | Pull request: build the image (proves the Dockerfile still works). `main`: build and push to `ghcr.io/aurynix/aurynix-apex` with tags `latest` and `sha-<commit>` |

- **No secrets or data needed:** tests use synthetic leads and an in-memory database, so CI never downloads `Leads.csv` or needs a trained model. Pushing the image uses the built-in `GITHUB_TOKEN`.
- **Fast reruns:** uv's package cache and Docker layers are cached between runs.
- **Only the latest run counts:** a new push to the same branch cancels the run still in progress.
- **The image has no model:** like locally, mount `models/` (from `make train`):

```bash
docker pull ghcr.io/aurynix/aurynix-apex:latest
docker run -p 8000:8000 -v ./models:/app/models:ro -v ./data:/app/data ghcr.io/aurynix/aurynix-apex:latest
```

The repository is private, so the image is private too: `docker login ghcr.io` with a GitHub token that has `read:packages` first.

