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
- `./data` is mounted for the local SQLite log (`data/apex.db`), `./reports` for monitoring reports. If `DATABASE_URL` is set in `.env`, the API uses Postgres instead.
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
| **Docker image** | after tests pass | Pull request: build the image (proves the Dockerfile still works). `main`: build and push to `ghcr.io/aurynix/aurynix-apex` with tags `latest` and `sha-<commit>`. Version tag `v0.2.0`: push as `0.2.0` |

- **No secrets or data needed:** tests use synthetic leads and an in-memory database, so CI never downloads `Leads.csv` or needs a trained model. Pushing the image uses the built-in `GITHUB_TOKEN`.
- **Fast reruns:** uv's package cache and Docker layers are cached between runs.
- **Only the latest run counts:** a new push to the same branch cancels the run still in progress.
- **The image has no model:** like locally, mount `models/` (from `make train`):

```bash
docker pull ghcr.io/aurynix/aurynix-apex:0.2.0   # or :latest
docker run -p 8000:8000 -v ./models:/app/models:ro -v ./data:/app/data ghcr.io/aurynix/aurynix-apex:latest
```

If the image is not public, run `docker login ghcr.io` first, with a GitHub token that has `read:packages`.

## Free demo on Render + Neon

[`render.yaml`](../render.yaml) creates two free web services from [`Dockerfile.render`](../Dockerfile.render): `aurynix-apex` (API) and `aurynix-apex-demo` (Streamlit). Their data lives in a free [Neon](https://neon.tech) Postgres database. This is a public demo for sample data, not a production setup.

**First deployment:**

1. In Neon, create a project in the same region as the services (Oregon: AWS US West 2) and copy its connection string.
2. In Render, choose **New → Blueprint**, select this repository, and click **Apply**. If the API's URL is not `https://aurynix-apex.onrender.com`, update `API_URL` on the demo service.
3. Set `DATABASE_URL` on the `aurynix-apex` service, under **Environment**. Render asks for it when a Blueprint is first created, but ignores `sync: false` variables on later updates, so an existing service needs it added by hand. Never put the value in this repository.

At startup the API creates its tables, and fills an empty database with [demo data](../README.md#demo-data) because `render.yaml` sets `SEED_DEMO_DATA=true`.

**Check it:**

```bash
API=https://aurynix-apex.onrender.com
curl $API/health                    # {"status":"ok",...}: does not touch the database
curl $API/health/db                 # {"status":"ok","backend":"postgresql"}
curl $API/monitoring/latest         # a report with n_samples and performance
```

`/health/db` wakes the database, so call it by hand and do not point an uptime monitor at it.

### Reset the demo data

Run these in Neon's SQL editor.

Remove only the demo rows (monitoring reports stay in `drift_runs`):

```sql
DELETE FROM outcomes WHERE prediction_id IN (SELECT id FROM predictions WHERE source = 'demo_seed');
DELETE FROM predictions WHERE source = 'demo_seed';
```

Start again with fresh timestamps, for example when the rows have aged out of the monitoring windows. This deletes **everything**, including visitors' predictions. Then restart the `aurynix-apex` service, and it seeds again because `predictions` is empty:

```sql
TRUNCATE outcomes, predictions, rejected_requests, drift_runs RESTART IDENTITY;
```

Rebuild the rows themselves with `make demo-seed-file` (needs `make split`), and commit `deploy/demo_seed.json`.

### New model

Free services have nothing to mount, so the model is part of the image. After `make train` or `make retrain`, run `make deploy-model`, commit `deploy/models/`, and merge to `main`. Do not commit a model trained on private data: this repository is public.

### Free plan limits

| Free plan limit | Effect on Apex |
|---|---|
| Sleeps after 15 minutes without traffic; waking takes about a minute | The demo and the API wake separately. The demo waits for the API (`API_WAKE_SECONDS`) and shows that it is starting |
| No persistent disk on Render | Local files are wiped on restart or sleep, including `reports/monitoring/latest.json`. Data in Neon stays, and `GET /monitoring/latest` reads from it. Without `DATABASE_URL` the API falls back to SQLite, and that data is wiped too |
| Neon storage and compute are limited | `/health` does not touch the database and the API keeps no idle connections, so Neon can scale to zero. The log grows by roughly 0.3 MB per 1,000 predictions (measured on SQLite), and anyone can send leads: check usage in the Neon dashboard |
| Seeded rows age | After 7 days a new monitoring run sees only visitors' predictions, and may report `insufficient_data`; after 30 days the seeded outcomes leave the performance window too. See [Reset the demo data](#reset-the-demo-data) |
| 750 free hours per month, shared across the workspace | Sleeping services use none, so do not keep the services awake with a pinger |
| 512 MB RAM, 0.1 CPU per service | Enough, but slower. Measured locally in Docker with these limits and Postgres on the same machine: API ready in about 24 s including the first seed, a batch of 10,000 leads 35 s, API 213 MB after that batch |
| Public URL, no login in the API | Use sample data only. Anyone can send leads and outcomes, and they now persist. Add authentication before real leads |

## Server sizing

Measured on the Docker stack (`make docker-up`, model trained on the lead data) with a load test: three batches of 10,000 leads, 1,000 single predictions sent 20 at a time, a monitoring run, and a demo session.

| | Memory | Notes |
|---|---|---|
| API at rest | 140 MB | model + explainer loaded once |
| API under load (peak) | **240 MB** | batch of 10,000 leads: 2.6 s; single predictions: ~20 per second per worker |
| Demo (Streamlit) | 50 MB | a client of the API; optional |
| Training (`make train` / `make retrain`) | **360–380 MB** peak | 4–6 s (leads 9,240 rows; bank 45,211 rows) |
| Monitoring run | within the API | 0.4 s on ~42,000 logged predictions |
| Docker image | 1.49 GB disk | no model inside |

**Recommendation:**

| Setup | RAM | CPU | Disk |
|---|---|---|---|
| Minimum (API + demo, training on the same server) | **1 GB** | 1 vCPU | 10 GB |
| Comfortable (room for OS, Docker, updates, a second API worker) | **2 GB** | 2 vCPU | 20 GB |

- Each extra API worker (`make run-prod` uses 2) adds about 240 MB.
- Disk grows with the prediction log (`data/apex.db`, or Postgres): about 0.3 MB per 1,000 predictions on SQLite.
- One worker handles about 20 single predictions per second, or thousands per second in batches, far above a typical lead volume.

