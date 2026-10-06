# API Reference

> Stage 13 (step 4.1). Code: [`src/apex/api/`](../src/apex/api/).

```bash
make train   # once: saves models/model.pkl + models/model_meta.json
make run     # dev, auto-reload → http://localhost:8000/docs
make run-prod  # production: 0.0.0.0:8000, 2 workers
```

At startup the API **loads** the saved model, its metadata, and the explainer, once. It never trains: training is a separate workflow (`make train`). If `models/model.pkl` is missing, startup stops with "Run `make train` first."

---

## Design

```
HTTP request
     │
     ▼
  Router            validate the request (schemas.py), nothing else
     │
     ▼
ScoringService      all the logic (service.py)
     ├── predict        probability from the saved pipeline
     ├── segment        High / Medium / Low from the saved thresholds
     ├── explain        top reasons up / down
     └── log            one row per lead in SQLite (data/apex.db)
     │
     ▼
   JSON
```

| File | Responsibility |
|---|---|
| `app.py` | FastAPI app, startup (lifespan), `/health`, router registration |
| `schemas.py` | Pydantic request / response models: validation only |
| `dependencies.py` | Load the model once, create the `ScoringService`, inject it into routers |
| `service.py` | Business logic: score, segment, explain, log, model info |
| `database.py` | SQLite `predictions` table |
| `routers/model.py`, `routers/predict.py` | Endpoints: validate → service → response |

Routers do not know how the model works.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check and loaded model version |
| `GET` | `/model/info` | Model version, training date, input fields, segment thresholds, test metrics |
| `POST` | `/predict/single` | Score one lead: score, segment, reasons |
| `POST` | `/predict/batch` | Score up to 10,000 leads (results keep input order) |
| `POST` | `/outcomes` | Send real results (converted or not) by `prediction_id` |
| `POST` | `/monitoring/run` | Run monitoring now: drift, data quality, performance ([monitoring.md](monitoring.md)) |
| `GET` | `/monitoring/latest` | Latest drift report |
| `GET` | `/monitoring/history` | Status and key numbers of past runs |

Reasons are part of every prediction, so a client never has to call a second endpoint to explain a score.

## Lead (request)

| Field | Type | Required | Notes |
|---|---|---|---|
| `lead_origin` | string | **yes** | e.g. `API`, `Landing Page Submission`, `Lead Add Form` |
| `lead_source` | string | no | e.g. `Google`, `Direct Traffic`, `Reference` |
| `do_not_email` | bool | no (`false`) | Lead opted out of emails |
| `total_visits` | int ≥ 0 | no | Website visits. If missing and `time_on_website` is 0, it is 0 |
| `time_on_website` | int ≥ 0 | no (`0`) | Seconds on the website |
| `specialization` | string | no | Only "given" vs. "missing" matters to the model |
| `occupation` | string | no | e.g. `Working Professional`, `Unemployed`, `Student` |

Unknown fields and negative numbers are rejected with `422`, and each rejected request is logged in `rejected_requests` (a data-quality signal for monitoring). Unknown category values (e.g. a new lead source) are accepted: the model treats them as rare.

## Examples

```bash
curl -X POST localhost:8000/predict/single -H 'content-type: application/json' -d '{
  "lead_origin": "Landing Page Submission",
  "lead_source": "Google",
  "total_visits": 3,
  "time_on_website": 1200,
  "specialization": "Finance Management",
  "occupation": "Working Professional"
}'
```

```json
{
  "prediction_id": 1042,
  "score": 0.9568,
  "segment": "high",
  "reasons": {
    "up": [
      "Occupation = Working Professional",
      "Website activity = 3 visits, 20 min on site",
      "Specialization = Given"
    ],
    "down": ["Lead origin = Landing Page Submission"]
  },
  "model_version": "apex-v0.1.0"
}
```

```bash
curl -X POST localhost:8000/predict/batch -H 'content-type: application/json' \
  -d '{"leads": [{"lead_origin": "API", "time_on_website": 30, "total_visits": 1, "do_not_email": true}]}'
```

```json
{
  "count": 1,
  "predictions": [
    {
      "prediction_id": 1043,
      "score": 0.0134,
      "segment": "low",
      "reasons": {
        "up": ["Lead origin = API", "Lead source = Missing"],
        "down": [
          "Website activity = 1 visit, 0 min on site",
          "Opted out of email = yes",
          "Occupation = Missing"
        ]
      },
      "model_version": "apex-v0.1.0"
    }
  ]
}
```

## Prediction log

Every scored lead is saved in `data/apex.db` (`config.json → paths.database`), table `predictions`:

| Column | Example |
|---|---|
| `id` | 1 |
| `created_at` | `2026-10-06T00:00:00+00:00` (UTC) |
| `model_version` | `apex-v0.1.0` |
| `lead` | the 7 input fields as JSON (raw column names) |
| `score` | 0.9568 |
| `segment` | `high` |

Drift monitoring reads this table ([monitoring.md](monitoring.md)).

## Outcomes

Every prediction returns a `prediction_id`. When the real result is known, send it back:

```bash
curl -X POST localhost:8000/outcomes -H 'content-type: application/json' \
  -d '{"outcomes": [{"prediction_id": 1042, "converted": true}, {"prediction_id": 1043, "converted": false}]}'
```

```json
{"saved": 2}
```

- Up to 10,000 outcomes per request. Sending an outcome again for the same prediction replaces it.
- If any `prediction_id` is unknown, **nothing** is saved and the API answers `404` with `{"detail": {"message": "Unknown prediction_id", "unknown": [...]}}`.
- Saved in table `outcomes` (`prediction_id`, `converted`, `recorded_at`) and used by the performance check in monitoring.
