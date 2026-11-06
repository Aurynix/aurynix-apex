"""Demo data, so a fresh public deployment has a real monitoring report to show.

With `SEED_DEMO_DATA=true`, API startup fills an empty database from deploy/demo_seed.json:
synthetic leads (apex.monitoring.synthetic), scored by the saved model, with timestamps
spread over the monitoring windows, then runs monitoring once.
The rows are marked `source = "demo_seed"`, and retraining ignores them. Nothing happens
if `predictions` already has rows, so restarts never add more.

No real lead is stored in the repository. A report on these rows shows how monitoring
works; it is not evidence of model quality.
"""

import json
import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from apex.api.database import insert_demo_rows
from apex.api.service import ScoringService
from apex.config import path
from apex.models.segment import assign
from apex.monitoring.monitor import Monitor

logger = logging.getLogger("uvicorn.error")  # shown in the server log at INFO

# Half of the rows in each monitoring window (config.json: 7 days, 30 days), kept
# inside the window edges. Outcomes arrive a day after the prediction.
RECENT_DAYS = (0.1, 6.5)
OLDER_DAYS = (7.5, 29.0)
OUTCOME_DELAY = timedelta(days=1)


def seed_enabled() -> bool:
    return os.environ.get("SEED_DEMO_DATA", "").strip().lower() == "true"


def load_seed_rows() -> list[dict[str, Any]]:
    """The rows of deploy/demo_seed.json: [{"lead": {...}, "converted": 0 or 1}, ...]."""
    with path("demo_seed").open(encoding="utf-8") as f:
        return json.load(f)


def timestamps(n: int, moment: datetime) -> tuple[list[str], list[str]]:
    """(created_at, recorded_at) for `n` rows, spread over the monitoring windows."""
    recent = n // 2
    ages = np.concatenate([np.linspace(*RECENT_DAYS, recent), np.linspace(*OLDER_DAYS, n - recent)])
    created = [moment - timedelta(days=float(age)) for age in ages]
    recorded = [min(when + OUTCOME_DELAY, moment) for when in created]
    return (
        [when.isoformat(timespec="seconds") for when in created],
        [when.isoformat(timespec="seconds") for when in recorded],
    )


def seed_demo_data(
    service: ScoringService, monitor: Monitor, rows: list[dict[str, Any]] | None = None
) -> int:
    """Fill an empty database with demo rows and run monitoring. Returns the rows saved."""
    if service.db.scalar("SELECT COUNT(*) FROM predictions"):
        return 0
    rows = load_seed_rows() if rows is None else rows
    leads = [row["lead"] for row in rows]
    # The same steps as ScoringService.score, which always logs at the current time.
    frame = pd.DataFrame(leads, columns=service.meta["input_fields"])
    scores = service.model.predict_proba(frame)[:, 1].round(4)
    segments = np.char.lower(assign(scores, service.meta["segments"]["thresholds"]))
    created, recorded = timestamps(len(rows), datetime.now(UTC))
    saved = insert_demo_rows(
        service.db,
        leads,
        scores.tolist(),
        segments.tolist(),
        [row["converted"] for row in rows],
        created,
        recorded,
        service.version,
    )
    if saved:
        monitor.run()
    return saved


def seed_on_startup(service: ScoringService, monitor: Monitor) -> None:
    """Seed if `SEED_DEMO_DATA=true`. A failure is logged and never stops the API."""
    if not seed_enabled():
        return
    try:
        saved = seed_demo_data(service, monitor)
    except Exception:
        logger.exception("Demo data was not seeded; the API continues without it.")
        return
    logger.info("Demo data: %s rows seeded.", saved)
