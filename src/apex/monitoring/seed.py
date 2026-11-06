"""Demo data, so a fresh public deployment has a real monitoring report to show.

With `SEED_DEMO_DATA=true`, API startup fills an empty database from deploy/demo_seed.json:
real leads from the held-out test set with their real outcomes, scored by the saved
model, with timestamps spread over the monitoring windows, then runs monitoring once.
The rows are marked `source = "demo_seed"`, and retraining ignores them. New rows are
added only if `predictions` is empty, so restarts never add more. Once a day at most,
the demo rows get fresh timestamps, so they stay inside the monitoring windows; nothing
else about them changes, and other rows are not touched.

    python -m apex.monitoring.seed   (or `make demo-seed-file`)

rebuilds deploy/demo_seed.json from the test split. It holds only the 7 model input
fields and the outcome: no IDs, no other columns. The final model was trained on all
leads, so performance on these rows is in-sample and slightly optimistic: the demo shows
how monitoring works, it is not new evidence of model quality.
"""

import json
import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from apex.api.database import insert_demo_rows, retime_demo_rows
from apex.api.service import ScoringService
from apex.config import load_config, path
from apex.data.split import load_splits
from apex.models.segment import assign
from apex.monitoring.monitor import Monitor

logger = logging.getLogger("uvicorn.error")  # shown in the server log at INFO

SEED_ROWS = 600
# Half of the rows in each monitoring window (config.json: 7 days, 30 days), kept
# inside the window edges. Outcomes arrive a day after the prediction.
RECENT_DAYS = (0.1, 6.5)
OLDER_DAYS = (7.5, 29.0)
OUTCOME_DELAY = timedelta(days=1)
REFRESH_AFTER = timedelta(days=1)


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


def refresh_demo_data(service: ScoringService, monitor: Monitor) -> int:
    """Move stale demo rows back into the monitoring windows and run monitoring.

    Only the timestamps of `demo_seed` rows change. Returns the rows changed.
    """
    moment = datetime.now(UTC)
    refreshed = retime_demo_rows(
        service.db,
        lambda n: timestamps(n, moment),
        (moment - REFRESH_AFTER).isoformat(timespec="seconds"),
    )
    if refreshed:
        monitor.run()
    return refreshed


def refresh_on_schedule(service: ScoringService, monitor: Monitor) -> None:
    """Refresh demo rows if `SEED_DEMO_DATA=true`."""
    if seed_enabled() and (refreshed := refresh_demo_data(service, monitor)):
        logger.info("Demo data: %s rows refreshed.", refreshed)


def seed_on_startup(service: ScoringService, monitor: Monitor) -> None:
    """Seed if `SEED_DEMO_DATA=true`. A failure is logged and never stops the API."""
    if not seed_enabled():
        return
    try:
        saved = seed_demo_data(service, monitor)
    except Exception:
        logger.exception("Demo data was not seeded; the API continues without it.")
        return
    if saved:
        logger.info("Demo data: %s rows seeded.", saved)


def build_seed_file(n: int = SEED_ROWS) -> list[dict[str, Any]]:
    """Save `n` real test-set leads and their outcomes to deploy/demo_seed.json."""
    cfg = load_config()
    test = load_splits(include_test=True)["test"].sample(
        n, random_state=cfg["split"]["random_state"]
    )
    fields = cfg["serving"]["input_fields"]
    leads = test[fields].astype(object).where(test[fields].notna(), None).to_dict(orient="records")
    rows = [
        {"lead": lead, "converted": int(converted)}
        for lead, converted in zip(leads, test[cfg["data"]["target"]], strict=True)
    ]
    file = path("demo_seed")
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(rows, indent=0) + "\n", encoding="utf-8")
    print(f"saved {len(rows)} rows ({sum(r['converted'] for r in rows)} converted): {file}")
    return rows


if __name__ == "__main__":
    build_seed_file()
