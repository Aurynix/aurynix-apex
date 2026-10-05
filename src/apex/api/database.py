"""SQLite log of every scored lead (`config.json → paths.database`).

One table for now: `predictions`. Drift monitoring (step 4.4) reads it later.
"""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    model_version TEXT NOT NULL,
    lead          TEXT NOT NULL,
    score         REAL NOT NULL,
    segment       TEXT NOT NULL
)
"""


def connect(db_path: Path) -> sqlite3.Connection:
    """Open the database (created on first use) and make sure the table exists."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute(SCHEMA)
    return conn


def log_predictions(
    conn: sqlite3.Connection,
    rows: list[dict],
    scores: list[float],
    segments: list[str],
    version: str,
) -> None:
    """Save one row per scored lead: the input fields, the score, and the segment."""
    now = datetime.now(UTC).isoformat(timespec="seconds")
    conn.executemany(
        "INSERT INTO predictions (created_at, model_version, lead, score, segment) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (now, version, json.dumps(row), score, segment)
            for row, score, segment in zip(rows, scores, segments, strict=True)
        ],
    )
    conn.commit()
