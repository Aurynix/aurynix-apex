"""SQLite storage (`config.json → paths.database`).

Tables:
- `predictions`: every scored lead (input fields, score, segment)
- `rejected_requests`: requests the API refused as invalid (422), for data quality
- `drift_runs`: one row per monitoring run, with the full report as JSON
- `outcomes`: the real result of a scored lead (converted or not), sent later
"""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    model_version TEXT NOT NULL,
    lead          TEXT NOT NULL,
    score         REAL NOT NULL,
    segment       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rejected_requests (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    path       TEXT NOT NULL,
    errors     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outcomes (
    prediction_id INTEGER PRIMARY KEY REFERENCES predictions(id),
    converted     INTEGER NOT NULL CHECK (converted IN (0, 1)),
    recorded_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS drift_runs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    n_samples  INTEGER NOT NULL,
    status     TEXT NOT NULL,
    score_psi  REAL,
    report     TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def init_db(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Create the tables if they do not exist yet."""
    conn.executescript(SCHEMA)
    return conn


def connect(db_path: Path) -> sqlite3.Connection:
    """Open the database (created on first use) with all tables."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return init_db(sqlite3.connect(db_path, check_same_thread=False))


def log_predictions(
    conn: sqlite3.Connection,
    rows: list[dict],
    scores: list[float],
    segments: list[str],
    version: str,
) -> list[int]:
    """Save one row per scored lead (input fields, score, segment); return their ids."""
    created = now()
    ids = [
        conn.execute(
            "INSERT INTO predictions (created_at, model_version, lead, score, segment) "
            "VALUES (?, ?, ?, ?, ?)",
            (created, version, json.dumps(row), score, segment),
        ).lastrowid
        for row, score, segment in zip(rows, scores, segments, strict=True)
    ]
    conn.commit()
    return ids


def record_outcomes(conn: sqlite3.Connection, outcomes: dict[int, bool]) -> list[int]:
    """Save {prediction_id: converted}. Returns unknown ids; if any, nothing is saved.

    Sending an outcome again for the same prediction replaces the earlier one.
    """
    ids = list(outcomes)
    marks = ",".join("?" * len(ids))
    known = {
        row[0] for row in conn.execute(f"SELECT id FROM predictions WHERE id IN ({marks})", ids)
    }
    unknown = [i for i in ids if i not in known]
    if unknown:
        return unknown
    recorded = now()
    conn.executemany(
        "INSERT INTO outcomes (prediction_id, converted, recorded_at) VALUES (?, ?, ?) "
        "ON CONFLICT(prediction_id) DO UPDATE SET "
        "converted = excluded.converted, recorded_at = excluded.recorded_at",
        [(i, int(converted), recorded) for i, converted in outcomes.items()],
    )
    conn.commit()
    return []


def read_outcome_leads(conn: sqlite3.Connection) -> list[tuple[dict, int]]:
    """(lead fields, converted) for every lead with a known outcome (for retraining checks)."""
    rows = conn.execute(
        "SELECT p.lead, o.converted FROM outcomes o JOIN predictions p ON p.id = o.prediction_id"
    ).fetchall()
    return [(json.loads(lead), converted) for lead, converted in rows]


def read_outcomes(conn: sqlite3.Connection, since: str) -> list[tuple[float, str, int]]:
    """(score, segment, converted) for outcomes recorded at or after `since`."""
    return conn.execute(
        "SELECT p.score, p.segment, o.converted FROM outcomes o "
        "JOIN predictions p ON p.id = o.prediction_id WHERE o.recorded_at >= ?",
        (since,),
    ).fetchall()


def log_rejection(conn: sqlite3.Connection, path: str, errors: list[dict]) -> None:
    """Save a request the API refused as invalid."""
    conn.execute(
        "INSERT INTO rejected_requests (created_at, path, errors) VALUES (?, ?, ?)",
        (now(), path, json.dumps(errors, default=str)),
    )
    conn.commit()


def read_predictions(conn: sqlite3.Connection, since: str) -> list[tuple[dict, float, str]]:
    """(lead, score, segment) for every prediction logged at or after `since`."""
    rows = conn.execute(
        "SELECT lead, score, segment FROM predictions WHERE created_at >= ?", (since,)
    ).fetchall()
    return [(json.loads(lead), score, segment) for lead, score, segment in rows]


def count_rejections(conn: sqlite3.Connection, since: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM rejected_requests WHERE created_at >= ?", (since,)
    ).fetchone()[0]


def save_drift_run(conn: sqlite3.Connection, report: dict[str, Any]) -> int:
    """Store a monitoring report; returns its id."""
    cursor = conn.execute(
        "INSERT INTO drift_runs (created_at, n_samples, status, score_psi, report) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            report["created_at"],
            report["n_samples"],
            report["status"],
            report.get("prediction", {}).get("score_psi"),
            json.dumps(report),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def drift_runs(conn: sqlite3.Connection, limit: int = 30) -> list[dict[str, Any]]:
    """Latest monitoring reports, newest first."""
    rows = conn.execute(
        "SELECT id, report FROM drift_runs ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [{"id": run_id, **json.loads(report)} for run_id, report in rows]
