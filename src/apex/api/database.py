"""Storage: Postgres when `DATABASE_URL` is set, otherwise local SQLite
(`config.json → paths.database`).

Tables:
- `predictions`: every scored lead (input fields, score, segment, source)
- `rejected_requests`: requests the API refused as invalid (422), for data quality
- `drift_runs`: one row per monitoring run, with the full report as JSON
- `outcomes`: the real result of a scored lead (converted or not), sent later

Tables are created on startup if they do not exist.
"""

import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    Column,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    Table,
    Text,
    create_engine,
    insert,
    inspect,
    select,
    text,
)
from sqlalchemy.engine import URL, Connection, make_url
from sqlalchemy.pool import NullPool, StaticPool

from apex.config import path

API_SOURCE = "api"
DEMO_SOURCE = "demo_seed"
_SETUP_LOCK_ID = 7203391  # Postgres advisory lock: one process at a time creates or seeds tables

metadata = MetaData()

predictions = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", Text, nullable=False),
    Column("model_version", Text, nullable=False),
    Column("lead", Text, nullable=False),
    Column("score", Float, nullable=False),
    Column("segment", Text, nullable=False),
    Column("source", Text, nullable=False, server_default=API_SOURCE),
    sqlite_autoincrement=True,
)
rejected_requests = Table(
    "rejected_requests",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", Text, nullable=False),
    Column("path", Text, nullable=False),
    Column("errors", Text, nullable=False),
    sqlite_autoincrement=True,
)
outcomes_table = Table(
    "outcomes",
    metadata,
    Column("prediction_id", Integer, ForeignKey("predictions.id"), primary_key=True),
    Column("converted", Integer, CheckConstraint("converted IN (0, 1)"), nullable=False),
    Column("recorded_at", Text, nullable=False),
)
drift_runs_table = Table(
    "drift_runs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", Text, nullable=False),
    Column("n_samples", Integer, nullable=False),
    Column("status", Text, nullable=False),
    Column("score_psi", Float),
    Column("report", Text, nullable=False),
    sqlite_autoincrement=True,
)


class Database:
    """One database, opened for SQLite or Postgres.

    SQLite: API threads share one connection, so they take turns through a lock.
    Postgres: a new connection per use and none kept open, so a serverless database
    (Neon) can scale to zero when there is no traffic.
    """

    def __init__(self, url: str | URL):
        self.url = make_url(url)
        self.backend = self.url.get_backend_name()
        if self.backend == "sqlite":
            self.engine = create_engine(
                self.url, poolclass=StaticPool, connect_args={"check_same_thread": False}
            )
            self._lock = threading.RLock()
        else:
            self.engine = create_engine(
                self.url,
                poolclass=NullPool,
                # prepare_threshold=None: works through connection poolers such as Neon's
                connect_args={"connect_timeout": 15, "prepare_threshold": None},
            )
            self._lock = nullcontext()

    @contextmanager
    def transaction(self) -> Iterator[Connection]:
        """A connection in one transaction: committed at the end, rolled back on an error."""
        with self._lock, self.engine.begin() as conn:
            yield conn

    def rows(self, sql: str, **params: Any) -> list[tuple]:
        with self.transaction() as conn:
            return [tuple(row) for row in conn.execute(text(sql), params)]

    def scalar(self, sql: str, **params: Any) -> Any:
        with self.transaction() as conn:
            return conn.execute(text(sql), params).scalar()

    def ping(self) -> None:
        """Raise if the database cannot be reached."""
        self.scalar("SELECT 1")

    def close(self) -> None:
        self.engine.dispose()


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def database_url() -> str | None:
    """The `DATABASE_URL` environment variable; None when it is unset or empty."""
    return os.environ.get("DATABASE_URL", "").strip() or None


def normalize_url(url: str) -> str:
    """Use the psycopg (v3) driver for plain Postgres URLs, as hosting providers give them."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url.removeprefix(prefix)
    return url


def _take_setup_lock(conn: Connection) -> None:
    """On Postgres, wait until no other process is creating or seeding tables."""
    if conn.dialect.name == "postgresql":
        conn.execute(text("SELECT pg_advisory_xact_lock(:id)"), {"id": _SETUP_LOCK_ID})


def init_db(db: Database) -> Database:
    """Create the tables if they do not exist yet, and add `source` to older databases."""
    with db.transaction() as conn:
        _take_setup_lock(conn)
        metadata.create_all(conn)
        columns = {column["name"] for column in inspect(conn).get_columns("predictions")}
        if "source" not in columns:
            conn.execute(
                text(
                    "ALTER TABLE predictions "
                    f"ADD COLUMN source TEXT NOT NULL DEFAULT '{API_SOURCE}'"
                )
            )
    return db


def connect(target: Path | str | None = None) -> Database:
    """Open a database with all tables.

    No target: `DATABASE_URL` if set, otherwise the local SQLite file. A Path is always
    a SQLite file (so simulations never touch the real log); a string is a database URL.
    """
    if target is None:
        target = database_url() or path("database")
    if isinstance(target, Path):
        target.parent.mkdir(parents=True, exist_ok=True)
        return init_db(Database(URL.create("sqlite", database=str(target))))
    return init_db(Database(normalize_url(target)))


def log_predictions(
    db: Database,
    rows: list[dict],
    scores: list[float],
    segments: list[str],
    version: str,
) -> list[int]:
    """Save one row per scored lead (input fields, score, segment); return their ids."""
    created = now()
    values = [
        {
            "created_at": created,
            "model_version": version,
            "lead": json.dumps(row),
            "score": score,
            "segment": segment,
            "source": API_SOURCE,
        }
        for row, score, segment in zip(rows, scores, segments, strict=True)
    ]
    with db.transaction() as conn:
        return _insert_predictions(conn, values)


def _insert_predictions(conn: Connection, values: list[dict]) -> list[int]:
    """Insert in one batch and return the ids in the same order as `values`."""
    if not values:
        return []
    statement = insert(predictions).returning(predictions.c.id, sort_by_parameter_order=True)
    return list(conn.execute(statement, values).scalars())


def insert_demo_rows(
    db: Database,
    leads: list[dict],
    scores: list[float],
    segments: list[str],
    converted: list[int],
    created_at: list[str],
    recorded_at: list[str],
    version: str,
) -> int:
    """Save demo predictions with their outcomes, only if `predictions` is empty.

    All or nothing, in one transaction. Returns the number of rows saved (0 if not empty).
    """
    with db.transaction() as conn:
        _take_setup_lock(conn)
        if conn.execute(select(predictions.c.id).limit(1)).first() is not None:
            return 0
        ids = _insert_predictions(
            conn,
            [
                {
                    "created_at": created,
                    "model_version": version,
                    "lead": json.dumps(lead),
                    "score": score,
                    "segment": segment,
                    "source": DEMO_SOURCE,
                }
                for lead, score, segment, created in zip(
                    leads, scores, segments, created_at, strict=True
                )
            ],
        )
        conn.execute(
            insert(outcomes_table),
            [
                {"prediction_id": prediction_id, "converted": result, "recorded_at": recorded}
                for prediction_id, result, recorded in zip(ids, converted, recorded_at, strict=True)
            ],
        )
    return len(ids)


def record_outcomes(db: Database, outcomes: dict[int, bool]) -> list[int]:
    """Save {prediction_id: converted}. Returns unknown ids; if any, nothing is saved.

    Sending an outcome again for the same prediction replaces the earlier one.
    """
    ids = list(outcomes)
    with db.transaction() as conn:
        statement = select(predictions.c.id).where(predictions.c.id.in_(ids))
        known = set(conn.execute(statement).scalars())
        unknown = [i for i in ids if i not in known]
        if unknown:
            return unknown
        recorded = now()
        conn.execute(
            text(
                "INSERT INTO outcomes (prediction_id, converted, recorded_at) "
                "VALUES (:prediction_id, :converted, :recorded_at) "
                "ON CONFLICT (prediction_id) DO UPDATE SET "
                "converted = excluded.converted, recorded_at = excluded.recorded_at"
            ),
            [
                {"prediction_id": i, "converted": int(converted), "recorded_at": recorded}
                for i, converted in outcomes.items()
            ],
        )
    return []


def read_outcome_leads(db: Database) -> list[tuple[dict, int]]:
    """(lead fields, converted) for every lead with a real outcome (for retraining checks).

    Demo rows are left out: they must never decide whether a model is replaced.
    """
    rows = db.rows(
        "SELECT p.lead, o.converted FROM outcomes o JOIN predictions p ON p.id = o.prediction_id "
        "WHERE p.source <> :demo",
        demo=DEMO_SOURCE,
    )
    return [(json.loads(lead), converted) for lead, converted in rows]


def read_outcomes(db: Database, since: str) -> list[tuple[float, str, int]]:
    """(score, segment, converted) for outcomes recorded at or after `since`."""
    return db.rows(
        "SELECT p.score, p.segment, o.converted FROM outcomes o "
        "JOIN predictions p ON p.id = o.prediction_id WHERE o.recorded_at >= :since",
        since=since,
    )


def log_rejection(db: Database, path: str, errors: list[dict]) -> None:
    """Save a request the API refused as invalid."""
    with db.transaction() as conn:
        conn.execute(
            insert(rejected_requests),
            {"created_at": now(), "path": path, "errors": json.dumps(errors, default=str)},
        )


def read_predictions(db: Database, since: str) -> list[tuple[dict, float, str]]:
    """(lead, score, segment) for every prediction logged at or after `since`."""
    rows = db.rows(
        "SELECT lead, score, segment FROM predictions WHERE created_at >= :since", since=since
    )
    return [(json.loads(lead), score, segment) for lead, score, segment in rows]


def count_rejections(db: Database, since: str) -> int:
    return db.scalar(
        "SELECT COUNT(*) FROM rejected_requests WHERE created_at >= :since", since=since
    )


def save_drift_run(db: Database, report: dict[str, Any]) -> int:
    """Store a monitoring report; returns its id."""
    statement = insert(drift_runs_table).returning(drift_runs_table.c.id)
    with db.transaction() as conn:
        return conn.execute(
            statement,
            {
                "created_at": report["created_at"],
                "n_samples": report["n_samples"],
                "status": report["status"],
                "score_psi": report.get("prediction", {}).get("score_psi"),
                "report": json.dumps(report),
            },
        ).scalar_one()


def drift_runs(db: Database, limit: int = 30) -> list[dict[str, Any]]:
    """Latest monitoring reports, newest first."""
    rows = db.rows("SELECT id, report FROM drift_runs ORDER BY id DESC LIMIT :limit", limit=limit)
    return [{"id": run_id, **json.loads(report)} for run_id, report in rows]
