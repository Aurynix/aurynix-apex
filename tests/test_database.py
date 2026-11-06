import sqlite3

import pytest

from apex.api import database
from apex.api.database import connect, database_url, normalize_url, read_outcome_leads
from tests.conftest import as_rows, synthetic_leads


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("postgres://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
        (
            "postgresql://u:p@host/db?sslmode=require",
            "postgresql+psycopg://u:p@host/db?sslmode=require",
        ),
        ("postgresql+psycopg://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
        ("sqlite:///local.db", "sqlite:///local.db"),
    ],
)
def test_normalize_url(given, expected):
    assert normalize_url(given) == expected


@pytest.mark.parametrize("value", [None, "", "   "])
def test_unset_or_empty_database_url_falls_back_to_the_local_sqlite_file(
    value, tmp_path, monkeypatch
):
    if value is None:
        monkeypatch.delenv("DATABASE_URL", raising=False)
    else:
        monkeypatch.setenv("DATABASE_URL", value)
    file = tmp_path / "nested" / "apex.db"
    monkeypatch.setattr(database, "path", lambda key: {"database": file}[key])

    assert database_url() is None
    db = connect()
    assert db.backend == "sqlite"
    assert file.exists()
    db.close()


def test_database_url_is_used_when_set(tmp_path, monkeypatch):
    file = tmp_path / "from_url.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{file}")
    monkeypatch.setattr(database, "path", lambda key: pytest.fail("must not use the local path"))
    db = connect()
    db.scalar("SELECT COUNT(*) FROM predictions")
    assert file.exists()
    db.close()


def test_postgres_url_selects_the_psycopg_driver_and_keeps_no_idle_connections():
    db = database.Database(normalize_url("postgres://u:p@host/db"))
    assert db.backend == "postgresql"
    assert db.engine.driver == "psycopg"
    assert db.engine.pool.__class__.__name__ == "NullPool"


def test_a_path_is_always_local_even_when_database_url_is_set(tmp_path, monkeypatch):
    """Simulations pass a Path, and must never write to the real database."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@unreachable.invalid/db")
    db = connect(tmp_path / "simulation.db")
    assert db.backend == "sqlite"
    db.close()


def test_connect_adds_source_to_a_database_from_an_older_version(tmp_path):
    file = tmp_path / "old.db"
    old = sqlite3.connect(file)
    old.execute(
        "CREATE TABLE predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,"
        " model_version TEXT NOT NULL, lead TEXT NOT NULL, score REAL NOT NULL,"
        " segment TEXT NOT NULL)"
    )
    old.execute(
        "INSERT INTO predictions (created_at, model_version, lead, score, segment) "
        "VALUES ('2026-10-06', 'apex-v0.1.0', '{}', 0.5, 'high')"
    )
    old.commit()
    old.close()

    db = connect(file)
    assert db.rows("SELECT id, source FROM predictions") == [(1, "api")]
    db.close()
    db = connect(file)  # opening it again changes nothing
    assert db.scalar("SELECT COUNT(*) FROM predictions") == 1
    db.close()


def test_api_predictions_are_marked_as_api(service):
    service.score(as_rows(synthetic_leads(1)))
    assert service.db.rows("SELECT source FROM predictions") == [("api",)]


def test_read_outcome_leads_returns_real_outcomes(service):
    row = as_rows(synthetic_leads(1))[0]
    prediction_id = service.score([row])[0]["prediction_id"]
    service.record_outcomes({prediction_id: True})
    assert read_outcome_leads(service.db) == [(row, 1)]
