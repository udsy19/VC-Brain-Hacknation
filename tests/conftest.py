"""Shared offline fixtures for both the A Memory contract and C's DB helpers."""

from __future__ import annotations

import os
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path

import pytest

from core.config import settings
from memory import db, store

_MANAGED_ENV = ("SCORE_MODEL", "SCORE_Q", "SCORE_R0", "SCORE_LAMBDA", "MEMORY_BACKEND")


@pytest.fixture(autouse=True, scope="session")
def _offline_db():
    from memory import db

    previous = os.environ.get("VCBRAIN_DB_PATH")
    db_path = Path(".pytest-vcbrain-session.db").resolve()
    os.environ["VCBRAIN_DB_PATH"] = str(db_path)
    yield
    db.reset_connections()
    db_path.unlink(missing_ok=True)
    if previous is None:
        os.environ.pop("VCBRAIN_DB_PATH", None)
    else:
        os.environ["VCBRAIN_DB_PATH"] = previous


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    for var in _MANAGED_ENV:
        monkeypatch.delenv(var, raising=False)
    # store._backend() infers "postgres" from a configured DATABASE_URL. settings is
    # built at import time, so without this the suite's backend — and whether it dials
    # out to a real database — depends on whoever's .env is on disk.
    monkeypatch.setattr("core.config.settings", replace(settings, database_url=""))

    # And the same for the ENVIRONMENT, not just the settings object. memory/db.py reads
    # os.getenv("DATABASE_URL") directly — core.config calls load_dotenv() at import, so
    # a developer's .env put the real hosted Postgres in front of every test that touches
    # db.connect(), and VCBRAIN_DB_PATH was ignored entirely.
    #
    # That is not merely slow, it is a correctness hazard both ways: the suite reads
    # whatever state the live system happens to be in, and anything it WRITES lands in
    # the shared database. It was found the hard way — a thesis saved by a read-only
    # deployment test moved `risk_appetite`, which moves the evidence bar, which failed
    # four gate-threshold tests that had nothing to do with the change.
    monkeypatch.delenv("DATABASE_URL", raising=False)
    db.reset_connections()

    store._pg = None
    store.reset()
    yield
    db.reset_connections()
    store._pg = None
    store.reset()
