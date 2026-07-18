"""Shared test fixtures. The store is a module-global singleton, so it must be
reset between tests or state leaks across them. Scoring env vars are cleared too,
so a test that flips SCORE_MODEL can't poison the next one."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from memory import store

_SCORING_ENV = ("SCORE_MODEL", "SCORE_Q", "SCORE_R0", "SCORE_LAMBDA")


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    for var in _SCORING_ENV:
        monkeypatch.delenv(var, raising=False)
    store.reset()
    yield
    store.reset()
