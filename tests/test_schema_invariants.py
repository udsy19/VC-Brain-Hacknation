"""Invariants that hold from H0, before any implementation exists.

These pass today. The as_of and injection tests (test_memory_asof.py,
test_bus_injection.py) are A's and B's to write as they build.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from backtest.runner import LookaheadError, assert_no_lookahead
from schema.events import Event, EventKind, Source, utcnow

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _event(observed_at: datetime) -> Event:
    return Event(
        kind=EventKind.REPO_ACTIVITY,
        source=Source.GITHUB,
        observed_at=observed_at,
        entity_id=uuid4(),
    )


def test_naive_datetime_is_rejected() -> None:
    """Naive datetimes silently break as_of comparisons — reject at the boundary."""
    with pytest.raises(ValueError, match="timezone-aware"):
        Event(
            kind=EventKind.REPO_ACTIVITY,
            source=Source.GITHUB,
            observed_at=datetime(2024, 1, 1),  # noqa: DTZ001 — the point of the test
        )


def test_lookahead_detector_catches_future_events() -> None:
    future = [_event(T0), _event(T0 + timedelta(days=30))]
    with pytest.raises(LookaheadError):
        assert_no_lookahead(future, as_of=T0 + timedelta(days=1))


def test_lookahead_detector_passes_clean_history() -> None:
    past = [_event(T0), _event(T0 + timedelta(days=1))]
    assert_no_lookahead(past, as_of=T0 + timedelta(days=2))


def test_ingested_at_defaults_but_observed_at_never_does() -> None:
    """observed_at is required. Defaulting it to now() is how the backtest gets poisoned."""
    e = _event(T0)
    assert e.observed_at == T0
    assert e.ingested_at > T0
    with pytest.raises(ValueError):
        Event(kind=EventKind.HN_POST, source=Source.HN)  # type: ignore[call-arg]


def test_utcnow_is_aware() -> None:
    assert utcnow().tzinfo is not None
