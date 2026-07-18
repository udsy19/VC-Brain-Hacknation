"""The insurance policy (A.md H3-8). Insert events at t1<t2<t3 and prove that a
read — or a score — pinned at t2 can never see the t3 event. No-lookahead is the
one invariant the whole time-machine backtest rests on."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from backtest.runner import assert_no_lookahead
from memory import score, store
from schema.events import Event, EventKind, Source

T1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
T2 = T1 + timedelta(days=30)
T3 = T1 + timedelta(days=60)


def _green(entity_id, observed_at, value) -> Event:
    return Event(
        entity_id=entity_id,
        kind=EventKind.GREEN_FLAG,
        source=Source.MANUAL,
        observed_at=observed_at,
        payload={"value": value, "self_consistency": 0.9},
    )


def _seed_three():
    entity = store.get_store().create_entity("T", "t")
    e1 = store.append(_green(entity.entity_id, T1, 0.6))
    e2 = store.append(_green(entity.entity_id, T2, 0.7))
    e3 = store.append(_green(entity.entity_id, T3, 0.8))
    return entity.entity_id, e1, e2, e3


def test_events_at_t2_never_return_the_t3_event() -> None:
    eid, e1, e2, e3 = _seed_three()
    got = store.events(entity_id=eid, as_of=T2)
    ids = {e.event_id for e in got}
    assert e1 in ids and e2 in ids
    assert e3 not in ids
    assert all(e.observed_at <= T2 for e in got)
    assert_no_lookahead(got, as_of=T2)  # must not raise


def test_founder_score_at_t2_excludes_the_future_event() -> None:
    eid, e1, e2, e3 = _seed_three()
    fs = score.founder(eid, as_of=T2)
    assert e1 in fs.contributing_event_ids
    assert e2 in fs.contributing_event_ids
    assert e3 not in fs.contributing_event_ids


def test_later_as_of_admits_the_later_event() -> None:
    eid, _e1, _e2, e3 = _seed_three()
    fs = score.founder(eid, as_of=T3)
    assert e3 in fs.contributing_event_ids
