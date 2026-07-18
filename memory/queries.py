"""as_of-scoped read helpers for C and D. Owner: A.

Every helper here funnels through store.events(as_of=...), so no-lookahead is
inherited, not re-implemented. If a helper ever reads self._events directly,
it's a bug.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from memory import store
from schema.events import Event, EventKind


def timeline(entity_id: UUID, as_of: datetime) -> list[Event]:
    """Every event about this person up to as_of, oldest first. D's trace walks it."""
    return store.get_store().events(entity_id=entity_id, as_of=as_of)


def company_timeline(company_id: UUID, as_of: datetime) -> list[Event]:
    return store.get_store().events(company_id=company_id, as_of=as_of)


def claims(company_id: UUID, as_of: datetime) -> list[Event]:
    """DECK_CLAIM events awaiting validation — what C's validator checks."""
    return store.get_store().events(company_id=company_id, as_of=as_of, kind=EventKind.DECK_CLAIM)


def latest_facts(entity_id: UUID, as_of: datetime) -> dict[str, Event]:
    """Most recent PROFILE_FACT per fact key, as of T. Later events supersede
    earlier ones by being *newer*, never by overwriting them — the old fact is
    still in the log, and a read at an earlier as_of still sees it."""
    facts: dict[str, Event] = {}
    for e in store.get_store().events(
        entity_id=entity_id, as_of=as_of, kind=EventKind.PROFILE_FACT
    ):
        key = str(e.payload.get("key", e.event_id))
        current = facts.get(key)
        if current is None or e.observed_at >= current.observed_at:
            facts[key] = e
    return facts


def contradicted_event_ids(entity_id: UUID, as_of: datetime) -> set[UUID]:
    """Event IDs a CONTRADICTION / contradicted VALIDATION_RESULT points at, as of
    T. The scorer uses this to keep contradicted claims out of the filter."""
    s = store.get_store()
    contradicted: set[UUID] = set()
    for e in s.events(entity_id=entity_id, as_of=as_of, kind=EventKind.CONTRADICTION):
        contradicted.update(_targets(e))
    for e in s.events(entity_id=entity_id, as_of=as_of, kind=EventKind.VALIDATION_RESULT):
        if str(e.payload.get("status", "")).lower() == "contradicted":
            contradicted.update(_targets(e))
    return contradicted


def _targets(event: Event) -> list[UUID]:
    """Event IDs a contradiction references. Accepts a single target_event_id or a
    list under contradicts / target_event_ids, so C has room to shape the payload."""
    raw: list = []
    for key in ("target_event_id", "target_event_ids", "contradicts"):
        v = event.payload.get(key)
        if isinstance(v, list):
            raw.extend(v)
        elif v is not None:
            raw.append(v)
    out: list[UUID] = []
    for item in raw:
        try:
            out.append(item if isinstance(item, UUID) else UUID(str(item)))
        except (ValueError, AttributeError):
            continue
    return out
