"""Append-only event store + entity registry. Owner: A.

Note the signature: as_of is REQUIRED and has no default. That is deliberate —
it makes the lookahead bug hard to write rather than merely discouraged.

Backend: an in-process append-only store. It mirrors the five tables in
schema/migrations/001_init.sql (events, entities, entity_aliases, companies,
merges) exactly, so the same code runs whether the demo is wired to Supabase or
not. In-memory is the default because a live demo that depends on a hosted DB
being healthy is a demo that fails (see D.md). The committed migration is what
gives us Postgres parity when we want persistence; the runtime store here is
what keeps the backtest and the demo hermetic and fast.

The append-only property is enforced structurally: this store exposes no update
or delete for events. Corrections are new events. (The SQL migration enforces
the same thing with a trigger, for the hosted path.)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from schema.events import Company, Entity, Event, utcnow


@dataclass(frozen=True)
class Alias:
    """One resolvable identifier pointing at an entity. Unique on (kind, value),
    exactly like entity_aliases in the SQL schema."""

    entity_id: UUID
    kind: str  # 'email' | 'url' | 'handle:github' | 'twitter' | 'context' | 'name'
    value: str  # caller-normalized; the store matches exactly, it never normalizes
    source: str


@dataclass(frozen=True)
class Merge:
    """An entity-resolution decision, including the AMBIGUOUS ones we refuse to
    guess on. Mirrors the merges table."""

    entity_a: UUID
    entity_b: UUID
    status: str  # 'merged' | 'ambiguous' | 'rejected'
    score: float
    rationale: str
    decided_at: datetime = field(default_factory=utcnow)


class EventStore:
    """The spine. Append-only event log + the entity/company/alias/merge registry
    every other module reads through."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._events: list[Event] = []
        self._entities: dict[UUID, Entity] = {}
        self._companies: dict[UUID, Company] = {}
        self._aliases: dict[tuple[str, str], Alias] = {}  # (kind, value) -> Alias
        self._merges: list[Merge] = []

    # -- events -------------------------------------------------------------

    def append(self, event: Event) -> UUID:
        """The only way anything enters history. Append-only: never overwrites."""
        with self._lock:
            self._events.append(event)
        return event.event_id

    def events(
        self,
        *,
        as_of: datetime,
        entity_id: UUID | None = None,
        company_id: UUID | None = None,
        kind: str | None = None,
    ) -> list[Event]:
        """Returns only events with observed_at <= as_of. No exceptions, no flags.

        This is the single chokepoint the no-lookahead invariant is enforced at:
        every downstream read (queries, scoring, backtest) goes through here, so a
        future event physically cannot reach a scorer scoped to an earlier as_of.
        """
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware — a naive as_of silently mis-filters")
        with self._lock:
            out = [
                e
                for e in self._events
                if e.observed_at <= as_of
                and (entity_id is None or e.entity_id == entity_id)
                and (company_id is None or e.company_id == company_id)
                and (kind is None or e.kind == kind)
            ]
        # Deterministic order: by world-time, then ingest-time, then id (stable).
        out.sort(key=lambda e: (e.observed_at, e.ingested_at, str(e.event_id)))
        return out

    # -- entities -----------------------------------------------------------

    def create_entity(self, display_name: str, name_normalized: str) -> Entity:
        entity = Entity(display_name=display_name, name_normalized=name_normalized)
        with self._lock:
            self._entities[entity.entity_id] = entity
        return entity

    def get_entity(self, entity_id: UUID) -> Entity | None:
        return self._entities.get(entity_id)

    def entities(self) -> list[Entity]:
        with self._lock:
            return list(self._entities.values())

    # -- aliases ------------------------------------------------------------

    def add_alias(self, entity_id: UUID, kind: str, value: str, source: str) -> UUID:
        """Bind an identifier to an entity. First-writer-wins: if (kind, value)
        already points elsewhere we DON'T reassign it (silently stealing an
        identifier is how you merge two people by accident) — we return the
        current owner so the caller can decide (usually: AMBIGUOUS)."""
        key = (kind, value)
        with self._lock:
            existing = self._aliases.get(key)
            if existing is not None:
                return existing.entity_id
            self._aliases[key] = Alias(entity_id=entity_id, kind=kind, value=value, source=source)
        return entity_id

    def find_by_alias(self, kind: str, value: str) -> UUID | None:
        alias = self._aliases.get((kind, value))
        return alias.entity_id if alias else None

    def aliases_for(self, entity_id: UUID) -> list[Alias]:
        with self._lock:
            return [a for a in self._aliases.values() if a.entity_id == entity_id]

    def aliases_by_kind(self, kind: str) -> list[Alias]:
        with self._lock:
            return [a for a in self._aliases.values() if a.kind == kind]

    # -- companies ----------------------------------------------------------

    def create_company(
        self,
        name: str,
        *,
        founder_entity_ids: list[UUID] | None = None,
        archetype: int | None = None,
    ) -> Company:
        company = Company(
            name=name,
            founder_entity_ids=list(founder_entity_ids or []),
            archetype=archetype,
        )
        with self._lock:
            self._companies[company.company_id] = company
        return company

    def get_company(self, company_id: UUID) -> Company | None:
        return self._companies.get(company_id)

    def companies(self) -> list[Company]:
        with self._lock:
            return list(self._companies.values())

    # -- merges -------------------------------------------------------------

    def record_merge(
        self, entity_a: UUID, entity_b: UUID, status: str, score: float, rationale: str
    ) -> Merge:
        merge = Merge(
            entity_a=entity_a, entity_b=entity_b, status=status, score=score, rationale=rationale
        )
        with self._lock:
            self._merges.append(merge)
        return merge

    def merges(self, *, status: str | None = None) -> list[Merge]:
        with self._lock:
            return [m for m in self._merges if status is None or m.status == status]

    # -- test / demo support ------------------------------------------------

    def reset(self) -> None:
        """Wipe all state. For test isolation and demo reseeding only — there is
        no per-record delete, because the log is append-only."""
        with self._lock:
            self._events.clear()
            self._entities.clear()
            self._companies.clear()
            self._aliases.clear()
            self._merges.clear()


# ---------------------------------------------------------------------------
# Module-level default store. SHARED §4 imports `store.append` / `store.events`
# directly; the richer entity API is reached via get_store().
# ---------------------------------------------------------------------------

_default = EventStore()


def get_store() -> EventStore:
    return _default


def append(event: Event) -> UUID:
    return _default.append(event)


def events(
    *,
    as_of: datetime,
    entity_id: UUID | None = None,
    company_id: UUID | None = None,
    kind: str | None = None,
) -> list[Event]:
    """Returns only events with observed_at <= as_of. No exceptions, no flags."""
    return _default.events(as_of=as_of, entity_id=entity_id, company_id=company_id, kind=kind)


def reset() -> None:
    _default.reset()
