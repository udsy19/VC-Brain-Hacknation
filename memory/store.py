"""Append-only event store. Owner: A.

Note the signature: as_of is REQUIRED and has no default. That is deliberate —
it makes the lookahead bug hard to write rather than merely discouraged.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from schema.events import Event


def append(event: Event) -> UUID:
    raise NotImplementedError("A: H1-3")


def events(
    *,
    as_of: datetime,
    entity_id: UUID | None = None,
    company_id: UUID | None = None,
    kind: str | None = None,
) -> list[Event]:
    """Returns only events with observed_at <= as_of. No exceptions, no flags."""
    raise NotImplementedError("A: H1-3")
