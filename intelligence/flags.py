"""Green-flag rules: the SENSOR feeding A's filter. Owner: C. See C.md H1-3.

30-50 interpretable YES/NO rules, trajectory-tuned. Each returns a GREEN_FLAG event
carrying its evidence span, so every score decomposes to rules_fired + source spans.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from schema.events import Event


def evaluate(entity_id: UUID, as_of: datetime) -> list[Event]:
    raise NotImplementedError("C: H1-3")


def observation(flag_events: list[Event]) -> tuple[float, float]:
    """(y_t, r_t) for A's filter. Agree this payload with A in person, not over chat."""
    raise NotImplementedError("C: H3-8")
