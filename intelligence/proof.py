"""PROOF PROTOCOL — the centerpiece. Owner: C. See C.md H8-12. Protect this block.

generate(): a founder-specific micro-challenge from the deck's central technical claim,
containing one ambiguous requirement (do they ask?) and one planted bad constraint
(do they push back?). The planted constraint is the sharpest signal in the system.

grade(): artifact quality + BEHAVIORAL trace (iteration count, time-to-first-commit,
latency profile, whether they challenged the bad constraint). Behavior is harder to fake.

Results become low-noise observations for A's filter -> the score visibly moves -> the
founder re-enters the gate. That re-entry is the demo.
"""

from __future__ import annotations

from uuid import UUID

from schema.events import Challenge, Event


def generate(company_id: UUID) -> Challenge:
    raise NotImplementedError("C: H8-12")


def grade(challenge_id: UUID, artifact: str, trace: dict) -> list[Event]:
    raise NotImplementedError("C: H8-12")
