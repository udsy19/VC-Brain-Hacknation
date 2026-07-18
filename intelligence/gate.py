"""Decision gate. Owner: C. See C.md H8-12.

The absence classifier is the delicate part: signal-absent-because-irrelevant
(a designer with no GitHub) vs signal-absent-and-suspicious (an infra founder
claiming a distributed system with no code anywhere). Get this wrong and we punish
exactly the founders this thesis exists to find.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from schema.events import GateDecision


def evaluate(company_id: UUID, as_of: datetime) -> GateDecision:
    raise NotImplementedError("C: H8-12")
