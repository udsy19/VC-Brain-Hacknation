"""Three axes, never averaged. Owner: C. See C.md H3-8."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from schema.events import ScreeningResult


def three_axis(company_id: UUID, as_of: datetime) -> ScreeningResult:
    raise NotImplementedError("C: H3-8")
