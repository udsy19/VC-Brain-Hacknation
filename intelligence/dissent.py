"""Dissent Engine. Owner: C. See C.md H12-16.

Same evidence graph, inverted objective. Prompt it ADVERSARIALLY — a polite balanced
take makes the whole feature read as theater. It must name the single load-bearing
claim that kills the thesis if false.

The recommendation stays null until dissent is opened, enforced in the API response
shape rather than the frontend, so it cannot be bypassed live on stage.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from schema.events import AntiMemo


def generate(company_id: UUID, as_of: datetime) -> AntiMemo:
    raise NotImplementedError("C: H12-16")
