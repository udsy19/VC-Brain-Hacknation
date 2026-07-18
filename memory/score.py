"""Founder Score: local-linear-trend Kalman filter. Owner: A. See A.md H8-12.

    x_t = [mu, nu]        mu = capability level, nu = momentum
    F   = [[1, dt], [0, 1]]
    Score = mu   Band = sqrt(P[0,0])   Trend = nu (structural, never a diff of scores)

Contradicted claims must never become observations — filter at the boundary here.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from schema.events import FounderScore


def founder(entity_id: UUID, as_of: datetime) -> FounderScore:
    raise NotImplementedError("A: H8-12")


def forecast(entity_id: UUID, as_of: datetime, k_days: int) -> tuple[float, float]:
    """k-step prediction interval — propagate P forward. Falls out of the filter."""
    raise NotImplementedError("A: H12-16")
