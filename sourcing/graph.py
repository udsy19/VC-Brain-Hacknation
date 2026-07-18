"""Collaboration graph + personalized PageRank. Owner: B. See B.md H8-12.

    hidden(v) = z(ppr(v)) - z(visibility(v))

High proximity to greatness, low individual visibility = the founder nobody emailed.
Edges are observed_at-stamped and MUST be as_of-filterable — the backtest replays this.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from schema.events import HiddenCandidate


def hidden_ranking(as_of: datetime, k: int = 50) -> list[HiddenCandidate]:
    raise NotImplementedError("B: H8-12")


def access_lift(picks: list[UUID]) -> float:
    """% of top-K with near-zero traditional visibility. The closing line of the pitch."""
    raise NotImplementedError("B: H12-16")
