"""Beta-Binomial with forgetting factor lambda. Owner: A.

Wired behind SCORE_MODEL=beta_binomial (score.founder dispatches to here). Verify
the flag works at H10, not H20: if the Kalman misbehaves in the demo, flip one env
var and every consumer keeps working because the output is the same FounderScore.

The model: each observation is a weighted success/failure on a Beta posterior; a
per-day forgetting factor lambda pulls the posterior back toward the uniform prior
over time, so old evidence fades and recent evidence dominates (that's the
'momentum' this simpler model can express). Reliable, low-noise observations
(proof events) weigh more; contradicted ones are already filtered upstream by
build_observations.
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from uuid import UUID

from schema.events import FounderScore


def _lambda() -> float:
    return float(os.getenv("SCORE_LAMBDA", "0.985"))  # per-day retention


def founder(entity_id: UUID, as_of: datetime) -> FounderScore:
    from memory.score import build_observations  # shared observation boundary

    obs = build_observations(entity_id, as_of)
    lam = _lambda()

    a, b = 1.0, 1.0  # uniform prior Beta(1, 1)
    contributing: list[UUID] = []
    last_t: datetime | None = None
    mu_before_last = 0.5
    mu_after_last = 0.5

    for o in obs:
        if last_t is not None:
            decay = lam ** ((o.observed_at - last_t).total_seconds() / 86400.0)
            a = 1.0 + (a - 1.0) * decay
            b = 1.0 + (b - 1.0) * decay

        weight = o.self_consistency / max(o.source_penalty, 1e-6)
        mu_before_last = a / (a + b)
        a += weight * o.value
        b += weight * (1.0 - o.value)
        mu_after_last = a / (a + b)

        contributing.extend(o.event_ids)
        last_t = o.observed_at

    # Forget forward to as_of: with no fresh evidence the posterior relaxes toward
    # the prior and the band widens — the honest cost of staleness.
    if last_t is not None:
        dt = (as_of - last_t).total_seconds() / 86400.0
        if dt > 0:
            decay = lam**dt
            a = 1.0 + (a - 1.0) * decay
            b = 1.0 + (b - 1.0) * decay

    mu = a / (a + b)
    variance = (a * b) / ((a + b) ** 2 * (a + b + 1.0))
    trend = mu_after_last - mu_before_last  # direction of the most recent evidence

    return FounderScore(
        entity_id=entity_id,
        as_of=as_of,
        mu=float(mu),
        band=float(math.sqrt(max(variance, 0.0))),
        trend=float(trend),
        contributing_event_ids=contributing,
        model="beta_binomial",
    )
