"""Founder Score: local-linear-trend Kalman filter. Owner: A. See A.md H8-12.

    x_t = [mu, nu]        mu = capability level, nu = momentum
    F   = [[1, dt], [0, 1]]     dt in days since last observation, from observed_at
    Score = mu   Band = sqrt(P[0,0])   Trend = nu (structural, never a diff of scores)

What this is NOT: a probability of success, or a promise of a return. It's a
trajectory estimate — where the founder's capability is and which way it's moving,
with honest uncertainty.

The Founder Score belongs to the *entity*, not a company. It persists across
companies and ideas because it reads the entity's whole event history, and a new
company doesn't erase those events.

Contradicted claims must never become observations — filtered at the boundary here
(build_observations), and the exclusion is logged.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from uuid import UUID

import numpy as np

from core.config import settings
from memory import queries, store
from schema.events import EventKind, FounderScore, Observation, Source

log = logging.getLogger(__name__)

# Kinds that carry a capability observation. Raw sourcing events (repo activity,
# papers, posts) are NOT observations on their own — C's green-flag layer turns
# them into GREEN_FLAG events, and proof events are the low-noise ones.
_OBSERVATION_KINDS = {
    EventKind.GREEN_FLAG,
    EventKind.PROOF_ARTIFACT,
    EventKind.PROOF_BEHAVIOR,
}

# Source reliability -> measurement-noise multiplier. Proof Protocol evidence is
# fresh, verified and behavioral, so it gets low noise and moves the score hard —
# that's the demo moment. This weights how independently checkable the *source
# channel* is — never who the founder is or where they trained (Invariant #3).
_SOURCE_PENALTY = {
    Source.PROOF_PROTOCOL: 0.5,
    Source.VALIDATOR: 0.8,
    Source.GITHUB: 1.0,
    Source.ARXIV: 1.0,
    Source.MANUAL: 1.0,
    Source.HN: 1.2,
    Source.WEB: 1.4,
    Source.DECK: 1.6,
}

# Prior: neutral capability, zero momentum, wide band. band0 = sqrt(0.25) = 0.5,
# so a founder we know nothing about reads mu=0.5 band=0.5 trend=0.0 — the honest
# "we have no evidence yet" answer (and it matches the H1-3 stub).
_X0 = np.array([0.5, 0.0])
_P0 = np.array([[0.25, 0.0], [0.0, 0.25]])
_H = np.array([[1.0, 0.0]])


_DAYS_PER_YEAR = 365.25


def _params() -> tuple[float, float]:
    """(q, r0), read from env so calibration (H12-16) can grid-search them without
    a code change. q = process-noise intensity, r0 = base measurement noise."""
    return (
        float(os.getenv("SCORE_Q", "0.01")),
        float(os.getenv("SCORE_R0", "0.08")),
    )


def _dt_years(later: datetime, earlier: datetime) -> float:
    """Δt between observations, in YEARS. A.md writes the transition in days, but
    the covariance prediction has a Δt²·P_trend term that overflows any sane band
    over multi-month gaps if Δt is in days (365² ≈ 1.3e5). Scaling to years keeps
    the filter numerically stable and makes ν directly readable as
    'capability change per year'. The math is identical; only the unit changes."""
    return (later - earlier).total_seconds() / 86400.0 / _DAYS_PER_YEAR


def _active_model() -> str:
    return os.getenv("SCORE_MODEL", settings.score_model)


# ---------------------------------------------------------------------------
# Observation contract — the boundary with C
# ---------------------------------------------------------------------------


def build_observations(entity_id: UUID, as_of: datetime) -> list[Observation]:
    """Map the entity's GREEN_FLAG / PROOF_* events (as of T) to typed observations.

    This is the ONLY place events become observations. We read the events C wrote;
    we never import C's flag code. Contradicted events are dropped here and the
    drop is logged, so a repriced claim can't quietly keep moving the score.
    """
    contradicted = queries.contradicted_event_ids(entity_id, as_of)
    obs: list[Observation] = []
    for e in store.get_store().events(entity_id=entity_id, as_of=as_of):
        if e.kind not in _OBSERVATION_KINDS:
            continue
        if e.event_id in contradicted:
            log.info("score: dropping contradicted observation %s for %s", e.event_id, entity_id)
            continue
        obs.append(_observation_from_event(e))
    obs.sort(key=lambda o: o.observed_at)
    return obs


def _observation_from_event(e) -> Observation:
    p = e.payload or {}
    if "value" in p:
        value = float(p["value"])
    elif "y" in p:
        value = float(p["y"])
    else:
        value = 1.0 if p.get("fired", True) else 0.0
    value = min(1.0, max(0.0, value))

    self_consistency = float(p.get("self_consistency", e.confidence))
    self_consistency = min(1.0, max(0.05, self_consistency))

    penalty = float(p.get("source_penalty", _SOURCE_PENALTY.get(e.source, 1.0)))

    rule_ids = p.get("rule_ids")
    if rule_ids is None:
        rule_ids = [str(p["rule_id"])] if "rule_id" in p else []

    return Observation(
        entity_id=e.entity_id,
        observed_at=e.observed_at,
        value=value,
        self_consistency=self_consistency,
        source_penalty=penalty,
        event_ids=[e.event_id],
        rule_ids=[str(r) for r in rule_ids],
    )


# ---------------------------------------------------------------------------
# The filter
# ---------------------------------------------------------------------------


def _F(dt_days: float) -> np.ndarray:
    return np.array([[1.0, dt_days], [0.0, 1.0]])


def _Q(dt_days: float, q: float) -> np.ndarray:
    """Continuous white-noise-acceleration process noise. Couples level and trend,
    so the band grows during silence and the momentum stays honest."""
    dt = max(dt_days, 0.0)
    return q * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])


def _run_filter(entity_id: UUID, as_of: datetime) -> tuple[np.ndarray, np.ndarray, list[UUID]]:
    q, r0 = _params()
    obs = build_observations(entity_id, as_of)

    x = _X0.copy()
    P = _P0.copy()
    contributing: list[UUID] = []
    last_t: datetime | None = None

    for o in obs:
        if last_t is not None:
            dt = _dt_years(o.observed_at, last_t)
            F = _F(dt)
            x = F @ x
            P = F @ P @ F.T + _Q(dt, q)

        r = max(r0 / o.self_consistency * o.source_penalty, 1e-6)
        S = (_H @ P @ _H.T).item() + r
        K = (P @ _H.T) / S  # (2,1)
        residual = o.value - (_H @ x).item()
        x = x + (K.flatten() * residual)
        P = (np.eye(2) - K @ _H) @ P

        contributing.extend(o.event_ids)
        last_t = o.observed_at

    # Predict forward to as_of: staleness must widen the band honestly. A founder
    # last seen a year ago is more uncertain than one seen yesterday.
    if last_t is not None:
        dt = _dt_years(as_of, last_t)
        if dt > 0:
            F = _F(dt)
            x = F @ x
            P = F @ P @ F.T + _Q(dt, q)

    return x, P, contributing


def founder(entity_id: UUID, as_of: datetime) -> FounderScore:
    """The interface everyone depends on. Dispatches to the fallback when
    SCORE_MODEL=beta_binomial — one env var, no consumer change."""
    if _active_model() == "beta_binomial":
        from memory import score_fallback

        return score_fallback.founder(entity_id, as_of)
    return _founder_kalman(entity_id, as_of)


def _founder_kalman(entity_id: UUID, as_of: datetime) -> FounderScore:
    x, P, contributing = _run_filter(entity_id, as_of)
    return FounderScore(
        entity_id=entity_id,
        as_of=as_of,
        mu=float(x[0]),
        band=float(np.sqrt(max(P[0, 0], 0.0))),
        trend=float(x[1]),
        contributing_event_ids=contributing,
        model="kalman",
    )


def forecast(entity_id: UUID, as_of: datetime, k_days: int) -> tuple[float, float]:
    """k-step prediction interval — propagate P forward k days. This is the
    'Area of Research 1' answer and it's free once the filter works."""
    q, _ = _params()
    x, P, _ = _run_filter(entity_id, as_of)
    dt = float(k_days) / _DAYS_PER_YEAR
    F = _F(dt)
    x = F @ x
    P = F @ P @ F.T + _Q(dt, q)
    return float(x[0]), float(np.sqrt(max(P[0, 0], 0.0)))
