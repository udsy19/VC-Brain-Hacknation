"""Founder Score: local-linear-trend Kalman filter.

The A scorer remains authoritative. C's payload shapes and contradiction references
are accepted at the observation boundary, while raw sourcing events remain ignored.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

import numpy as np

from core.config import settings
from memory import queries, store
from schema.events import Event, EventKind, FounderScore, Observation as SchemaObservation, Source

log = logging.getLogger(__name__)

_OBSERVATION_KINDS = {
    EventKind.GREEN_FLAG,
    EventKind.PROOF_ARTIFACT,
    EventKind.PROOF_BEHAVIOR,
}
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

MU0 = 0.5
R0 = 0.08
P0 = (0.25, 0.25)  # compatibility constants for C's calibration tests
_X0 = np.array([MU0, 0.0])
_P0 = np.array([[P0[0], 0.0], [0.0, P0[1]]])
_H = np.array([[1.0, 0.0]])
_DAYS_PER_YEAR = 365.25
_CLAIM_REF_KEYS = ("claim_id", "claim_ids", "supporting_claim_ids", "supporting_claims")


@dataclass(frozen=True)
class Observation:
    """Compatibility view used by C's diagnostics; A uses SchemaObservation."""

    event_id: UUID
    observed_at: datetime
    y: float
    r: float


@dataclass(frozen=True)
class ObservationSet:
    kept: list[Observation]
    dropped_contradicted: list[UUID]


def _params() -> tuple[float, float]:
    return float(os.getenv("SCORE_Q", "0.01")), float(os.getenv("SCORE_R0", str(R0)))


def _dt_years(later: datetime, earlier: datetime) -> float:
    return (later - earlier).total_seconds() / 86400.0 / _DAYS_PER_YEAR


def _active_model() -> str:
    return os.getenv("SCORE_MODEL", settings.score_model)


def _as_uuid(value: object) -> UUID | None:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


def _claim_refs(payload: dict) -> set[UUID]:
    refs: set[UUID] = set()
    for key in _CLAIM_REF_KEYS:
        value = payload.get(key)
        items = value if isinstance(value, (list, tuple)) else [value]
        for item in items:
            if isinstance(item, dict):
                item = item.get("claim_id")
            if parsed := _as_uuid(item):
                refs.add(parsed)
    return refs


def _verdict_entries(payload: dict) -> list[dict]:
    nested = payload.get("claims") or payload.get("verdicts")
    return (
        [entry for entry in nested if isinstance(entry, dict)]
        if isinstance(nested, list)
        else [payload]
    )


def contradicted_claim_ids(as_of: datetime) -> set[UUID]:
    out: set[UUID] = set()
    for event in store.events(as_of=as_of, kind=EventKind.VALIDATION_RESULT):
        for entry in _verdict_entries(event.payload):
            status = str(entry.get("status") or entry.get("verdict") or "").lower()
            if status.endswith("contradicted") and (claim_id := _as_uuid(entry.get("claim_id"))):
                out.add(claim_id)
    return out


def _derive_y(payload: dict) -> float | None:
    for key in ("value", "y", "yes_rate", "score"):
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(np.clip(value, 0.0, 1.0))
    flags = payload.get("flags")
    if isinstance(flags, list) and flags:
        numerator = denominator = 0.0
        for flag in flags:
            if not isinstance(flag, dict):
                continue
            weight = flag.get("weight", 1.0)
            weight = float(weight) if isinstance(weight, (int, float)) else 1.0
            denominator += weight
            numerator += weight if bool(flag.get("fired")) else 0.0
        if denominator:
            return float(np.clip(numerator / denominator, 0.0, 1.0))
    if "fired" in payload:
        return 1.0 if payload["fired"] else 0.0
    return None


def _noise(event: Event, payload: dict) -> float:
    consistency = payload.get("self_consistency", event.confidence)
    consistency = float(np.clip(consistency, 0.05, 1.0))
    penalty = float(payload.get("source_penalty", _SOURCE_PENALTY.get(event.source, 1.0)))
    return max(_params()[1] / consistency * penalty, 1e-6)


def _observation_from_event(event: Event, entity_id: UUID) -> SchemaObservation | None:
    payload = event.payload if isinstance(event.payload, dict) else {}
    value = _derive_y(payload)
    if value is None:
        return None
    consistency = float(np.clip(payload.get("self_consistency", event.confidence), 0.05, 1.0))
    penalty = float(payload.get("source_penalty", _SOURCE_PENALTY.get(event.source, 1.0)))
    rule_ids = payload.get("rule_ids")
    if rule_ids is None:
        rule_ids = [str(payload["rule_id"])] if "rule_id" in payload else []
    return SchemaObservation(
        entity_id=entity_id,
        observed_at=event.observed_at,
        value=value,
        self_consistency=consistency,
        source_penalty=penalty,
        event_ids=[event.event_id],
        rule_ids=[str(rule_id) for rule_id in rule_ids],
    )


def build_observations(entity_id: UUID, as_of: datetime) -> list[SchemaObservation]:
    contradicted_events = queries.contradicted_event_ids(entity_id, as_of)
    contradicted_claims = contradicted_claim_ids(as_of)
    observations: list[SchemaObservation] = []
    for event in store.get_store().events(entity_id=entity_id, as_of=as_of):
        if event.kind not in _OBSERVATION_KINDS:
            continue
        if (
            event.event_id in contradicted_events
            or _claim_refs(event.payload) & contradicted_claims
        ):
            log.info(
                "score: dropping contradicted observation %s for %s", event.event_id, entity_id
            )
            continue
        if observation := _observation_from_event(event, entity_id):
            observations.append(observation)
        else:
            log.debug("score: unrecognised payload shape on %s (%s)", event.event_id, event.kind)
    observations.sort(key=lambda item: (item.observed_at, str(item.event_ids[0])))
    return observations


def observations(entity_id: UUID, as_of: datetime) -> ObservationSet:
    """C-compatible diagnostic view over the shared A observation boundary."""
    kept_schema = build_observations(entity_id, as_of)
    kept_ids = {event_id for item in kept_schema for event_id in item.event_ids}
    kept = [
        Observation(item.event_ids[0], item.observed_at, item.value, _noise_for_schema(item))
        for item in kept_schema
    ]
    dropped: list[UUID] = []
    for event in store.get_store().events(entity_id=entity_id, as_of=as_of):
        if event.kind in _OBSERVATION_KINDS and _derive_y(event.payload) is not None:
            if event.event_id not in kept_ids:
                dropped.append(event.event_id)
    return ObservationSet(kept, dropped)


def _noise_for_schema(observation: SchemaObservation) -> float:
    return max(_params()[1] / observation.self_consistency * observation.source_penalty, 1e-6)


def _F(dt_years: float) -> np.ndarray:
    return np.array([[1.0, dt_years], [0.0, 1.0]])


def _Q(dt_years: float, q: float) -> np.ndarray:
    dt = max(dt_years, 0.0)
    return q * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])


def _run_filter(entity_id: UUID, as_of: datetime) -> tuple[np.ndarray, np.ndarray, list[UUID]]:
    q, _ = _params()
    observations_for_entity = build_observations(entity_id, as_of)
    x = _X0.copy()
    covariance = _P0.copy()
    contributing: list[UUID] = []
    last_t: datetime | None = None
    for observation in observations_for_entity:
        if last_t is not None:
            transition = _F(_dt_years(observation.observed_at, last_t))
            x = transition @ x
            covariance = transition @ covariance @ transition.T + _Q(
                _dt_years(observation.observed_at, last_t), q
            )
        measurement_noise = max(
            _params()[1] / observation.self_consistency * observation.source_penalty, 1e-6
        )
        innovation = (_H @ covariance @ _H.T).item() + measurement_noise
        gain = (covariance @ _H.T) / innovation
        residual = observation.value - (_H @ x).item()
        x = x + gain.flatten() * residual
        covariance = (np.eye(2) - gain @ _H) @ covariance
        contributing.extend(observation.event_ids)
        last_t = observation.observed_at
    if last_t is not None:
        gap = _dt_years(as_of, last_t)
        if gap > 0:
            transition = _F(gap)
            x = transition @ x
            covariance = transition @ covariance @ transition.T + _Q(gap, q)
    return x, covariance, contributing


def founder(entity_id: UUID, as_of: datetime) -> FounderScore:
    if _active_model() == "beta_binomial":
        from memory import score_fallback

        return score_fallback.founder(entity_id, as_of)
    state, covariance, contributing = _run_filter(entity_id, as_of)
    return FounderScore(
        entity_id=entity_id,
        as_of=as_of,
        mu=float(state[0]),
        band=float(np.sqrt(max(covariance[0, 0], 0.0))),
        trend=float(state[1]),
        contributing_event_ids=contributing,
        model="kalman",
    )


def forecast(entity_id: UUID, as_of: datetime, k_days: int) -> tuple[float, float]:
    q, _ = _params()
    state, covariance, _ = _run_filter(entity_id, as_of)
    transition = _F(float(k_days) / _DAYS_PER_YEAR)
    state = transition @ state
    covariance = transition @ covariance @ transition.T + _Q(float(k_days) / _DAYS_PER_YEAR, q)
    return float(state[0]), float(np.sqrt(max(covariance[0, 0], 0.0)))
