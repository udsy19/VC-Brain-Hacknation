"""Founder scoring with a calibrated observation boundary and Kalman filter.

Builder A's local-linear-trend filter remains authoritative. Builder C's green-flag
calibration, payload shapes, proof weighting, and contradiction references are
integrated at the event-to-observation boundary.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
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
OBSERVATION_KINDS = tuple(_OBSERVATION_KINDS)

# A's source weighting remains the baseline. C's proof-kind multiplier makes
# verified behavioral evidence materially less noisy without rewarding raw volume.
#
# WHAT THIS TABLE IS, AND WHAT IT IS NOT. Read as "code is trustworthy and prose is
# not" it is indefensible: that is true of a company whose product is code and false
# everywhere else, and it would mean the only channel a non-technical founder has is
# the noisiest one BY CONSTRUCTION, with no route to improvement however much
# independent confirmation arrives. The defensible reading, and the one these numbers
# actually track, is TIMESTAMP AUTHORITY AND THIRD-PARTY OBSERVABILITY: GitHub and HN
# carry server-assigned times the subject cannot set, arXiv carries a v1 submission
# date, a deck carries whatever the founder typed this morning. Nothing in that
# argument mentions code.
#
# The missing axis was CORROBORATION, and it is supplied below rather than by
# re-tuning these constants. See `_corroboration_multiplier`.
_SOURCE_PENALTY = {
    Source.PROOF_PROTOCOL: 0.15,
    Source.VALIDATOR: 0.6,
    Source.GITHUB: 0.6,
    Source.ARXIV: 0.7,
    Source.MANUAL: 1.2,
    Source.HN: 0.9,
    Source.WEB: 1.0,
    Source.DECK: 2.0,
}
SOURCE_PENALTY = _SOURCE_PENALTY
KIND_NOISE = {
    EventKind.GREEN_FLAG: 1.0,
    EventKind.PROOF_ARTIFACT: 0.2,
    EventKind.PROOF_BEHAVIOR: 0.2,
}

# --- Proof Protocol calibration -------------------------------------------------
# Spec 2c: a Proof Protocol result is a "high-weight, low-noise" observation, but its
# "confidence intervals stay wide and are displayed". Those two clauses pull in
# opposite directions and the multipliers above only honoured the first: the source
# penalty (0.15) and the kind noise (0.2) COMPOUND to an effective 0.03, roughly 30x
# less noise than any other observation. Two graded proof events then moved a
# cold-start founder with no public footprint to mu 0.944 with a band of 0.036 -- a
# higher score AND a tighter interval than a founder with 18 months of shipped
# artifacts. That inverts the requirement it was meant to implement.
#
# The corrective is not to make proof weak. It stays the strongest single observation
# type. It is to bound what a single 60-90 minute exercise can ever establish:
#
#   PROOF_NOISE_FLOOR  A proof result samples behaviour under one contrived task. That
#       sampling error does not shrink no matter how clean the submission is, so proof
#       observations carry an irreducible noise floor. Because the posterior level
#       variance after a diffuse-prior update is approximately r, this floor is what
#       keeps the band wide and displayed rather than a post-hoc clamp on the output.
#
#   PROOF_REPEAT_EXPONENT  Each grading of a challenge appends a fresh (artifact,
#       behaviour) pair under a new uuid5, so re-running the demo beat used to raise
#       the score without bound. Successive proof results for one entity are near-
#       duplicate measurements of the same latent trait, not independent draws, so the
#       k-th one is down-weighted by k**PROOF_REPEAT_EXPONENT. With exponent 2 the
#       total information from proof converges (sum 1/k**2 = pi**2/6), which bounds the
#       band from below at sqrt(6/pi**2) ~= 0.78 of the single-proof band no matter how
#       many times the exercise is repeated. An accumulated real track record has no
#       such ceiling, so it can always out-certain proof given enough milestones.
#
# The floor sits just below the noise of the strongest possible non-proof observation
# -- a GITHUB or VALIDATOR green flag at perfect self-consistency, r = r0 * 0.6 = 0.048
# -- so proof remains the strongest single observation type, and remains so flatly,
# immune to the source and consistency penalties that inflate everything else. It is
# still a measurement rather than a revelation, which is what the floor encodes.
#
# Note that mu is only weakly sensitive to this constant (0.846 -> 0.785 as the floor
# sweeps 0.05 -> 0.14): against a diffuse cold-start prior any informative reading pulls
# the level to roughly the observed value, which is correct Bayesian behaviour. It is
# the BAND, not the level, that must carry the caveat -- exactly what spec 2c asks for.
PROOF_NOISE_FLOOR = 0.045
PROOF_REPEAT_EXPONENT = 2.0
_PROOF_KINDS = frozenset({EventKind.PROOF_ARTIFACT, EventKind.PROOF_BEHAVIOR})

MU0 = 0.5
R0 = 0.08
P0 = (0.25, 0.25)  # public compatibility constants for calibration diagnostics
_X0 = np.array([MU0, 0.0])
_P0 = np.array([[P0[0], 0.0], [0.0, P0[1]]])
H = np.array([[1.0, 0.0]])
_H = H
_DAYS_PER_YEAR = 365.25
_CLAIM_REF_KEYS = ("claim_id", "claim_ids", "supporting_claim_ids", "supporting_claims")

# Green-flag rates are sensor yes-rates, not capability scores. This monotone
# calibration puts them on the same scale as the other observations while retaining
# the raw event payload and shrinking unknown evidence counts toward the cohort prior.
RATE_MID = 0.30
RATE_SLOPE = 8.0
# Keep calibration on the capability scale without inflating a high raw signal;
# this preserves the anti-volume invariant for repeated evidence.
SCORE_FLOOR, SCORE_CEIL = 0.12, 0.80
RATE_PRIOR = 0.26
SHRINK_K = 8.0


@dataclass(frozen=True)
class Observation:
    """Compatibility view used by C diagnostics; A uses ``SchemaObservation``."""

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


# --- Request-scoped memoization -------------------------------------------------
#
# Both functions below are pure in (args, store contents), and both were being
# recomputed many times per API request over an unchanged store:
#
#   * `contradicted_claim_ids` scans EVERY validation event in the corpus and is called
#     once per company by `build_observations`. That is O(companies x corpus) — the term
#     that turns a linear endpoint quadratic as the corpus grows.
#   * `founder` runs the whole Kalman filter, and `GET /companies` triggers it TWICE for
#     every company: once directly, and once more inside `intelligence.gate.evaluate`.
#
# The cache is deliberately NOT a module-level lru_cache. A global cache keyed on
# (entity, as_of) goes stale the moment anything appends to the store under the same
# cutoff — which is exactly what tests and the ingest path do — and a scorer that
# silently returns a pre-append answer is a much worse bug than a slow endpoint. So the
# cache exists only inside an explicit `scoring_cache()` block, where the caller is
# asserting the store does not change for the duration. Outside one, behaviour is
# byte-for-byte what it was before.
_CACHE: ContextVar[dict | None] = ContextVar("score_cache", default=None)


@contextmanager
def scoring_cache():
    """Memoize pure scoring reads for the duration of one read-only request.

    Nests safely: an inner block reuses the outer cache rather than shadowing it, so a
    handler that wraps a helper which also wraps does not lose the outer hits.
    """
    if _CACHE.get() is not None:
        yield
        return
    token = _CACHE.set({})
    try:
        yield
    finally:
        _CACHE.reset(token)


def contradicted_claim_ids(as_of: datetime) -> set[UUID]:
    cache = _CACHE.get()
    key = ("contradicted_claims", as_of)
    if cache is not None and key in cache:
        return cache[key]
    out = _contradicted_claim_ids(as_of)
    if cache is not None:
        cache[key] = out
    return out


def _contradicted_claim_ids(as_of: datetime) -> set[UUID]:
    out: set[UUID] = set()
    for event in store.events(as_of=as_of, kind=EventKind.VALIDATION_RESULT):
        for entry in _verdict_entries(event.payload):
            status = str(entry.get("status") or entry.get("verdict") or "").lower()
            if status.endswith("contradicted") and (claim_id := _as_uuid(entry.get("claim_id"))):
                out.add(claim_id)
    return out


def calibrate(rate: float, n_flags: int = 0) -> float:
    """Map a weighted green-flag yes-rate monotonically to capability scale."""
    try:
        count = int(n_flags)
    except (TypeError, ValueError):
        count = 0
    n = count if count > 0 else int(SHRINK_K)
    shrunk = (float(rate) * n + RATE_PRIOR * SHRINK_K) / (n + SHRINK_K)
    squashed = 1.0 / (1.0 + float(np.exp(-RATE_SLOPE * (shrunk - RATE_MID))))
    return float(SCORE_FLOOR + squashed * (SCORE_CEIL - SCORE_FLOOR))


def _flag_count(payload: dict) -> int:
    flags = payload.get("flags")
    if isinstance(flags, list):
        return len(flags)
    count = payload.get("n_flags") or payload.get("evaluated")
    return int(count) if isinstance(count, (int, float)) and not isinstance(count, bool) else 0


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
            weight = (
                float(weight)
                if isinstance(weight, (int, float)) and not isinstance(weight, bool)
                else 1.0
            )
            denominator += weight
            numerator += weight if bool(flag.get("fired")) else 0.0
        if denominator:
            return float(np.clip(numerator / denominator, 0.0, 1.0))
    # A per-rule RECEIPT is not an observation. Receipts carry `rule_id` and `fired`
    # so the trace can show which rules fired and on what evidence; the reading itself
    # lives on the single rollup. Scoring receipts too counts a founder once per rule
    # instead of once per evaluation — it narrows the band by the square root of the
    # rule count and lets how MANY rules exist outweigh how well the founder did.
    # Measured with this clause active: the adversarial burst outscored its own
    # legitimate control, 37.0 to 17.5, inverting the beat that exists to show we do
    # not false-positive fast builders.
    if "rule_id" in payload:
        return None
    if "fired" in payload:
        return 1.0 if payload["fired"] else 0.0
    return None


# --- Corroboration, the axis the source table was missing ------------------------
#
# PENALISE SELF-PUBLISHED, REWARD INDEPENDENTLY CORROBORATED — orthogonally to
# whether the artifact is code. `intelligence/flags._corroboration_reading` counts
# how many channels stand behind a reading that are neither self-attested (the deck,
# a MANUAL note) nor self-published (a domain the founder controls, per
# `core.search.SELF_PUBLISHED_HINTS`), and this prices that count.
#
# The curve is the one `data/traits.json` already argues for and is concave for the
# same reason: one channel is a claim, two are a claim plus a witness, three are a
# pattern, and the third witness adds less than the second. It multiplies OBSERVATION
# NOISE only. It cannot move y_t, so corroboration widens or narrows the band without
# ever inventing capability — the same separation the trait layer maintains.
#
# NEUTRAL BY DEFAULT AND DELIBERATELY SO: an observation whose payload carries no
# corroboration reading multiplies by 1.0. Nothing that predates this change moves.
_CORROBORATION_MULTIPLIER = {0: 1.25, 1: 1.0, 2: 0.85}
_CORROBORATION_FLOOR = 0.75  # three or more independent witnesses


def _corroboration_multiplier(payload: dict) -> float:
    """Noise multiplier from the number of independent witnesses, or 1.0 if unstated."""
    channels = payload.get("independent_channels")
    if not isinstance(channels, int) or isinstance(channels, bool) or channels < 0:
        return 1.0
    if payload.get("self_published_only") is True:
        # Everything we hold is the founder describing themselves. That is not a
        # finding about their quality — it is a statement that nobody else has said
        # anything yet, and the band should show it.
        return _CORROBORATION_MULTIPLIER[0]
    return _CORROBORATION_MULTIPLIER.get(channels, _CORROBORATION_FLOOR)


def _source_penalty(event: Event, payload: dict) -> float:
    value = payload.get("source_penalty", _SOURCE_PENALTY.get(event.source, 1.0))
    base = (
        float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 1.0
    )
    return base * _corroboration_multiplier(payload)


def _kind_noise(event: Event) -> float:
    return KIND_NOISE.get(event.kind, 1.0)


def _noise(event: Event, payload: dict) -> float:
    consistency = payload.get("self_consistency", event.confidence)
    consistency = (
        float(consistency)
        if isinstance(consistency, (int, float)) and not isinstance(consistency, bool)
        else 1.0
    )
    consistency = float(np.clip(consistency, 0.05, 1.0))
    return max(
        _params()[1] / consistency * _source_penalty(event, payload) * _kind_noise(event), 1e-6
    )


def _observation_from_event(event: Event, entity_id: UUID) -> SchemaObservation | None:
    payload = event.payload if isinstance(event.payload, dict) else {}
    value = _derive_y(payload)
    if value is None:
        return None
    if event.kind == EventKind.GREEN_FLAG:
        value = calibrate(value, _flag_count(payload))
    consistency = payload.get("self_consistency", event.confidence)
    consistency = (
        float(consistency)
        if isinstance(consistency, (int, float)) and not isinstance(consistency, bool)
        else 1.0
    )
    consistency = float(np.clip(consistency, 0.05, 1.0))
    penalty = _source_penalty(event, payload) * _kind_noise(event)
    rule_ids = payload.get("rule_ids")
    if isinstance(rule_ids, (list, tuple)):
        normalized_rule_ids = [str(rule_id) for rule_id in rule_ids]
    elif rule_ids is None:
        normalized_rule_ids = [str(payload["rule_id"])] if "rule_id" in payload else []
    else:
        normalized_rule_ids = [str(rule_ids)]
    return SchemaObservation(
        entity_id=entity_id,
        observed_at=event.observed_at,
        value=value,
        self_consistency=consistency,
        source_penalty=penalty,
        event_ids=[event.event_id],
        rule_ids=normalized_rule_ids,
    )


def _apply_proof_calibration(
    observations_for_entity: list[SchemaObservation], proof_event_ids: set[UUID]
) -> list[SchemaObservation]:
    """Floor proof noise and give repeated proof results diminishing returns.

    Both adjustments land on ``source_penalty`` rather than on the filter, so the
    resulting ``r`` stays derivable from the stored observation and the score remains
    auditable end to end. See the PROOF_NOISE_FLOOR block above for the reasoning.
    """
    r0 = _params()[1]
    calibrated: list[SchemaObservation] = []
    seen = 0
    for observation in observations_for_entity:
        if not proof_event_ids.intersection(observation.event_ids):
            calibrated.append(observation)
            continue
        seen += 1
        floor_penalty = PROOF_NOISE_FLOOR * observation.self_consistency / r0
        penalty = max(observation.source_penalty, floor_penalty) * seen**PROOF_REPEAT_EXPONENT
        calibrated.append(observation.model_copy(update={"source_penalty": penalty}))
    return calibrated


def _contradicted_events_for(entity_id: UUID, as_of: datetime) -> set[UUID]:
    """`queries.contradicted_event_ids`, but ONE corpus-wide pair of queries per request.

    The per-entity version issues two kind-filtered queries every time it is called,
    and it is called once per founder. Against a remote database that is two network
    round trips per company for a pair of event kinds that are RARE — the corpus
    currently holds zero validation results, so both queries usually return nothing
    and cost a round trip anyway. Profiling `GET /companies` at 60 companies put 7.0s
    of 7.8s inside psycopg's wait, across 327 queries; this pair was 120 of them.

    Filtering a corpus-wide fetch by entity_id in Python is exactly equivalent to
    asking the database to filter per entity, so the returned set is unchanged. Only
    valid inside `scoring_cache()`, where the store is asserted not to change;
    outside one it falls straight through to the per-entity query.
    """
    cache = _CACHE.get()
    if cache is None:
        return queries.contradicted_event_ids(entity_id, as_of)

    key = ("contradiction_index", as_of)
    index = cache.get(key)
    if index is None:
        index = {}
        backend = store.get_store()
        for kind in (EventKind.CONTRADICTION, EventKind.VALIDATION_RESULT):
            for event in backend.events(as_of=as_of, kind=kind):
                if event.entity_id is None:
                    continue
                if kind is EventKind.VALIDATION_RESULT and (
                    str(event.payload.get("status", "")).lower() != "contradicted"
                ):
                    continue
                index.setdefault(event.entity_id, set()).update(queries._targets(event))
        cache[key] = index
    return index.get(entity_id, set())


def _observation_events_for(entity_id: UUID, as_of: datetime) -> list[Event]:
    """This entity's SCORABLE events — the only kinds `build_observations` reads.

    The unindexed form fetches every event for the entity and discards all but three
    kinds, once per founder. Inside a request that is one round trip per company to
    retrieve rows that are then mostly thrown away. Indexed, it is three kind-scoped
    queries for the whole corpus, grouped once.

    Ordering mirrors the backend's own `order by observed_at, ingested_at, event_id`
    exactly. `build_observations` re-sorts afterwards and everything before that sort is
    order-independent, so this is belt-and-braces — but an index that silently reorders
    equal-timestamped events would be an unpleasant thing to discover later, and
    matching the query costs nothing.
    """
    cache = _CACHE.get()
    if cache is None:
        return [
            e
            for e in store.get_store().events(entity_id=entity_id, as_of=as_of)
            if e.kind in _OBSERVATION_KINDS
        ]

    key = ("observation_index", as_of)
    index = cache.get(key)
    if index is None:
        index = {}
        backend = store.get_store()
        for kind in _OBSERVATION_KINDS:
            for event in backend.events(as_of=as_of, kind=kind):
                if event.entity_id is not None:
                    index.setdefault(event.entity_id, []).append(event)
        for events in index.values():
            events.sort(key=lambda e: (e.observed_at, e.ingested_at, str(e.event_id)))
        cache[key] = index
    return index.get(entity_id, [])


def build_observations(entity_id: UUID, as_of: datetime) -> list[SchemaObservation]:
    contradicted_events = _contradicted_events_for(entity_id, as_of)
    contradicted_claims = contradicted_claim_ids(as_of)
    observations_for_entity: list[SchemaObservation] = []
    proof_event_ids: set[UUID] = set()
    for event in _observation_events_for(entity_id, as_of):
        if event.kind in _PROOF_KINDS:
            proof_event_ids.add(event.event_id)
        if (
            event.event_id in contradicted_events
            or _claim_refs(event.payload) & contradicted_claims
        ):
            log.info(
                "score: dropping contradicted observation %s for %s", event.event_id, entity_id
            )
            continue
        if observation := _observation_from_event(event, entity_id):
            observations_for_entity.append(observation)
        else:
            log.debug("score: unrecognised payload shape on %s (%s)", event.event_id, event.kind)
    observations_for_entity.sort(key=lambda item: (item.observed_at, str(item.event_ids[0])))

    # Exact duplicate evidence at one world-time is one measurement, not N votes.
    # Keep all event IDs so the score remains auditable and receipts are lossless.
    deduplicated: list[SchemaObservation] = []
    for observation in observations_for_entity:
        if deduplicated:
            previous = deduplicated[-1]
            same_measurement = (
                previous.observed_at == observation.observed_at
                and previous.value == observation.value
                and previous.self_consistency == observation.self_consistency
                and previous.source_penalty == observation.source_penalty
                and previous.rule_ids == observation.rule_ids
            )
            if same_measurement:
                deduplicated[-1] = previous.model_copy(
                    update={"event_ids": [*previous.event_ids, *observation.event_ids]}
                )
                continue
        deduplicated.append(observation)
    return _apply_proof_calibration(deduplicated, proof_event_ids)


def observations(entity_id: UUID, as_of: datetime) -> ObservationSet:
    """C-compatible diagnostic view over A's shared observation boundary."""
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


# Momentum is not permanent. A trend measured from past readings must not be
# extrapolated indefinitely: velocity decays with MOMENTUM_HALFLIFE_DAYS, which bounds
# total drift to v0 / DECAY_RATE, and uncertainty never exceeds the no-evidence prior.
#
# This decay applies to EVERY propagation, not only to trailing silence. Applying it
# only after the last observation (as this filter previously did) left the between-
# observation transition as the undamped [[1, dt], [0, 1]], and that is what made the
# band WIDEN as evidence accumulated. Under the undamped transition the propagated
# level variance is P00 + 2*dt*P01 + dt**2 * P11 + Q00; measured on Tensorpage's real
# 11-observation history the dt**2 * P11 term contributed 1e-3 to 7e-3 per step while
# Q00 contributed only 1e-5 to 5e-5. Process noise was NOT the cause. The cause was
# that the level inherits the full velocity variance scaled by dt**2, so the band
# tracked observation SPACING rather than observation COUNT: it fell while readings
# were monthly and rose the moment the cadence slipped to bi-monthly.
#
# Damping the velocity bounds the velocity-to-level coupling at 1/DECAY_RATE instead
# of letting it grow linearly in dt, so a longer gap can no longer dominate the update.
# The band then falls monotonically with every observation on real, irregularly spaced
# histories. The half-life is 180 days rather than 90 because 90 damps so hard that the
# level under-tracks its own readings (Tensorpage settled at mu 0.607 against readings
# of 0.672); at 180 days the level tracks the data (0.668) and the trend still decays
# correctly as a series plateaus.
MOMENTUM_HALFLIFE_DAYS = 180.0
_DECAY_RATE = np.log(2.0) / (MOMENTUM_HALFLIFE_DAYS / _DAYS_PER_YEAR)


def _F(dt_years: float) -> np.ndarray:
    """Transition for a local-linear-trend level whose velocity decays exponentially.

    ``dt_years`` is in YEARS, so the velocity component of the state -- reported as
    ``FounderScore.trend`` -- is in capability units per YEAR. See ``trend_per_days``.
    """
    dt = max(float(dt_years), 0.0)
    decay = float(np.exp(-_DECAY_RATE * dt))
    return np.array([[1.0, (1.0 - decay) / _DECAY_RATE], [0.0, decay]])


def _cap_covariance(covariance: np.ndarray) -> np.ndarray:
    """Cap each variance at the no-evidence prior WITHOUT destroying positive-definiteness.

    THIS IS THE FIX FOR THE DIVERGENCE, AND THE OLD CODE HERE WAS THE CAUSE OF IT.
    The cap itself is sound and is kept: however long the silence, we can never be more
    uncertain about a founder than we were before we had ever heard of them. What was
    wrong was writing the diagonal entries independently --

        covariance[0, 0] = min(covariance[0, 0], P0[0])
        covariance[1, 1] = min(covariance[1, 1], P0[1])

    -- which shrinks a variance while leaving the covariance term that references it
    untouched, and a 2x2 with a large off-diagonal relative to its diagonals is not a
    covariance matrix at all. Measured on the real corpus, `peerd` propagates across a
    7.25-year gap to

        P = [[1.465898, 0.262807], [0.262807, 0.072498]]   det = +3.72e-02  (valid)

    and the clamp rewrites P00 to 0.25, giving det = 0.25*0.072498 - 0.262807**2 =
    -5.09e-02 with eigenvalues (0.4386, -0.1161). The posterior is already indefinite
    BEFORE the measurement update runs; `(I - K H) P` then feeds the negative eigenvalue
    into P11 and every later step inherits it. That is the whole mechanism -- it is not
    dt sign, not the velocity coupling, and not asymmetry.

    The correct way to impose a per-variance ceiling is a symmetric diagonal rescaling,
    P -> D P D with D = diag(sqrt(min(1, P0[i] / P_ii))). That is a CONGRUENCE transform,
    so it preserves positive semi-definiteness exactly, and it reproduces the intended
    ceiling exactly (the new P_ii is min(P_ii, P0[i])). The off-diagonal shrinks by the
    same factors, which is the correct reading: capping how uncertain we are about the
    level must also cap how much of that uncertainty can be attributed to the velocity.
    On the peerd step above it yields det = +6.35e-03 with the diagonal at the ceiling.

    This is a numerical correction, not a mask. Nothing is clamped to a floor and no
    absolute value is taken; a genuinely invalid posterior would still reach the
    divergence guard, which stays exactly as it is.
    """
    variances = np.diag(covariance)
    ceilings = np.array([P0[0], P0[1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(variances > ceilings, ceilings / variances, 1.0)
    scale = np.sqrt(np.clip(ratios, 0.0, 1.0))
    return (covariance * scale[:, None]) * scale[None, :]


def _propagate(
    x: np.ndarray, covariance: np.ndarray, dt_years: float, q: float
) -> tuple[np.ndarray, np.ndarray]:
    """Advance state and covariance across ``dt_years``, damping momentum as it goes.

    The level integrates a decaying velocity rather than a constant one, so a short
    burst of improvement cannot compound into an unbounded forecast. Covariance is
    capped at the no-evidence prior by ``_cap_covariance``, which does so without
    breaking the positive-definiteness the filter depends on.
    """
    transition = _F(dt_years)
    x = transition @ x
    covariance = transition @ covariance @ transition.T + _Q(dt_years, q)
    return x, _cap_covariance(covariance)


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
            x, covariance = _propagate(x, covariance, _dt_years(observation.observed_at, last_t), q)
        measurement_noise = _noise_for_schema(observation)
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
            x, covariance = _propagate(x, covariance, gap, q)
    return x, covariance, contributing


def trend_per_days(trend: float, days: float) -> float:
    """Convert ``FounderScore.trend`` (capability per YEAR) to capability per ``days``.

    ``trend`` is the velocity component of the filter state, and every transition is
    built from ``_dt_years``, so its unit is capability-per-year -- NOT per day. A
    renderer that wants a per-30-day delta must divide by 365.25, not multiply by 30:
    ``trend_per_days(fs.trend, 30)``, i.e. ``fs.trend * 30 / 365.25``. Treating the
    value as per-day overstates it by a factor of 365.25.
    """
    return float(trend) * float(days) / _DAYS_PER_YEAR


def founder(entity_id: UUID, as_of: datetime) -> FounderScore:
    """Score ``entity_id`` from evidence observed at or before ``as_of``.

    Units of the returned ``FounderScore``:

    * ``mu``   -- capability on a 0..1 scale.
    * ``band`` -- one posterior standard deviation of ``mu``, same 0..1 scale.
    * ``trend`` -- velocity of ``mu`` in capability units per YEAR. Use
      ``trend_per_days`` to render it over any other horizon.
    """
    cache = _CACHE.get()
    key = ("founder", entity_id, as_of, _active_model())
    if cache is not None and key in cache:
        return cache[key]
    result = _founder_uncached(entity_id, as_of)
    if cache is not None:
        cache[key] = result
    return result


def _founder_uncached(entity_id: UUID, as_of: datetime) -> FounderScore:
    if _active_model() == "beta_binomial":
        from memory import score_fallback

        return score_fallback.founder(entity_id, as_of)
    state, covariance, contributing = _run_filter(entity_id, as_of)

    # A DIVERGED FILTER IS NOT A CONFIDENT ZERO.
    #
    # `covariance` is a posterior covariance, so its diagonal cannot be negative. The
    # update below uses the textbook `P = (I - K H) P` form, which is not guaranteed to
    # stay positive-definite in floating point, and it loses that property when a long
    # propagation makes P large before an update. Sourcing real founders exposed this
    # immediately: the GitHub scanner stamps a profile with the ACCOUNT CREATION date,
    # so someone who opened an account in 2013 and shipped in 2026 hands the filter a
    # 13-year gap between two observations. 8 of 130 founders diverged that way.
    #
    # What made it dangerous was the reporting, not the divergence. `band` was
    # `sqrt(max(cov, 0.0))` and `mu` was `clip(state, 0, 1)`, so a state of -23.6 with a
    # variance of -3.6 was published as mu=0.000, band=0.000 — a founder whose four
    # observations averaged 0.36 presented as a CONFIDENT ZERO, the most damaging output
    # this system can produce about a real person. `max(..., 0.0)` turned "the numbers
    # are invalid" into "we are perfectly certain", which is the absence-vs-weakness
    # failure in its purest form.
    #
    # So an invalid posterior returns the UNINFORMATIVE PRIOR: we ran the filter and it
    # did not converge, therefore we do not know. That is the honest reading, and it is
    # strictly safer than the clip. This cannot re-tune anything, because a posterior
    # that was already valid is returned unchanged.
    diverged = (
        not np.all(np.isfinite(state))
        or not np.all(np.isfinite(covariance))
        or covariance[0, 0] < 0.0
        or covariance[1, 1] < 0.0
    )
    if diverged:
        log.warning(
            "score: filter diverged for %s (state=%s, var=%s) — returning the "
            "uninformative prior rather than a confident zero",
            entity_id,
            state.tolist(),
            [covariance[0, 0], covariance[1, 1]],
        )
        state, covariance = _X0.copy(), _P0.copy()

    return FounderScore(
        entity_id=entity_id,
        as_of=as_of,
        mu=float(np.clip(state[0], 0.0, 1.0)),
        band=float(np.sqrt(max(covariance[0, 0], 0.0))),
        trend=float(state[1]),
        contributing_event_ids=contributing,
        model="kalman",
    )


def forecast(entity_id: UUID, as_of: datetime, k_days: int) -> tuple[float, float]:
    q, _ = _params()
    state, covariance, _ = _run_filter(entity_id, as_of)
    dt_years = float(k_days) / _DAYS_PER_YEAR
    state, covariance = _propagate(state, covariance, dt_years, q)
    return float(np.clip(state[0], 0.0, 1.0)), float(np.sqrt(max(covariance[0, 0], 0.0)))
