"""Decision gate. Owner: C. See C.md H8-12.

The absence classifier is the delicate part: signal-absent-because-irrelevant
(a designer with no GitHub) vs signal-absent-and-suspicious (an infra founder
claiming a distributed system with no code anywhere). Get this wrong and we punish
exactly the founders this thesis exists to find.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from intelligence import flags
from schema.events import Event, EventKind, FounderScore, GateDecision, GateOutcome, Source

_TECHNICAL_CLAIM_TERMS = (
    "distributed",
    "runtime",
    "compiler",
    "inference",
    "scheduler",
    "kernel",
)


def _claim_text(event: Event) -> str:
    value = event.payload.get("claim") or event.evidence_span or ""
    return str(value).lower()


def decide(
    company_id: UUID, founder_score: FounderScore, events: list[Event], as_of: datetime
) -> GateDecision:
    """Pure decision policy. Confidence remains explicit through the founder band."""
    evidence = [
        event
        for event in events
        if event.company_id == company_id
        and event.observed_at <= as_of
        and event.kind != EventKind.INTEGRITY
        and not flags.is_impeached(event)
    ]
    has_code_source = any(
        (
            event.kind in {EventKind.REPO_ACTIVITY, EventKind.COMMIT_BURST, EventKind.RELEASE}
            and event.source == Source.GITHUB
        )
        or (
            event.kind == EventKind.PROOF_ARTIFACT
            and event.source == Source.PROOF_PROTOCOL
            and (
                event.payload.get("works") is True
                and event.payload.get("sound") is True
                and isinstance(event.payload.get("artifact"), str)
                and bool(event.payload["artifact"].strip())
            )
        )
        for event in evidence
    )
    has_technical_claim = any(
        event.kind == EventKind.DECK_CLAIM
        and any(term in _claim_text(event) for term in _TECHNICAL_CLAIM_TERMS)
        for event in evidence
    )
    suspicious_absence = has_technical_claim and not has_code_source

    if founder_score.mu + founder_score.band < 0.45:
        return GateDecision(
            company_id=company_id,
            outcome=GateOutcome.NO_CALL,
            rationale="Even the upper confidence bound remains below the call threshold.",
            absence_is_suspicious=suspicious_absence,
        )
    if founder_score.mu >= 0.70 and founder_score.band <= 0.20:
        return GateDecision(
            company_id=company_id,
            outcome=GateOutcome.PROCEED,
            rationale="Capability evidence is strong enough and uncertainty is sufficiently narrow.",
            absence_is_suspicious=suspicious_absence,
        )
    if suspicious_absence and founder_score.mu < 0.60:
        return GateDecision(
            company_id=company_id,
            outcome=GateOutcome.NO_CALL,
            rationale="A central technical claim lacks the directly relevant artifact evidence.",
            absence_is_suspicious=True,
        )
    return GateDecision(
        company_id=company_id,
        outcome=GateOutcome.PROOF_PROTOCOL,
        rationale="Evidence is promising but too thin or uncertain; create a targeted proof.",
        absence_is_suspicious=suspicious_absence,
    )


def evaluate(company_id: UUID, as_of: datetime) -> GateDecision:
    """Store-backed wrapper. Every read and founder score is scoped to as_of."""
    from memory import score, store

    events = store.events(company_id=company_id, as_of=as_of)
    entity_ids = sorted(
        {event.entity_id for event in events if event.entity_id is not None}, key=str
    )
    if len(entity_ids) == 1:
        founder_score = score.founder(entity_ids[0], as_of)
    elif not entity_ids:
        founder_score = FounderScore(
            entity_id=UUID(int=0),
            as_of=as_of,
            mu=0.5,
            band=0.5,
            trend=0.0,
            contributing_event_ids=[],
        )
    else:
        return GateDecision(
            company_id=company_id,
            outcome=GateOutcome.PROOF_PROTOCOL,
            rationale="Founder identity is ambiguous; resolve ownership before making a call.",
            absence_is_suspicious=False,
        )
    return decide(company_id, founder_score, events, as_of)
