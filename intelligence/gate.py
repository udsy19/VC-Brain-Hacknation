"""Decision gate. Owner: C. See C.md H8-12.

The absence classifier is the delicate part: signal-absent-because-irrelevant
(a designer with no GitHub) vs signal-absent-and-suspicious (an infra founder
claiming a distributed system with no code anywhere). Get this wrong and we punish
exactly the founders this thesis exists to find.

The abstention boundary is the second delicate part, and it is no longer a constant.
When a conformal calibration is supplied (see intelligence/conformal.py), the gate
abstains because a 1-alpha prediction interval straddles the clearing threshold — a
reason it can state out loud — and it PROCEEDs when that interval sits wholly above the
threshold. When the calibration cannot be earned, the base policy below still runs and
the rationale says the conformal layer was not calibrated. The fallback is never silent.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from intelligence import flags
from intelligence import conformal
from intelligence.conformal import ConformalCalibration
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
    company_id: UUID,
    founder_score: FounderScore,
    events: list[Event],
    as_of: datetime,
    *,
    calibration: ConformalCalibration | None = None,
) -> GateDecision:
    """Pure decision policy. Confidence remains explicit through the founder band.

    ``calibration`` is optional and defaults to off: with no calibration the historical
    constant-threshold policy runs unchanged. Supply one and the conformal interval
    governs the abstention boundary instead, with its own reasoning attached.
    """
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

    interval = calibration.interval(founder_score.mu, founder_score.band) if calibration else None
    note = calibration.describe(interval) if calibration else None

    def _decision(outcome: GateOutcome, rationale: str) -> GateDecision:
        return GateDecision(
            company_id=company_id,
            outcome=outcome,
            rationale=f"{rationale} {note}" if note else rationale,
            absence_is_suspicious=suspicious_absence,
        )

    if interval is not None:
        # The conformal boundary supersedes the constants for exactly two calls: abstain
        # when the interval straddles the threshold, proceed when it clears it outright.
        # Everything below the threshold still runs the base ladder, so PROOF_PROTOCOL —
        # "promising but thin, go get evidence" — keeps its meaning.
        if interval.verdict == "ambiguous":
            return _decision(
                GateOutcome.NO_CALL,
                "No call: the evidence cannot distinguish clearing from not clearing.",
            )
        if interval.verdict == "clears" and not suspicious_absence:
            return _decision(
                GateOutcome.PROCEED,
                "Proceed: the calibrated interval clears the threshold outright.",
            )

    if founder_score.mu + founder_score.band < 0.45:
        return _decision(
            GateOutcome.NO_CALL,
            "Even the upper confidence bound remains below the call threshold.",
        )
    if founder_score.mu >= 0.70 and founder_score.band <= 0.20:
        return _decision(
            GateOutcome.PROCEED,
            "Capability evidence is strong enough and uncertainty is sufficiently narrow.",
        )
    if suspicious_absence and founder_score.mu < 0.60:
        return _decision(
            GateOutcome.NO_CALL,
            "A central technical claim lacks the directly relevant artifact evidence.",
        )
    return _decision(
        GateOutcome.PROOF_PROTOCOL,
        "Evidence is promising but too thin or uncertain; create a targeted proof.",
    )


def evaluate(
    company_id: UUID, as_of: datetime, *, alpha: float = conformal.DEFAULT_ALPHA
) -> GateDecision:
    """Store-backed wrapper. Every read and founder score is scoped to as_of.

    ``alpha`` is the conformal target error rate and is stated in the rationale. The
    calibration is built from the labelled backtest cohort at the same cutoff, minus this
    company, so nothing calibrates on the point it is judging.
    """
    from memory import score, store

    calibration = conformal.from_store(as_of, alpha=alpha).for_company(company_id)
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
    return decide(company_id, founder_score, events, as_of, calibration=calibration)
