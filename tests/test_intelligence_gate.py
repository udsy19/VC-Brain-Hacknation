"""RED-phase tests for the locked pure and store-backed gate contract."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from intelligence import gate
from schema.events import Event, EventKind, FounderScore, GateOutcome, Source

T0 = datetime(2025, 4, 5, tzinfo=timezone.utc)
COMPANY_ID = UUID("10000000-0000-0000-0000-000000000000")
LOW_ENTITY = UUID("20000000-0000-0000-0000-000000000000")
HIGH_ENTITY = UUID("90000000-0000-0000-0000-000000000000")


def _score(*, mu: float, band: float, entity_id: UUID = LOW_ENTITY) -> FounderScore:
    return FounderScore(entity_id=entity_id, as_of=T0, mu=mu, band=band, trend=0.0)


def _event(
    *,
    source: Source,
    claim: str = "workflow collaboration",
    entity_id: UUID = LOW_ENTITY,
    observed_at: datetime = T0,
    integrity_flags: list[str] | None = None,
    kind: EventKind = EventKind.DECK_CLAIM,
    company_id: UUID = COMPANY_ID,
) -> Event:
    return Event(
        company_id=company_id,
        entity_id=entity_id,
        kind=kind,
        source=source,
        observed_at=observed_at,
        payload={"claim": claim},
        integrity_flags=integrity_flags or [],
    )


@pytest.mark.parametrize(
    ("mu", "band", "expected"),
    [
        (0.70, 0.20, GateOutcome.PROCEED),
        (0.80, 0.05, GateOutcome.PROCEED),
        (0.20, 0.20, GateOutcome.NO_CALL),
        (0.44, 0.00, GateOutcome.NO_CALL),
        (0.50, 0.20, GateOutcome.PROOF_PROTOCOL),
        (0.69, 0.20, GateOutcome.PROOF_PROTOCOL),
        (0.70, 0.21, GateOutcome.PROOF_PROTOCOL),
    ],
)
def test_decide_thresholds(mu, band, expected) -> None:
    decision = gate.decide(COMPANY_ID, _score(mu=mu, band=band), [], T0)
    assert decision.company_id == COMPANY_ID
    assert decision.outcome is expected
    assert decision.rationale.strip()
    assert decision.absence_is_suspicious is False


@pytest.mark.parametrize(
    "keyword", ["distributed", "runtime", "compiler", "inference", "scheduler", "kernel"]
)
def test_low_score_technical_claim_without_github_changes_proof_to_no_call(keyword) -> None:
    decision = gate.decide(
        COMPANY_ID,
        _score(mu=0.59, band=0.20),
        [_event(source=Source.DECK, claim=f"A {keyword.upper()} system")],
        T0,
    )
    assert decision.outcome is GateOutcome.NO_CALL
    assert decision.absence_is_suspicious is True
    assert decision.rationale.strip()


def test_suspicious_absence_does_not_override_proceed_or_score_at_point_six() -> None:
    technical = [_event(source=Source.DECK, claim="distributed execution")]
    proceed = gate.decide(COMPANY_ID, _score(mu=0.75, band=0.15), technical, T0)
    proof_case = gate.decide(COMPANY_ID, _score(mu=0.60, band=0.20), technical, T0)
    assert proceed.outcome is GateOutcome.PROCEED
    assert proof_case.outcome is GateOutcome.PROOF_PROTOCOL
    assert proceed.absence_is_suspicious is True
    assert proof_case.absence_is_suspicious is True


def test_github_evidence_prevents_suspicious_absence() -> None:
    events = [
        _event(source=Source.DECK, claim="compiler pipeline"),
        _event(source=Source.GITHUB, claim="implementation event", kind=EventKind.REPO_ACTIVITY),
    ]
    decision = gate.decide(COMPANY_ID, _score(mu=0.55, band=0.20), events, T0)
    assert decision.outcome is GateOutcome.PROOF_PROTOCOL
    assert decision.absence_is_suspicious is False


def test_non_code_or_foreign_events_cannot_clear_suspicious_absence() -> None:
    events = [
        _event(source=Source.DECK, claim="compiler pipeline"),
        _event(source=Source.GITHUB, claim="not an artifact"),
        _event(
            source=Source.GITHUB,
            kind=EventKind.REPO_ACTIVITY,
            company_id=UUID("30000000-0000-0000-0000-000000000000"),
        ),
    ]
    decision = gate.decide(COMPANY_ID, _score(mu=0.55, band=0.20), events, T0)
    assert decision.outcome is GateOutcome.NO_CALL
    assert decision.absence_is_suspicious is True


def test_failed_proof_artifact_does_not_clear_suspicious_absence() -> None:
    failed_proof = _event(
        source=Source.PROOF_PROTOCOL,
        kind=EventKind.PROOF_ARTIFACT,
        claim="",
    )
    failed_proof.payload = {"artifact": "submission", "works": False, "sound": True}
    decision = gate.decide(
        COMPANY_ID,
        _score(mu=0.55, band=0.20),
        [_event(source=Source.DECK, claim="compiler pipeline"), failed_proof],
        T0,
    )
    assert decision.outcome is GateOutcome.NO_CALL
    assert decision.absence_is_suspicious is True


def test_irrelevant_source_absence_never_penalizes() -> None:
    decision = gate.decide(
        COMPANY_ID,
        _score(mu=0.55, band=0.20),
        [_event(source=Source.DECK, claim="customer workflow design")],
        T0,
    )
    assert decision.outcome is GateOutcome.PROOF_PROTOCOL
    assert decision.absence_is_suspicious is False


def test_future_and_integrity_marked_events_are_ignored() -> None:
    events = [
        _event(
            source=Source.DECK,
            claim="runtime engine",
            observed_at=T0 + timedelta(seconds=1),
        ),
        _event(
            source=Source.DECK,
            claim="scheduler engine",
            integrity_flags=["injection_stripped"],
        ),
    ]
    decision = gate.decide(COMPANY_ID, _score(mu=0.55, band=0.20), events, T0)
    assert decision.outcome is GateOutcome.PROOF_PROTOCOL
    assert decision.absence_is_suspicious is False


def test_evaluate_scopes_store_and_refuses_ambiguous_founder_identity(monkeypatch) -> None:
    events = [
        _event(source=Source.DECK, entity_id=HIGH_ENTITY),
        _event(source=Source.DECK, entity_id=LOW_ENTITY),
    ]
    store_calls: list[dict] = []
    founder_calls: list[tuple[UUID, datetime]] = []

    def fake_events(**kwargs):
        store_calls.append(kwargs)
        return events

    def fake_founder(entity_id, as_of):
        founder_calls.append((entity_id, as_of))
        return _score(mu=0.7, band=0.2, entity_id=entity_id)

    monkeypatch.setattr("memory.store.events", fake_events)
    monkeypatch.setattr("memory.score.founder", fake_founder)

    result = gate.evaluate(COMPANY_ID, T0)

    assert store_calls == [{"company_id": COMPANY_ID, "as_of": T0}]
    assert founder_calls == []
    assert result.outcome is GateOutcome.PROOF_PROTOCOL
