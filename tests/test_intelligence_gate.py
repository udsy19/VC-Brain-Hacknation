"""RED-phase tests for the locked pure and store-backed gate contract.

The threshold cases below are deliberately written against the THESIS's parameters rather
than against literal numbers. The gate's two boundaries — the evidence bar and the clearing
score — are fund policy now, so a test that hardcoded 0.70/0.20 would be asserting that the
fund may never change its mind, which is the opposite of what `thesis.json` is for. What is
actually invariant is the SHAPE: clearing the score with uncertainty inside the bar
proceeds, missing either does not.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from core import thesis as thesis_mod
from intelligence import gate
from schema.events import Event, EventKind, FounderScore, GateOutcome, Source

THESIS = thesis_mod.load()
BAR = thesis_mod.evidence_bar(THESIS)  # band ceiling — how much certainty we demand
CLEARS = gate.clearing_score(THESIS)  # founder-axis level we write a cheque against
NARROW = BAR - 0.05  # comfortably inside the bar
MIDDLING = CLEARS - 0.05  # promising, but short of the clearing score

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
        # Clears the score, uncertainty inside the bar.
        (CLEARS, NARROW, GateOutcome.PROCEED),
        (CLEARS + 0.10, 0.05, GateOutcome.PROCEED),
        # Even the optimistic bound falls short of the clearing score.
        (0.20, 0.20, GateOutcome.NO_CALL),
        (CLEARS - 0.11, 0.00, GateOutcome.NO_CALL),
        # Promising but under the score: go and get a proof.
        (MIDDLING, NARROW, GateOutcome.PROOF_PROTOCOL),
        (CLEARS - 0.01, NARROW, GateOutcome.PROOF_PROTOCOL),
        # Clears the score, but uncertainty is wider than this fund's evidence bar.
        (CLEARS, BAR + 0.01, GateOutcome.PROOF_PROTOCOL),
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


def test_suspicious_absence_does_not_override_a_clearly_strong_score() -> None:
    """Above the absence floor a strong score still proceeds, with the flag recorded."""
    technical = [_event(source=Source.DECK, claim="distributed execution")]
    proceed = gate.decide(COMPANY_ID, _score(mu=0.75, band=0.15), technical, T0)
    assert proceed.outcome is GateOutcome.PROCEED
    assert proceed.absence_is_suspicious is True


def test_suspicious_absence_is_checked_before_the_proceed_rule() -> None:
    """The absence veto outranks the clearing score, not the other way round.

    The clearing score is fund policy and may sit BELOW the absence floor — under the
    shipped thesis it does (0.55 vs 0.60). A company in that window clears the fund's
    score while its central technical claim has no artifact behind it anywhere. Rule order
    is the only thing that stops it proceeding, so this pins the order rather than the
    numbers: whatever the thesis says, a suspicious company under the floor never proceeds.
    """
    floor = gate.SUSPICIOUS_ABSENCE_FLOOR
    mu = (CLEARS + floor) / 2
    if not CLEARS < floor:
        pytest.skip("this thesis puts the clearing score at or above the absence floor")
    assert CLEARS <= mu < floor
    decision = gate.decide(
        COMPANY_ID,
        _score(mu=mu, band=NARROW),
        [_event(source=Source.DECK, claim="distributed execution")],
        T0,
    )
    assert decision.outcome is GateOutcome.NO_CALL
    assert decision.absence_is_suspicious is True


def test_github_evidence_prevents_suspicious_absence() -> None:
    events = [
        _event(source=Source.DECK, claim="compiler pipeline"),
        _event(source=Source.GITHUB, claim="implementation event", kind=EventKind.REPO_ACTIVITY),
    ]
    decision = gate.decide(COMPANY_ID, _score(mu=MIDDLING, band=NARROW), events, T0)
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
    decision = gate.decide(COMPANY_ID, _score(mu=MIDDLING, band=NARROW), events, T0)
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
        _score(mu=MIDDLING, band=NARROW),
        [_event(source=Source.DECK, claim="compiler pipeline"), failed_proof],
        T0,
    )
    assert decision.outcome is GateOutcome.NO_CALL
    assert decision.absence_is_suspicious is True


def test_irrelevant_source_absence_never_penalizes() -> None:
    decision = gate.decide(
        COMPANY_ID,
        _score(mu=MIDDLING, band=NARROW),
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
    decision = gate.decide(COMPANY_ID, _score(mu=MIDDLING, band=NARROW), events, T0)
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


# ---------------------------------------------------------------------------
# The thesis drives both boundaries. See intelligence/gate.py's docstring for why
# the detection threshold and the investment threshold are not interchangeable.
# ---------------------------------------------------------------------------


def _thesis(*, appetite: float, clears: float | None = None) -> dict:
    t = {**THESIS, "risk_appetite": {"value": appetite}}
    if clears is not None:
        t["clearing_score"] = {"value": clears}
    return t


def test_evidence_bar_is_the_band_ceiling_and_a_bolder_fund_tolerates_more_uncertainty() -> None:
    """The band ceiling comes from risk appetite, not from a constant in the gate."""
    cautious, bold = _thesis(appetite=0.0), _thesis(appetite=1.0)
    # A band that a bold fund tolerates and a cautious one does not.
    band = (thesis_mod.evidence_bar(cautious) + thesis_mod.evidence_bar(bold)) / 2
    score = _score(mu=0.90, band=band)
    assert gate.decide(COMPANY_ID, score, [], T0, thesis=bold).outcome is GateOutcome.PROCEED
    assert (
        gate.decide(COMPANY_ID, score, [], T0, thesis=cautious).outcome
        is GateOutcome.PROOF_PROTOCOL
    )


def test_uncertainty_wider_than_the_bar_is_proof_protocol_not_no_call() -> None:
    """Not enough evidence to hold a view is a different answer from a genuine tie.

    This is what keeps a cold-start founder out of the investable set regardless of where
    the clearing score sits: with almost no evidence the band stays near the 0.5 prior,
    which no plausible evidence bar admits.
    """
    bar = thesis_mod.evidence_bar(THESIS)
    for mu in (0.20, 0.50, 0.95):
        decision = gate.decide(COMPANY_ID, _score(mu=mu, band=bar + 0.10), [], T0)
        assert decision.outcome is GateOutcome.PROOF_PROTOCOL
        assert "evidence bar" in decision.rationale


def test_clearing_score_is_read_from_the_thesis_not_from_the_backtest() -> None:
    """Editing the thesis moves the decision; the backtest's 0.62 does not appear here."""
    score = _score(mu=0.60, band=NARROW)
    strict = _thesis(appetite=thesis_mod.risk_appetite(THESIS), clears=0.80)
    loose = _thesis(appetite=thesis_mod.risk_appetite(THESIS), clears=0.50)
    assert gate.decide(COMPANY_ID, score, [], T0, thesis=loose).outcome is GateOutcome.PROCEED
    assert gate.decide(COMPANY_ID, score, [], T0, thesis=strict).outcome is not GateOutcome.PROCEED


def test_clearing_score_falls_back_explicitly_when_the_thesis_omits_it() -> None:
    assert gate.clearing_score({}) == gate.DEFAULT_CLEARING_SCORE
    assert gate.clearing_score({"clearing_score": "nonsense"}) == gate.DEFAULT_CLEARING_SCORE
    assert gate.clearing_score({"clearing_score": {"value": 0.42}}) == 0.42
    assert gate.clearing_score({"clearing_score": 0.42}) == 0.42


def test_shipped_thesis_clearing_score_is_not_the_backtest_detection_threshold() -> None:
    """The two answer different questions, so borrowing one for the other is the bug."""
    from intelligence import conformal

    assert gate.clearing_score(THESIS) != conformal.clearing_threshold()
