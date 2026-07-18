"""Two cross-module failures that no single owner could see.

Both are silent: nothing errors, the numbers just quietly become untrustworthy.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timezone

from api import attest, memo
from core import pipeline
from schema.events import Event, EventKind, Source

T0 = datetime(2024, 6, 1, tzinfo=timezone.utc)


# --- 1. as_of must reach the validator ---------------------------------------


def test_memo_threads_as_of_into_the_validator() -> None:
    """A memo generated at a historical cutoff must not validate against today.

    check_claims defaults as_of to now(), so omitting the argument is a lookahead
    leak that raises nothing and shows up only as a backtest that looks too good.
    """
    src = inspect.getsource(memo._verdicts)
    assert "as_of" in inspect.signature(memo._verdicts).parameters
    assert "check_claims(company_id, as_of)" in src

    caller = inspect.getsource(memo.generate_memo)
    assert "_verdicts(cid, as_of)" in caller, "generate_memo must pass its cutoff through"


def test_pipeline_threads_as_of_into_the_validator() -> None:
    assert "as_of" in inspect.signature(pipeline._check_claims).parameters
    assert "check_claims(company_id, as_of)" in inspect.getsource(pipeline._check_claims)


def test_pipeline_does_not_duplicate_validator_events() -> None:
    """The validator persists its own VALIDATION_RESULT, stamped when the evidence
    existed. Writing a second copy at the run cutoff both duplicated the verdict and
    dated it wrongly."""
    src = inspect.getsource(pipeline.derive)
    assert "EventKind.VALIDATION_RESULT" not in src


# --- 2. the proof trace must be attested --------------------------------------


def _events(confidence: float = 0.8) -> list[Event]:
    return [
        Event(
            kind=EventKind.PROOF_BEHAVIOR,
            source=Source.PROOF_PROTOCOL,
            observed_at=T0,
            confidence=confidence,
            payload={"value": 0.9, "y": 0.9},
        )
    ]


def test_self_reported_pushback_is_not_counted_as_observed() -> None:
    """The whole point: asserting you pushed back must not buy the same weight as
    having been observed doing it."""
    attest.reset()
    trace = {"pushed_back_on_constraint": True, "questions_asked": ["is 32 fixed?"]}
    _, att = attest.attest("unknown-challenge", trace)

    assert att["challenge_anchored"] is False
    assert "pushed_back_on_constraint" in att["self_reported_fields"]
    assert "pushed_back_on_constraint" not in att["attested_fields"]
    assert att["trust"] < 0.6


def test_server_anchored_submission_scores_higher_trust() -> None:
    attest.reset()
    attest.record_issue("ch-1", T0)
    _, anchored = attest.attest("ch-1", {"pushed_back_on_constraint": True})
    _, floating = attest.attest("ch-2", {"pushed_back_on_constraint": True})

    assert anchored["challenge_anchored"] is True
    assert "started_at" in anchored["attested_fields"]
    assert anchored["trust"] > floating["trust"]


def test_client_supplied_timestamps_are_overwritten_by_server_observation() -> None:
    """A submitter claiming they finished in 90 seconds does not get to say so."""
    attest.reset()
    attest.record_issue("ch-1", T0)
    merged, _ = attest.attest(
        "ch-1", {"started_at": "2099-01-01T00:00:00+00:00", "submitted_at": "2099-01-01T00:01:00+00:00"}
    )
    assert merged["started_at"] == T0.isoformat()
    assert not merged["submitted_at"].startswith("2099")
    assert merged["elapsed_seconds"] > 0


def test_attestation_scales_event_confidence_down() -> None:
    """Trust multiplies confidence rather than sitting beside it — the filter treats
    PROOF_* events as low-noise, and an unattested trace must not buy that weight."""
    attest.reset()
    _, att = attest.attest("nope", {"pushed_back_on_constraint": True})
    out = attest.apply(_events(0.8), att)
    assert out[0].confidence < 0.8
    assert "unattested_trace" in out[0].integrity_flags
    assert out[0].payload["attestation"]["trust"] == att["trust"]


def test_demo_path_is_labelled_not_scored_as_real() -> None:
    attest.reset()
    _, att = attest.attest("ch", {}, demo=True)
    assert att["demo_seeded"] is True
    assert "pre-recorded" in att["note"]
