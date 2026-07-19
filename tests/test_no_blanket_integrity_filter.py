"""The blanket-integrity-filter regression guard.

This bug has now shipped twice. `intelligence/flags.py` defines IMPEACHING_FLAGS to
separate integrity flags that mean "the content was TAMPERED WITH" from those that
merely ANNOTATE provenance (transliterated_name, non_english_source, date_inferred,
ocr_low_conf). The fix was applied in flags.py and nowhere else, so seven other
evidence filters kept testing `not event.integrity_flags` — voiding the entire
non-Latin-script cohort.

The failure is INVISIBLE when it happens: the founders do not error, they score at
the untouched prior, which reads as "average founder" rather than "we could not read
this". Nothing but a test will catch it. Hence two independent guards below:

  1. a structural guard — no evidence module may test `.integrity_flags` for
     truthiness at all; the only sanctioned form is `flags.is_impeached(event)`
  2. behavioural guards — provenance-flagged evidence must survive each public
     entry point, and tampered evidence must not
"""

from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from intelligence import dissent, flags, gate, validator
from schema.events import (
    Axis,
    ClaimStatus,
    Event,
    EventKind,
    FounderScore,
    ScreeningResult,
    Source,
)

ROOT = Path(__file__).resolve().parent.parent

# Modules that decide which evidence counts. These are the filters the fix must hold
# across; a new one added here without is_impeached() is exactly the regression.
GUARDED = [*sorted((ROOT / "intelligence").glob("*.py")), ROOT / "api" / "memo.py"]

# Flags that annotate WHERE evidence came from. None of these may ever disqualify it.
PROVENANCE_FLAGS = ["transliterated_name", "non_english_source", "date_inferred", "ocr_low_conf"]

T0 = datetime(2024, 6, 1, tzinfo=timezone.utc)
COMPANY_ID = UUID("10000000-0000-0000-0000-000000000000")
ENTITY_ID = UUID("20000000-0000-0000-0000-000000000000")


# ---------------------------------------------------------------------------
# 1. Structural guard
# ---------------------------------------------------------------------------


def _boolean_contexts(tree: ast.AST):
    """Every expression the interpreter will evaluate for truthiness."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp, ast.While, ast.Assert)):
            yield node.test
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            yield node.operand
        elif isinstance(node, ast.BoolOp):
            yield from node.values
        elif isinstance(node, (ast.comprehension,)):
            yield from node.ifs


def _blanket_tests(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sorted(
        node.lineno
        for node in _boolean_contexts(tree)
        if isinstance(node, ast.Attribute) and node.attr == "integrity_flags"
    )


@pytest.mark.parametrize("path", GUARDED, ids=lambda p: p.name)
def test_no_module_tests_integrity_flags_for_truthiness(path: Path) -> None:
    """`not event.integrity_flags` treats a provenance NOTE as a disqualification.

    Use `flags.is_impeached(event)`. If you genuinely need to exclude tampered
    content, that IS the function — it already means exactly that.
    """
    offenders = _blanket_tests(path)
    assert not offenders, (
        f"{path.relative_to(ROOT)} tests .integrity_flags for truthiness at line(s) "
        f"{offenders}. This voids provenance-flagged evidence — the entire Type 6 "
        f"cohort. Use flags.is_impeached(event) instead."
    )


def test_the_guard_actually_detects_the_blanket_form() -> None:
    """A guard that cannot fail protects nothing."""
    src = "x = [e for e in events if e.kind != K and not e.integrity_flags]"
    assert _blanket_tests_from_source(src), "guard failed to flag the comprehension form"
    assert _blanket_tests_from_source("if claim.integrity_flags:\n    pass\n"), (
        "guard failed to flag the if-statement form"
    )
    assert not _blanket_tests_from_source("x = [e for e in events if not is_impeached(e)]")


def _blanket_tests_from_source(src: str) -> list[int]:
    tree = ast.parse(src)
    return [
        n.lineno
        for n in _boolean_contexts(tree)
        if isinstance(n, ast.Attribute) and n.attr == "integrity_flags"
    ]


# ---------------------------------------------------------------------------
# 2. Behavioural guards
# ---------------------------------------------------------------------------


def test_provenance_flags_are_not_impeaching_and_tampering_is() -> None:
    for flag in PROVENANCE_FLAGS:
        assert not flags.is_impeached(_event(integrity_flags=[flag])), (
            f"{flag} annotates provenance; it must never disqualify evidence"
        )
    for flag in sorted(flags.IMPEACHING_FLAGS):
        assert flags.is_impeached(_event(integrity_flags=[flag])), (
            f"{flag} means the content was tampered with; it must disqualify evidence"
        )


def _event(
    *,
    kind: EventKind = EventKind.REPO_ACTIVITY,
    source: Source = Source.GITHUB,
    integrity_flags: list[str] | None = None,
    observed_at: datetime = T0,
    payload: dict | None = None,
    text: str = "marker",
) -> Event:
    return Event(
        event_id=uuid4(),
        entity_id=ENTITY_ID,
        company_id=COMPANY_ID,
        kind=kind,
        source=source,
        source_url="https://example.com/x",
        observed_at=observed_at,
        payload=payload if payload is not None else {"text": text},
        evidence_span=text,
        confidence=0.9,
        integrity_flags=integrity_flags or [],
    )


@pytest.mark.parametrize("flag", PROVENANCE_FLAGS)
def test_flags_evaluate_events_keeps_provenance_flagged_evidence(flag: str) -> None:
    """The measured regression: 100% of the non-Latin-script cohort's events carry
    transliterated_name, so a blanket filter left all three founders at zero flags."""
    events = [
        _event(integrity_flags=[flag], observed_at=T0 - timedelta(days=d), text=f"commit {d}")
        for d in range(6)
    ]
    out = flags.evaluate_events(events, entity_id=ENTITY_ID, as_of=T0)
    assert out, f"{flag} voided the entire evidence base"

    clean = flags.evaluate_events(
        [_event(observed_at=e.observed_at, text="commit") for e in events],
        entity_id=ENTITY_ID,
        as_of=T0,
    )
    assert len(out) == len(clean), f"{flag} changed which rules were evaluated"


def test_flags_evaluate_events_still_drops_tampered_evidence() -> None:
    events = [
        _event(integrity_flags=["injection_stripped"], observed_at=T0 - timedelta(days=d))
        for d in range(6)
    ]
    assert flags.evaluate_events(events, entity_id=ENTITY_ID, as_of=T0) == []


@pytest.mark.parametrize("flag", PROVENANCE_FLAGS)
def test_validator_still_attempts_a_provenance_flagged_claim(flag: str) -> None:
    claim = _event(kind=EventKind.DECK_CLAIM, source=Source.DECK, integrity_flags=[flag])
    verdict = validator.check_claim(claim, [])
    assert verdict.status is ClaimStatus.UNVERIFIABLE, (
        f"{flag} caused the validator to refuse to even look at the claim"
    )


def test_validator_refuses_a_tampered_claim() -> None:
    claim = _event(
        kind=EventKind.DECK_CLAIM, source=Source.DECK, integrity_flags=["injection_stripped"]
    )
    assert validator.check_claim(claim, []).status is ClaimStatus.NOT_ATTEMPTED


@pytest.mark.parametrize("flag", PROVENANCE_FLAGS)
def test_dissent_sees_provenance_flagged_evidence(flag: str) -> None:
    """The anti-memo must argue from the same evidence graph as the memo. A blanket
    filter here made the bear case blind to ~20-25% of the Type 6 evidence."""
    seen: dict = {}

    def judge(prompt, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after capturing the packet")

    dissent.generate_from_evidence(
        COMPANY_ID,
        T0,
        [_event(integrity_flags=[flag], text="flagged marker")],
        _screening(),
        judge,
    )
    assert "flagged marker" in seen.get("untrusted", ""), (
        f"{flag} hid evidence from the bear case that the memo keeps"
    )


def test_dissent_excludes_tampered_evidence() -> None:
    seen: dict = {}

    def judge(prompt, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop")

    dissent.generate_from_evidence(
        COMPANY_ID,
        T0,
        [_event(integrity_flags=["injection_stripped"], text="tampered marker")],
        _screening(),
        judge,
    )
    assert "tampered marker" not in seen.get("untrusted", "")


@pytest.mark.parametrize("flag", PROVENANCE_FLAGS)
def test_gate_counts_provenance_flagged_evidence(flag: str) -> None:
    events = [
        _event(kind=EventKind.DECK_CLAIM, source=Source.DECK, integrity_flags=[flag]),
        _event(integrity_flags=[flag]),
    ]
    flagged = gate.decide(COMPANY_ID, _score(), events, T0)
    clean = gate.decide(
        COMPANY_ID, _score(), [_event(kind=e.kind, source=e.source) for e in events], T0
    )
    assert flagged.outcome is clean.outcome, (
        f"{flag} changed the gate outcome; provenance must not move the decision"
    )
    assert flagged.absence_is_suspicious == clean.absence_is_suspicious


def _score(mu: float = 0.55, band: float = 0.20) -> FounderScore:
    return FounderScore(
        entity_id=ENTITY_ID,
        mu=mu,
        band=band,
        trend=0.0,
        as_of=T0,
        contributing_event_ids=[],
    )


def _axis(score: float = 0.5) -> Axis:
    return Axis(score=score, trend=0.0, confidence=0.5, evidence_event_ids=[])


def _screening() -> ScreeningResult:
    return ScreeningResult(
        company_id=COMPANY_ID,
        as_of=T0,
        founder=_axis(),
        market=_axis(),
        idea_vs_market=_axis(),
    )
