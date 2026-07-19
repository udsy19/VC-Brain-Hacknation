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

WHICH QUESTION THIS GATE ASKS
-----------------------------
The third delicate part, and the reason this module reads `data/seed/thesis.json`.

This gate used to decide against `data/seed/backtest.json`'s threshold of 0.62. That
number answers **"would our score have DETECTED this founder before they broke out?"**
— it is calibrated on Docker, Hugging Face, Supabase and Vercel at their pre-breakout
truncation dates, founders with a year or more of sustained public build history.

This gate asks a different question: **"should this fund proceed on this pre-seed
company?"** Requiring Docker-at-truncation evidence from a seed-stage founder is a
category error, and it produced the obvious symptom: abstention on essentially
everyone, so the recommendation stage downstream could never fire.

So both knobs now come from the fund's own stated policy, and neither is borrowed from
the backtest:

  BAND CEILING    ``core.thesis.evidence_bar()`` — how much certainty this fund demands
                  before it will hold an opinion at all. Derived from `risk_appetite`.
                  The old hardcoded 0.20 was, exactly, this function's value at the
                  neutral appetite of 0.5; the constant was always a thesis parameter
                  that had not been wired up.
  CLEARING SCORE  ``thesis.json::clearing_score.value`` — the founder-axis level this
                  fund will write a cheque against, with its derivation stated in that
                  file. NOT the backtest's 0.62, which stays where it is because the
                  backtest's H12 fame check is a detection test.

The backtest keeps the detection threshold. Two thresholds, each named for the question
it answers, neither standing in for the other.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from core import thesis as thesis_mod
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


# Applied when thesis.json declares no `clearing_score`. Equal to the shipped thesis's
# value so a missing field does not silently re-open the gate; see that file for the
# derivation. A default here rather than in core/thesis.py's DEFAULTS because this is the
# only consumer, and the number means nothing outside a decision.
DEFAULT_CLEARING_SCORE = 0.55

# The absence classifier's own veto boundary, and deliberately NOT one of the two thesis
# parameters above. "A central technical claim with no artifact anywhere" is a statement
# about the integrity of the evidence, not about how much risk this fund tolerates — a
# bolder thesis buys thinner evidence, it does not buy unsupported claims. It is checked
# BEFORE the base-ladder proceed rule: once the clearing score can sit below this line,
# rule order is the only thing stopping a suspicious company from clearing on score alone.
SUSPICIOUS_ABSENCE_FLOOR = 0.60


def _claim_text(event: Event) -> str:
    value = event.payload.get("claim") or event.evidence_span or ""
    return str(value).lower()


def clearing_score(thesis: dict | None = None) -> float:
    """The founder-axis level this fund writes a cheque against. Fund policy, not a measurement.

    Read from `thesis.json::clearing_score.value`. This is the INVESTMENT threshold; the
    DETECTION threshold lives in `data/seed/backtest.json` and is not interchangeable with
    it. See this module's docstring and the rationale in the thesis file.
    """
    t = thesis or thesis_mod.load()
    cs = t.get("clearing_score")
    if isinstance(cs, dict):
        v = cs.get("value")
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else (
            DEFAULT_CLEARING_SCORE
        )
    if isinstance(cs, (int, float)) and not isinstance(cs, bool):
        return float(cs)
    return DEFAULT_CLEARING_SCORE


def decide(
    company_id: UUID,
    founder_score: FounderScore,
    events: list[Event],
    as_of: datetime,
    *,
    calibration: ConformalCalibration | None = None,
    thesis: dict | None = None,
) -> GateDecision:
    """Pure decision policy. Confidence remains explicit through the founder band.

    ``calibration`` is optional and defaults to off: with no calibration the base ladder
    runs on the thesis's own two parameters. Supply one and the conformal interval governs
    the abstention boundary instead, with its own reasoning attached.

    ``thesis`` is read fresh per call rather than memoised, because `PUT /thesis` edits the
    file at runtime and a cached evidence bar would make the control panel a picture again.
    """
    t = thesis if thesis is not None else thesis_mod.load()
    bar = thesis_mod.evidence_bar(t)
    clears_at = clearing_score(t)
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

    # The fund's evidence bar is a PRECONDITION on holding an opinion at all, so it is
    # checked before the conformal verdict rather than alongside it. A band wider than the
    # bar is not "we cannot tell which side of the line this falls on" — it is "we have not
    # gathered enough to have a view", and the system already has a name for that. Routing
    # it to NO_CALL would report a tie where there was never a contest. This is also what
    # keeps a cold-start founder (almost no evidence, band near the 0.5 prior) out of the
    # investable set no matter where the clearing score sits.
    if founder_score.band > bar:
        return _decision(
            GateOutcome.PROOF_PROTOCOL,
            f"Uncertainty ({founder_score.band:.2f}) is wider than this thesis's evidence "
            f"bar ({bar:.2f}); not enough evidence to hold a view. Create a targeted proof.",
        )

    if interval is not None:
        # The conformal boundary supersedes the base ladder for exactly two calls: abstain
        # when the interval straddles the threshold, proceed when it clears it outright.
        # Everything below the threshold still runs the base ladder, so PROOF_PROTOCOL —
        # "promising but thin, go get evidence" — keeps its meaning. The threshold it is
        # handed is now the INVESTMENT one; the layer itself is untouched.
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

    if founder_score.mu + founder_score.band < clears_at:
        return _decision(
            GateOutcome.NO_CALL,
            f"Even the upper confidence bound remains below this thesis's clearing score "
            f"({clears_at:.2f}).",
        )
    # Ordered ahead of the proceed rule on purpose. The clearing score is now fund policy
    # and may legitimately sit below this floor, at which point the old ordering would have
    # let a company proceed on a technical claim with no artifact behind it anywhere.
    if suspicious_absence and founder_score.mu < SUSPICIOUS_ABSENCE_FLOOR:
        return _decision(
            GateOutcome.NO_CALL,
            "A central technical claim lacks the directly relevant artifact evidence.",
        )
    if founder_score.mu >= clears_at:
        return _decision(
            GateOutcome.PROCEED,
            f"Capability evidence clears this thesis's score ({clears_at:.2f}) and "
            f"uncertainty is inside its evidence bar ({bar:.2f}).",
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

    The calibration is anchored on the INVESTMENT threshold, not the backtest's detection
    threshold. The cohort is still the labelled one — those are the only outcomes we have —
    but the nonconformity score measures distance from the line we are actually deciding
    against, which is the whole point of handing conformal a threshold rather than letting
    it read one.
    """
    from memory import score, store

    t = thesis_mod.load()
    calibration = conformal.from_store(
        as_of, alpha=alpha, threshold=clearing_score(t)
    ).for_company(company_id)
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
