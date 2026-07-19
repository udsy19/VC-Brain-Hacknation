"""Conformal no-call thresholding: does it calibrate, does it refuse, does it discriminate.

These tests are deliberately behavioural. The failure mode this layer exists to avoid is
code that looks implemented and measures nothing, so the assertions here are about the
DISTRIBUTION of outcomes over a corpus — a layer that abstains on everything, or on
nothing, fails these — and about every degenerate case returning "not calibrated" rather
than a confident-looking interval it did not earn.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from intelligence import conformal, gate
from schema.events import Event, EventKind, FounderScore, GateOutcome, Source

T0 = datetime(2025, 4, 5, tzinfo=timezone.utc)
COMPANY_ID = UUID("10000000-0000-0000-0000-000000000000")
TAU = 0.62


def _point(mu: float, band: float, cleared: bool, *, company_id: UUID | None = None):
    return conformal.CalibrationPoint(
        label=f"{'w' if cleared else 'c'}-{mu}",
        company_id=company_id,
        mu=mu,
        band=band,
        cleared=cleared,
        nonconformity=conformal.nonconformity(mu, band, cleared, TAU),
    )


def _cohort() -> list[conformal.CalibrationPoint]:
    """Nine points shaped like the real backtest cohort: 4 cleared high, 5 not-cleared low."""
    return [
        _point(0.80, 0.275, True),
        _point(0.79, 0.275, True),
        _point(0.79, 0.273, True),
        _point(0.78, 0.275, True),
        _point(0.28, 0.266, False),
        _point(0.26, 0.265, False),
        _point(0.22, 0.265, False),
        _point(0.22, 0.265, False),
        _point(0.22, 0.265, False),
    ]


def _score(mu: float, band: float) -> FounderScore:
    return FounderScore(entity_id=uuid4(), as_of=T0, mu=mu, band=band, trend=0.0)


# --------------------------------------------------------------------------------------
# The mechanism
# --------------------------------------------------------------------------------------


def test_alpha_sets_the_minimum_calibration_size() -> None:
    """The quantile index must land inside the sample. That is what bounds alpha."""
    assert conformal.required_points(0.10) == 9
    assert conformal.required_points(0.125) == 7
    assert conformal.required_points(0.05) == 19


def test_nonconformity_is_wrong_side_band_widths() -> None:
    """A company that cleared while scoring above tau is negative: no wrong-side margin."""
    assert conformal.nonconformity(0.80, 0.20, True, TAU) == pytest.approx(-0.9)
    assert conformal.nonconformity(0.20, 0.20, False, TAU) == pytest.approx(-2.1)
    # The score got it backwards: cleared, but scored below tau. Positive nonconformity.
    assert conformal.nonconformity(0.40, 0.20, True, TAU) == pytest.approx(1.1)


def test_calibrates_on_a_cohort_shaped_set_and_reports_its_parts() -> None:
    cal = conformal.calibrate(_cohort(), alpha=0.125, threshold=TAU)
    assert cal.calibrated is True
    assert cal.n == 9 and cal.n_cleared == 4 and cal.n_not_cleared == 5
    assert cal.alpha == 0.125
    assert cal.z is not None and cal.z > 0
    # z is the widest wrong-side margin at the quantile index, not a constant someone typed.
    residuals = sorted(p.nonconformity for p in _cohort())
    assert cal.z == pytest.approx(-residuals[8])


def test_interval_is_centred_on_mu_and_scaled_by_band() -> None:
    cal = conformal.calibrate(_cohort(), alpha=0.125, threshold=TAU)
    tight = cal.interval(0.55, 0.02)
    wide = cal.interval(0.55, 0.40)
    assert tight is not None and wide is not None
    assert (tight.upper - tight.lower) < (wide.upper - wide.lower)
    # Same mu, different evidence quality: the tight one resolves, the wide one abstains.
    assert tight.verdict == "does_not_clear"
    assert wide.verdict == "ambiguous"


def test_verdicts_split_on_the_threshold() -> None:
    cal = conformal.calibrate(_cohort(), alpha=0.125, threshold=TAU)
    assert cal.interval(0.95, 0.05).verdict == "clears"
    assert cal.interval(0.10, 0.05).verdict == "does_not_clear"
    assert cal.interval(0.62, 0.20).verdict == "ambiguous"


def test_rationale_states_alpha_the_size_and_the_interval() -> None:
    cal = conformal.calibrate(_cohort(), alpha=0.125, threshold=TAU)
    text = cal.describe(cal.interval(0.62, 0.20))
    assert "0.125" in text
    assert "n=9" in text
    assert "0.62" in text
    assert "9 labelled outcomes" in text  # the sample size travels with the guarantee


# --------------------------------------------------------------------------------------
# Refusal: every degenerate case must say "not calibrated", never fake an interval
# --------------------------------------------------------------------------------------


def test_too_few_points_is_not_calibrated() -> None:
    cal = conformal.calibrate(_cohort()[:5], alpha=0.10, threshold=TAU)
    assert cal.calibrated is False
    assert cal.z is None
    assert cal.interval(0.62, 0.05) is None
    assert "at least 9" in cal.reason and "only 5" in cal.reason


def test_single_class_is_not_calibrated() -> None:
    cal = conformal.calibrate(
        [_point(0.8 - i * 0.01, 0.2, True) for i in range(9)], alpha=0.125, threshold=TAU
    )
    assert cal.calibrated is False
    assert "one outcome class" in cal.reason


def test_calibration_that_cannot_separate_is_not_calibrated() -> None:
    """Scores on the wrong side of tau make z <= 0; the interval would never abstain."""
    scrambled = [
        _point(0.30, 0.20, True),  # cleared but scored low
        _point(0.90, 0.20, False),  # did not clear but scored high
        *_cohort()[:7],
    ]
    cal = conformal.calibrate(scrambled, alpha=0.125, threshold=TAU)
    assert cal.calibrated is False
    assert "degenerate quantile" in cal.reason
    assert cal.interval(0.62, 0.05) is None


def test_uncalibrated_describe_says_so_and_names_the_fallback() -> None:
    cal = conformal.calibrate([], alpha=0.125, threshold=TAU)
    text = cal.describe(None)
    assert "not calibrated" in text and "fell back" in text


def test_alpha_must_be_a_probability() -> None:
    with pytest.raises(ValueError):
        conformal.required_points(0.0)
    with pytest.raises(ValueError):
        conformal.required_points(1.0)


# --------------------------------------------------------------------------------------
# Leakage: nothing calibrates on the point it is judging
# --------------------------------------------------------------------------------------


def test_for_company_drops_the_company_under_evaluation() -> None:
    member = uuid4()
    points = _cohort()
    points[0] = _point(0.80, 0.275, True, company_id=member)
    cal = conformal.calibrate(points, alpha=0.125, threshold=TAU)
    held_out = cal.for_company(member)
    assert held_out.n == 8
    assert all(p.company_id != member for p in held_out.points)


def test_for_company_is_a_no_op_for_a_non_member() -> None:
    cal = conformal.calibrate(_cohort(), alpha=0.125, threshold=TAU)
    assert cal.for_company(uuid4()) is cal


def test_exclusion_that_breaks_alpha_falls_back_and_says_why() -> None:
    """At alpha=0.10 a member's own evaluation legitimately loses its guarantee."""
    member = uuid4()
    points = _cohort()
    points[0] = _point(0.80, 0.275, True, company_id=member)
    cal = conformal.calibrate(points, alpha=0.10, threshold=TAU)
    assert cal.calibrated is True
    held_out = cal.for_company(member)
    assert held_out.calibrated is False
    assert "calibrating on the point being judged" in held_out.reason


# --------------------------------------------------------------------------------------
# The gate: the interval governs abstention, and the fallback is never silent
# --------------------------------------------------------------------------------------


def test_gate_without_calibration_keeps_the_historical_policy() -> None:
    """Default-off. The constant-threshold ladder is untouched when no calibration is given."""
    assert gate.decide(COMPANY_ID, _score(0.75, 0.15), [], T0).outcome is GateOutcome.PROCEED
    assert gate.decide(COMPANY_ID, _score(0.20, 0.20), [], T0).outcome is GateOutcome.NO_CALL
    assert gate.decide(COMPANY_ID, _score(0.50, 0.20), [], T0).outcome is GateOutcome.PROOF_PROTOCOL


def test_gate_abstains_when_the_interval_straddles_and_explains() -> None:
    cal = conformal.calibrate(_cohort(), alpha=0.125, threshold=TAU)
    decision = gate.decide(COMPANY_ID, _score(0.64, 0.25), [], T0, calibration=cal)
    assert decision.outcome is GateOutcome.NO_CALL
    assert "straddles the threshold" in decision.rationale
    assert "alpha=0.125" in decision.rationale
    assert "n=9" in decision.rationale


def test_gate_does_not_abstain_when_the_interval_clears_outright() -> None:
    cal = conformal.calibrate(_cohort(), alpha=0.125, threshold=TAU)
    decision = gate.decide(COMPANY_ID, _score(0.90, 0.05), [], T0, calibration=cal)
    assert decision.outcome is GateOutcome.PROCEED
    assert "entirely above" in decision.rationale


def test_gate_below_the_threshold_still_runs_the_base_ladder() -> None:
    """Conformal owns abstention, not deprioritisation. PROOF_PROTOCOL keeps its meaning."""
    cal = conformal.calibrate(_cohort(), alpha=0.125, threshold=TAU)
    decision = gate.decide(COMPANY_ID, _score(0.50, 0.05), [], T0, calibration=cal)
    assert decision.outcome is GateOutcome.PROOF_PROTOCOL
    assert "entirely below" in decision.rationale


def test_gate_says_out_loud_when_the_conformal_layer_did_not_calibrate() -> None:
    uncalibrated = conformal.calibrate(_cohort()[:3], alpha=0.125, threshold=TAU)
    decision = gate.decide(COMPANY_ID, _score(0.50, 0.20), [], T0, calibration=uncalibrated)
    assert decision.outcome is GateOutcome.PROOF_PROTOCOL  # base policy, unchanged
    assert "not calibrated" in decision.rationale
    assert "fell back" in decision.rationale


def test_suspicious_absence_withholds_the_conformal_proceed() -> None:
    """The absence classifier is orthogonal: a clearing interval does not buy past it.

    The pre-existing ladder still owns what happens next (here: PROOF_PROTOCOL, because
    the band is too wide for its own PROCEED rule), which is the point — conformal adds a
    reason to abstain, it does not overrule the artifact-evidence check.
    """
    cal = conformal.calibrate(_cohort(), alpha=0.125, threshold=TAU)
    technical = [
        Event(
            company_id=COMPANY_ID,
            entity_id=uuid4(),
            kind=EventKind.DECK_CLAIM,
            source=Source.DECK,
            observed_at=T0,
            payload={"claim": "a distributed runtime"},
        )
    ]
    score = _score(0.79, 0.275)
    assert cal.interval(score.mu, score.band).verdict == "clears"
    decision = gate.decide(COMPANY_ID, score, technical, T0, calibration=cal)
    assert decision.absence_is_suspicious is True
    assert decision.outcome is GateOutcome.PROOF_PROTOCOL


# --------------------------------------------------------------------------------------
# Discrimination: the whole point. A layer that abstains on all or none is broken.
# --------------------------------------------------------------------------------------


def test_abstention_discriminates_across_a_score_sweep() -> None:
    """Sweep mu across [0, 1] at a realistic band; abstention must be a middle region."""
    cal = conformal.calibrate(_cohort(), alpha=0.125, threshold=TAU)
    verdicts = {round(mu, 2): cal.interval(mu, 0.15).verdict for mu in [i / 20 for i in range(21)]}
    ambiguous = [mu for mu, v in verdicts.items() if v == "ambiguous"]
    clears = [mu for mu, v in verdicts.items() if v == "clears"]
    denied = [mu for mu, v in verdicts.items() if v == "does_not_clear"]

    assert clears and denied and ambiguous, f"no discrimination: {verdicts}"
    # Not everything, and not nothing.
    assert 0 < len(ambiguous) < len(verdicts)
    # The ambiguous region is contiguous and brackets the threshold.
    assert min(ambiguous) < TAU < max(ambiguous)
    assert max(denied) < min(ambiguous) and min(clears) > max(ambiguous)


def test_wide_bands_abstain_where_tight_bands_decide() -> None:
    """Two companies at the same score: abstention tracks evidence quality, not just level."""
    cal = conformal.calibrate(_cohort(), alpha=0.125, threshold=TAU)
    assert cal.interval(0.70, 0.02).verdict == "clears"
    assert cal.interval(0.70, 0.50).verdict == "ambiguous"


def test_store_backed_calibration_is_not_calibrated_when_offline() -> None:
    """No cohort in the store means no guarantee. It must not invent one."""
    conformal.reset_cache()
    try:
        cal = conformal.from_store(T0)
        assert cal.calibrated is False
        assert cal.z is None
    finally:
        conformal.reset_cache()
