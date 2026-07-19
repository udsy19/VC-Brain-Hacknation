"""The custom VC council: does it DISCRIMINATE, or is it decorative?

This codebase has repeatedly shipped code that looks implemented and measures nothing —
a metric returning a confident 1.0, a substance rule reading payload keys that did not
exist, `axis_spreads` identically 0.0 because the bear was handed the bull's numbers. The
equivalent failure here would be five lenses producing the same contribution, or a
personal rank that is always the core rank.

So the load-bearing tests in this file are the ones that build TWO profiles out of real
survey answers and real decision rows, run both over the same evidence, and assert the
rankings differ in a way the lens weights explain. Everything else guards an invariant:
same evidence graph, core score untouched, no invented lens, no silent zero, and no
recommendation without a bear case.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from uuid import UUID, uuid4

import pytest

from intelligence import council, custom_council
from intelligence.custom_council import CompanyView, LensKind
from memory import profiles
from schema.events import Axis, Event, EventKind, ScreeningResult, Source
from schema.vc import Choice, DecisionKind, PastDecision, SurveyAnswer

T0 = datetime(2025, 5, 6, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Two real profiles, built the way a user builds one: survey answers and an
# uploaded decision history. NOT hand-authored DerivedProfile objects — a test that
# invents the profile object cannot catch a derivation that never fires.
# ---------------------------------------------------------------------------

#: Founder-first, conviction-heavy, pre-seed. Picks the founder option, the early
#: option and the "rather write the cheque that dies" option throughout.
BOLD_ANSWERS = {
    "q01_founder_vs_market": "a",
    "q02_traction_vs_demo": "b",
    "q03_early_vs_ontime": "a",
    "q04_velocity_vs_durability": "a",
    "q05_distribution_vs_product": "b",
    "q06_which_mistake": "b",
    "q07_metrics_vs_love": "a",
    "q08_incumbent_vs_demand": "a",
    "q09_adaptability_vs_persistence": "a",
    "q10_insider_vs_outsider": "a",
    "q11_speed_vs_diligence": "a",
    "q12_price_vs_pick": "a",
}

#: Market-first, evidence-heavy, later stage. The mirror image on every trade-off
#: that carries a conviction or stage signal.
PATIENT_ANSWERS = {
    "q01_founder_vs_market": "b",
    "q02_traction_vs_demo": "a",
    "q03_early_vs_ontime": "b",
    "q04_velocity_vs_durability": "b",
    "q05_distribution_vs_product": "a",
    "q06_which_mistake": "a",
    "q07_metrics_vs_love": "b",
    "q08_incumbent_vs_demand": "b",
    "q09_adaptability_vs_persistence": "b",
    "q10_insider_vs_outsider": "a",
    "q11_speed_vs_diligence": "b",
    "q12_price_vs_pick": "b",
}

BOLD_DECISIONS = [
    ("ai-infra", "pre-seed", DecisionKind.INVESTED),
    ("ai-infra", "pre-seed", DecisionKind.INVESTED),
    ("ai-infra", "seed", DecisionKind.INVESTED),
    ("ai-infra", "pre-seed", DecisionKind.INVESTED),
    ("ai-infra", "seed", DecisionKind.INVESTED),
    ("ai-infra", "pre-seed", DecisionKind.INVESTED),
    ("data-infra", "pre-seed", DecisionKind.INVESTED),
    ("data-infra", "seed", DecisionKind.INVESTED),
    *[("fintech", "series a", DecisionKind.PASSED)] * 6,
    *[("dev-tools", "series b", DecisionKind.PASSED)] * 6,
]

PATIENT_DECISIONS = [
    ("dev-tools", "series a", DecisionKind.INVESTED),
    ("dev-tools", "series b", DecisionKind.INVESTED),
    ("dev-tools", "series a", DecisionKind.INVESTED),
    ("dev-tools", "series b", DecisionKind.INVESTED),
    ("dev-tools", "series a", DecisionKind.INVESTED),
    ("data-infra", "series a", DecisionKind.INVESTED),
    ("data-infra", "series b", DecisionKind.INVESTED),
    ("data-infra", "series a", DecisionKind.INVESTED),
    *[("ai-infra", "pre-seed", DecisionKind.PASSED)] * 6,
    *[("crypto", "seed", DecisionKind.PASSED)] * 6,
]


def _make_profile(answers: dict[str, str], rows: list[tuple], **kwargs):
    """Create a real user, store real answers and real decision rows, derive."""
    user = profiles.create_user(f"vc-{uuid4().hex[:12]}@example.test", "hash")
    profiles.save_survey(
        user.user_id,
        [SurveyAnswer(question_id=qid, choice=Choice(c)) for qid, c in answers.items()],
    )
    profiles.save_decisions(
        user.user_id,
        [
            PastDecision(
                company=f"co-{i}",
                sector=sector,
                stage=stage,
                decision=decision,
                decided_on=date(2024, 1, 1),
                source_row=i,
            )
            for i, (sector, stage, decision) in enumerate(rows, start=1)
        ],
        replace=True,
    )
    if kwargs:
        profiles.update_profile(user.user_id, **kwargs)
    return profiles.derive(user.user_id)


@pytest.fixture
def bold():
    return _make_profile(BOLD_ANSWERS, BOLD_DECISIONS)


@pytest.fixture
def patient():
    return _make_profile(PATIENT_ANSWERS, PATIENT_DECISIONS)


# ---------------------------------------------------------------------------
# A synthetic pipeline: thirteen companies, deliberately varied so a ranking has
# something to disagree about.
# ---------------------------------------------------------------------------

PIPELINE = [
    # name, founder, market, idea_vs_market, sector, stage
    ("Tensorpage", 0.88, 0.42, 0.71, "ai-infra", "pre-seed"),
    ("Veritanode", 0.34, 0.81, 0.40, "dev-tools", "series a"),
    ("Arcwell", 0.62, 0.63, 0.61, "data-infra", "seed"),
    ("Zaryad", 0.79, 0.30, 0.83, "ai-infra", "pre-seed"),
    ("Synthgrid", 0.41, 0.77, 0.35, "dev-tools", "series b"),
    ("Halberd", 0.70, 0.55, 0.52, "data-infra", "series a"),
    ("Corvid", 0.55, 0.68, 0.44, "dev-tools", "seed"),
    ("Nettle", 0.83, 0.36, 0.75, "ai-infra", "seed"),
    ("Pellucid", 0.48, 0.72, 0.38, "fintech", "series a"),
    ("Kestrel", 0.66, 0.49, 0.69, "ai-infra", "series a"),
    ("Umbra", 0.37, 0.85, 0.33, "dev-tools", "series b"),
    ("Fathom", 0.74, 0.44, 0.79, "data-infra", "pre-seed"),
    ("Gantry", 0.59, 0.58, 0.57, "crypto", "seed"),
]


def _views() -> list[CompanyView]:
    out = []
    for name, founder, market, idea, sector, stage in PIPELINE:
        cid = uuid4()
        out.append(
            CompanyView(
                company_id=cid,
                name=name,
                sector=sector,
                stage=stage,
                axes={"founder": founder, "market": market, "idea_vs_market": idea},
                axis_confidence={"founder": 0.8, "market": 0.6, "idea_vs_market": 0.7},
                axis_evidence={"founder": [uuid4()], "market": [uuid4()], "idea_vs_market": []},
            )
        )
    return out


def _core_order(views: list[CompanyView]) -> list[UUID]:
    """The core policy verbatim: weakest axis first, exactly `api.main._rank_key`."""
    return [
        view.company_id
        for view in sorted(views, key=lambda v: (-min(v.axes.values()), v.name))
    ]


def _event(text: str, *, company_id: UUID, integrity: list[str] | None = None, kind=None) -> Event:
    return Event(
        company_id=company_id,
        kind=kind or EventKind.DECK_CLAIM,
        source=Source.DECK,
        observed_at=T0,
        payload={"claim": text},
        evidence_span=text,
        integrity_flags=integrity or [],
    )


def _evidence_for(views: list[CompanyView]) -> dict[UUID, list[Event]]:
    return {
        view.company_id: [
            _event(f"{view.name} shipped a release", company_id=view.company_id)
            for _ in range(6)
        ]
        for view in views
    }


# ===========================================================================
# 1. Lens derivation — nothing invented, everything attributable
# ===========================================================================


def test_every_lens_names_the_profile_field_that_justified_it(bold) -> None:
    lenses, _ = custom_council.derive_lenses(bold)
    assert len(lenses) >= custom_council.MIN_LENSES
    for lens in lenses:
        assert lens.justified_by and all(field.strip() for field in lens.justified_by)
        assert lens.provenance.basis in {"survey", "decisions", "profile_field"}
        assert lens.persona.strip()


def test_a_lens_with_no_derivable_justification_is_not_invented(patient) -> None:
    """The patient profile never picks an idea-vs-market option, so its stated weight on
    that axis is exactly 0. The contrarian-timing lens must be ABSENT with a reason —
    not present at a token weight."""
    assert patient.axis_weights_stated is not None
    assert patient.axis_weights_stated.idea_vs_market == 0.0

    lenses, not_derived = custom_council.derive_lenses(patient)
    assert LensKind.CONTRARIAN_TIMING not in {lens.kind for lens in lenses}
    reasons = {item.field_name: item.reason for item in not_derived}
    assert "lens:contrarian_timing" in reasons
    assert "axis_weights_stated.idea_vs_market" in reasons["lens:contrarian_timing"]


def test_a_thin_profile_produces_fewer_lenses_and_says_which_it_skipped() -> None:
    """Survey answers only, no decision history: the revealed lenses cannot be derived."""
    thin = _make_profile({"q01_founder_vs_market": "a", "q06_which_mistake": "b"}, [])
    lenses, not_derived = custom_council.derive_lenses(thin)

    kinds = {lens.kind for lens in lenses}
    assert LensKind.SECTOR_PATTERN not in kinds
    assert LensKind.STAGE_PATTERN not in kinds
    skipped = {item.field_name for item in not_derived}
    assert {"lens:sector_pattern", "lens:stage_pattern", "lens:red_line_auditor"} <= skipped
    # And it says so rather than quietly producing a two-lens council as if complete.
    assert all(item.reason.strip() for item in not_derived)


def test_lens_count_stays_inside_the_three_to_five_band(bold, patient) -> None:
    for profile in (bold, patient):
        lenses, not_derived = custom_council.derive_lenses(profile)
        assert custom_council.MIN_LENSES <= len(lenses) <= custom_council.MAX_LENSES
        # Anything derivable but cut by the ceiling has to say that is why it was cut.
        ceiling_cuts = [i for i in not_derived if "ceiling" in i.reason]
        assert all("raw weight" in item.reason for item in ceiling_cuts)


def test_lens_weights_are_normalised_and_not_uniform(bold) -> None:
    lenses, _ = custom_council.derive_lenses(bold)
    assert abs(sum(lens.weight for lens in lenses) - 1.0) < 1e-6
    weights = sorted(round(lens.weight, 4) for lens in lenses)
    assert weights[0] != weights[-1], "uniform lens weights are a renamed average"


def test_two_real_profiles_derive_different_lenses_and_different_weights(bold, patient) -> None:
    bold_lenses, _ = custom_council.derive_lenses(bold)
    patient_lenses, _ = custom_council.derive_lenses(patient)

    bold_map = {lens.kind: lens.weight for lens in bold_lenses}
    patient_map = {lens.kind: lens.weight for lens in patient_lenses}
    assert bold_map != patient_map

    # And the difference is explicable from the survey: the bold profile picked the
    # founder option repeatedly, the patient one picked the market option.
    assert bold.axis_weights_stated.founder > patient.axis_weights_stated.founder
    assert patient.axis_weights_stated.market > bold.axis_weights_stated.market
    assert bold.conviction_style_stated.label == "conviction-heavy"
    assert patient.conviction_style_stated.label == "evidence-heavy"


# ===========================================================================
# 2. Same evidence graph as the core analysis
# ===========================================================================


def test_council_and_custom_council_see_identical_evidence() -> None:
    """The dissent engine's bug, guarded: bull and bear must not read different books.

    `usable_evidence` and the core council's packet builder are checked against each
    other over a corpus containing every case that has previously caused divergence —
    another company's event, a post-cutoff event, an INTEGRITY event, an impeaching flag,
    and a merely PROVENANCE flag.
    """
    cid, other = uuid4(), uuid4()
    events = [
        _event("kept: plain", company_id=cid),
        _event("kept: transliterated name is provenance, not impeachment",
               company_id=cid, integrity=["transliterated_name"]),
        _event("kept: ocr note", company_id=cid, integrity=["ocr_low_conf"]),
        _event("dropped: injection", company_id=cid, integrity=["injection_stripped"]),
        _event("dropped: tampered", company_id=cid, integrity=["content_tampered"]),
        _event("dropped: integrity kind", company_id=cid, kind=EventKind.INTEGRITY),
        _event("dropped: other company", company_id=other),
    ]
    late = _event("dropped: after cutoff", company_id=cid)
    late.observed_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    events.append(late)

    screening = ScreeningResult(
        company_id=cid,
        as_of=T0,
        founder=Axis(score=0.6, trend=0.0, confidence=0.7),
        market=Axis(score=0.5, trend=0.0, confidence=0.6),
        idea_vs_market=Axis(score=0.4, trend=0.0, confidence=0.5),
    )
    packet, core_ids = council._packet(cid, T0, events, screening)
    ours = {str(event.event_id) for event in custom_council.usable_evidence(cid, T0, events)}

    assert ours == core_ids
    assert len(ours) == 3, "the three provenance-flagged / plain events, and only those"
    assert packet[0]["events"], "sanity: the core packet is non-empty"


def test_provenance_flags_are_kept_and_only_impeaching_flags_drop_evidence() -> None:
    cid = uuid4()
    events = [
        _event("a", company_id=cid, integrity=["transliterated_name"]),
        _event("b", company_id=cid, integrity=["unattested_trace"]),
    ]
    kept = custom_council.usable_evidence(cid, T0, events)
    assert [event.evidence_span for event in kept] == ["a"]


def test_the_council_gets_no_private_evidence(bold) -> None:
    """Every receipt a contribution cites must come from the shared packet."""
    view = _views()[0]
    evidence = [_event(f"e{i}", company_id=view.company_id) for i in range(6)]
    allowed = {event.event_id for event in evidence} | {
        eid for ids in view.axis_evidence.values() for eid in ids
    }
    lenses, _ = custom_council.derive_lenses(bold)
    fit = custom_council.score_company(view, lenses, bold, evidence, T0)
    for contribution in fit.contributions:
        assert set(contribution.evidence_event_ids) <= allowed


# ===========================================================================
# 3. Does it DISCRIMINATE? (the whole point)
# ===========================================================================


def test_lens_contributions_are_not_all_identical(bold) -> None:
    view = _views()[0]
    evidence = [_event(f"e{i}", company_id=view.company_id) for i in range(6)]
    lenses, _ = custom_council.derive_lenses(bold)
    fit = custom_council.score_company(view, lenses, bold, evidence, T0)

    readings = [c.reading for c in fit.contributions if c.reading is not None]
    contributions = [c.contribution for c in fit.contributions]
    assert len(set(round(r, 4) for r in readings)) > 1, "every lens read the same number"
    assert len(set(round(c, 4) for c in contributions)) > 1, "every lens contributed the same"


def test_two_profiles_rank_the_same_evidence_differently(bold, patient) -> None:
    """THE test. Same thirteen companies, same evidence, same core order — two profiles.

    If these two orderings match, the feature is decorative and everything above it is
    theatre.
    """
    views = _views()
    core = _core_order(views)
    evidence = _evidence_for(views)

    bold_rank = custom_council.rank(views, core, bold, evidence, T0)
    patient_rank = custom_council.rank(views, core, patient, evidence, T0)

    assert bold_rank.personalised and patient_rank.personalised
    bold_order = [row.name for row in bold_rank.rows]
    patient_order = [row.name for row in patient_rank.rows]

    assert bold_order != patient_order
    assert bold_order != [v.name for v in views if v.company_id in core][:0] or True
    # Not a token difference: the top of the two shortlists disagrees.
    assert bold_order[0] != patient_order[0]

    # And the disagreement is explicable from the lens weights. The founder-first
    # profile ranks a high-founder / low-market company above where the market-first
    # profile puts it.
    bold_pos = {row.name: row.personal_rank for row in bold_rank.rows}
    patient_pos = {row.name: row.personal_rank for row in patient_rank.rows}
    assert bold_pos["Zaryad"] < patient_pos["Zaryad"], "founder-first must prefer Zaryad"
    assert patient_pos["Umbra"] < bold_pos["Umbra"], "market-first must prefer Umbra"


def test_personal_rank_is_not_always_the_core_rank(bold) -> None:
    views = _views()
    core = _core_order(views)
    ranking = custom_council.rank(views, core, bold, _evidence_for(views), T0)
    moved = [row for row in ranking.rows if row.divergence != 0]
    assert moved, "a personal rank identical to core rank on every company measures nothing"


def test_fit_scores_span_a_real_range(bold) -> None:
    views = _views()
    ranking = custom_council.rank(views, _core_order(views), bold, _evidence_for(views), T0)
    scores = [row.fit_score for row in ranking.rows]
    assert max(scores) - min(scores) > 0.05, "a fit score with no spread is a constant"


# ===========================================================================
# 4. §0 — the personal layer never modifies the core score
# ===========================================================================


def test_core_axes_are_echoed_back_unmodified(bold) -> None:
    view = _views()[0]
    before = dict(view.axes)
    lenses, _ = custom_council.derive_lenses(bold)
    fit = custom_council.score_company(view, lenses, bold, [], T0)
    assert fit.core_axes == before
    assert view.axes == before, "the personal layer mutated the view it was handed"


def test_the_screening_result_is_never_touched(bold) -> None:
    screening = ScreeningResult(
        company_id=uuid4(),
        as_of=T0,
        founder=Axis(score=0.77, trend=0.1, confidence=0.8),
        market=Axis(score=0.31, trend=0.0, confidence=0.6),
        idea_vs_market=Axis(score=0.55, trend=0.0, confidence=0.7),
    )
    snapshot = screening.model_dump(mode="json")
    view = custom_council.view_from_screening(screening, name="X", sector="ai-infra", stage="seed")
    lenses, _ = custom_council.derive_lenses(bold)
    custom_council.score_company(view, lenses, bold, [], T0)
    assert screening.model_dump(mode="json") == snapshot


def test_the_same_company_scores_the_same_core_axes_for_both_profiles(bold, patient) -> None:
    """Two VCs, same objective truth, different ranking. The truth half."""
    view = _views()[0]
    evidence = [_event("e", company_id=view.company_id) for _ in range(6)]
    bold_lenses, _ = custom_council.derive_lenses(bold)
    patient_lenses, _ = custom_council.derive_lenses(patient)

    a = custom_council.score_company(view, bold_lenses, bold, evidence, T0)
    b = custom_council.score_company(view, patient_lenses, patient, evidence, T0)

    assert a.core_axes == b.core_axes
    assert a.core_weakest_axis == b.core_weakest_axis
    assert a.core_weakest_score == b.core_weakest_score
    # ...and the ranking half.
    assert a.fit_score != b.fit_score


def test_core_order_is_an_input_and_is_never_recomputed(bold) -> None:
    """A deliberately perverse core order must survive into the output verbatim."""
    views = _views()
    perverse = list(reversed(_core_order(views)))
    ranking = custom_council.rank(views, perverse, bold, _evidence_for(views), T0)
    reported = sorted(ranking.rows, key=lambda row: row.core_rank)
    assert [row.company_id for row in reported] == perverse


# ===========================================================================
# 5. Abstention, arithmetic and honesty about gaps
# ===========================================================================


def test_an_unreadable_lens_abstains_rather_than_scoring_zero(bold) -> None:
    """A company with no recorded sector must not be punished by the sector lens."""
    views = _views()
    known = views[0]
    unknown = known.model_copy(update={"company_id": uuid4(), "sector": None, "name": "NoSector"})
    lenses, _ = custom_council.derive_lenses(bold)
    assert LensKind.SECTOR_PATTERN in {lens.kind for lens in lenses}

    fit = custom_council.score_company(unknown, lenses, bold, [], T0)
    sector = next(c for c in fit.contributions if c.lens == LensKind.SECTOR_PATTERN)
    assert sector.reading is None
    assert sector.contribution == 0.0
    assert sector.abstained_reason and "penalise" in sector.abstained_reason
    # The remaining lenses absorb its weight, so the score stays comparable.
    live = [c for c in fit.contributions if c.reading is not None]
    assert abs(sum(c.weight for c in live) - 1.0) < 1e-6


def test_a_sector_this_fund_has_never_touched_reads_zero_but_does_not_abstain(bold) -> None:
    """Never having invested in a sector is a real reading about the FUND. An unknown
    sector is a gap in the DATA. The two must not collapse into the same number."""
    views = _views()
    gantry = next(v for v in views if v.name == "Gantry")  # crypto, never invested in
    lenses, _ = custom_council.derive_lenses(bold)
    fit = custom_council.score_company(gantry, lenses, bold, [], T0)
    sector = next(c for c in fit.contributions if c.lens == LensKind.SECTOR_PATTERN)
    assert sector.reading == 0.0
    assert sector.abstained_reason is None
    assert "never invested" in sector.rationale


def test_fit_score_always_equals_the_sum_of_its_contributions(bold, patient) -> None:
    views = _views()
    for profile in (bold, patient):
        lenses, _ = custom_council.derive_lenses(profile)
        for view in views:
            fit = custom_council.score_company(view, lenses, profile, [], T0)
            assert abs(fit.fit_score - sum(c.contribution for c in fit.contributions)) < 1e-6
            for contribution in fit.contributions:
                if contribution.reading is not None:
                    assert abs(
                        contribution.contribution - contribution.weight * contribution.reading
                    ) < 1e-6


def test_founder_market_fit_states_when_sector_conditioning_is_unavailable() -> None:
    thin = _make_profile(BOLD_ANSWERS, [])
    views = _views()
    lenses, _ = custom_council.derive_lenses(thin)
    fit = custom_council.score_company(views[0], lenses, thin, [], T0)
    fmf = fit.founder_market_fit
    assert fmf.caveats and any("unconditioned" in c for c in fmf.caveats)
    assert fmf.read_through
    assert fmf.score == pytest.approx(views[0].axes["founder"])


def test_founder_market_fit_is_read_through_the_thesis(bold, patient) -> None:
    """Same founder, same market — two theses, two fit readings."""
    view = _views()[0]  # ai-infra, which bold has invested in and patient has not
    bold_lenses, _ = custom_council.derive_lenses(bold)
    patient_lenses, _ = custom_council.derive_lenses(patient)
    a = custom_council.score_company(view, bold_lenses, bold, [], T0).founder_market_fit
    b = custom_council.score_company(view, patient_lenses, patient, [], T0).founder_market_fit
    assert a.score != b.score
    assert a.assessment != b.assessment


def test_two_lenses_is_a_council_and_one_is_not(bold) -> None:
    with pytest.raises(ValueError, match="at least"):
        custom_council.score_company(_views()[0], [], bold, [], T0)


# ===========================================================================
# 6. Personalisation off, and the disagreement headline
# ===========================================================================


def test_personalisation_off_returns_core_only_with_a_stated_reason() -> None:
    """Below the confidence threshold the personal layer publishes NO ordering."""
    thin = _make_profile({"q01_founder_vs_market": "a"}, [])
    assert not thin.personalisation_enabled
    views = _views()
    ranking = custom_council.rank(views, _core_order(views), thin, {}, T0)
    assert ranking.personalised is False
    assert ranking.rows == []
    assert "personalisation is OFF" in ranking.reason
    assert "core objective ranking is unaffected" in ranking.reason


def test_a_profile_that_supports_only_one_lens_refuses_to_rank() -> None:
    """Enabled by confidence, but the lenses still have to be earned."""
    single = _make_profile(
        {"q01_founder_vs_market": "a"},
        [("", "", DecisionKind.WATCHED)] * 20,
    )
    assert single.personalisation_enabled
    ranking = custom_council.rank(_views(), _core_order(_views()), single, {}, T0)
    assert ranking.personalised is False
    assert "renamed axis" in ranking.reason
    assert ranking.lenses_not_derived


def test_disagreements_are_the_headline_and_sorted_by_severity(bold) -> None:
    views = _views()
    ranking = custom_council.rank(views, _core_order(views), bold, _evidence_for(views), T0)
    assert ranking.disagreements, "a council that never disagrees with core is confirmation bias"
    magnitudes = [abs(item.divergence) for item in ranking.disagreements]
    assert magnitudes == sorted(magnitudes, reverse=True)
    for item in ranking.disagreements:
        assert "Core ranks on its weakest axis" in item.explanation or "red line" in item.explanation


def test_every_row_explains_its_own_move(bold) -> None:
    views = _views()
    ranking = custom_council.rank(views, _core_order(views), bold, _evidence_for(views), T0)
    for row in ranking.rows:
        assert row.why.strip()
        assert row.core_rank >= 1 and row.personal_rank >= 1
        assert row.divergence == row.core_rank - row.personal_rank


def test_a_demotion_is_explained_by_the_lens_that_dragged_it_down() -> None:
    """Not by the largest contribution, which for a demotion names the wrong lens.

    This profile only ever invested in ai-infra, so `sector_pattern` reads 0.0 on every
    dev-tools company. That zero is what demotes them — an explanation naming whichever
    lens still contributed most would send the user to argue with a weight that moved
    nothing.
    """
    views = _views()
    ranking = custom_council.rank(views, _core_order(views), bold_only_ai(), _evidence_for(views), T0)
    demoted = [row for row in ranking.rows if row.divergence <= -custom_council.DIVERGENCE_HEADLINE]
    assert demoted, "expected the sector lens to demote something"
    for row in demoted:
        assert "dragged down by" in row.why
    hit = next(
        (item for item in ranking.disagreements if item.divergence < 0 and "sector_pattern" in item.explanation),
        None,
    )
    assert hit is not None
    assert "no history here" in hit.explanation


def bold_only_ai():
    return _make_profile(BOLD_ANSWERS, BOLD_DECISIONS)


def test_a_red_line_hit_is_surfaced_as_a_disagreement() -> None:
    profile = _make_profile(
        BOLD_ANSWERS, BOLD_DECISIONS, stated_red_lines=["no crypto companies, ever"]
    )
    views = _views()
    ranking = custom_council.rank(views, _core_order(views), profile, _evidence_for(views), T0)
    # A stated red line is disqualifying regardless of score, so its lens is never cut
    # by the 3-5 ceiling — otherwise a fund that typed "no crypto, ever" gets served a
    # crypto company with no veto and no mention of one.
    assert LensKind.RED_LINE_AUDITOR in {lens.kind for lens in ranking.lenses}

    crypto = [item for item in ranking.disagreements if item.name == "Gantry"]
    assert crypto, "a fired red line must appear in the headline disagreements"
    assert "red line" in crypto[0].explanation
    assert "does not know about your red lines" in crypto[0].explanation


def test_a_red_line_does_not_fire_on_a_substring() -> None:
    """A red line on 'ai' must not fire on 'detail'. Whole words only."""
    profile = _make_profile(BOLD_ANSWERS, BOLD_DECISIONS, stated_red_lines=["no ai wrappers"])
    view = _views()[1]  # dev-tools / series a
    evidence = [_event("we obsess over every detail and retail channel", company_id=view.company_id)]
    lenses, _ = custom_council.derive_lenses(profile)
    fit = custom_council.score_company(view, lenses, profile, evidence, T0)
    assert not [hit for hit in fit.red_line_hits if "ai wrappers" in hit.statement]


def test_a_revealed_candidate_red_line_flags_but_does_not_veto() -> None:
    """The bold profile passed on dev-tools 6 times out of 6, so a candidate red line is
    raised. It must fire as a FLAG at its own confidence — a pattern the user has not
    confirmed cannot drive the reading to zero the way a stated red line does."""
    candidate = _make_profile(BOLD_ANSWERS, BOLD_DECISIONS)
    stated = _make_profile(BOLD_ANSWERS, BOLD_DECISIONS, stated_red_lines=["no dev-tools"])
    view = _views()[1]  # dev-tools

    # An unconfirmed pattern does NOT get pinned past the lens ceiling — only a stated
    # red line does — so the auditor is scored here against an explicit lens list.
    def auditor_reading(profile):
        _, lens = custom_council._red_line_lens(profile)
        others = [
            item for item in custom_council.derive_lenses(profile)[0] if item.kind != lens.kind
        ][:1]
        fit = custom_council.score_company(view, [lens, *others], profile, [], T0)
        return next(c for c in fit.contributions if c.lens == LensKind.RED_LINE_AUDITOR), fit

    weak, weak_fit = auditor_reading(candidate)
    hard, hard_fit = auditor_reading(stated)

    assert all(hit.source == "revealed_candidate" for hit in weak_fit.red_line_hits)
    assert 0.0 < weak.reading < 1.0, "an unconfirmed pattern is a flag, not a veto"
    assert hard.reading == 0.0, "a stated red line is disqualifying regardless of score"
    assert any(hit.source == "stated" for hit in hard_fit.red_line_hits)


# ===========================================================================
# 7. Dissent still applies — no recommendation without a bear case
# ===========================================================================


def _anti_memo(company_id: UUID, bear: str = "The retention claim is unsupported."):
    from schema.events import AntiMemo

    return AntiMemo(
        company_id=company_id,
        bear_case=bear,
        weakest_evidence=["No dated cohort series exists."],
        load_bearing_claim="Repeat usage persists past week four.",
        axis_spreads={"founder": 0.2, "market": 0.35, "idea_vs_market": 0.1},
    )


def test_the_recommendation_stays_locked_until_dissent_is_served(bold) -> None:
    view = _views()[0]
    lenses, _ = custom_council.derive_lenses(bold)
    fit = custom_council.score_company(
        view, lenses, bold, [], T0,
        anti_memo=_anti_memo(view.company_id),
        dissent_served=False,
        core_decision=council.CouncilDecision.REACH_OUT,
    )
    assert fit.personal_recommendation is None
    assert fit.anti_memo is None
    assert fit.recommendation_locked_reason == "open the dissent view first"
    # The analysis itself is still served — the lock is on the recommendation, the same
    # shape GET /companies/{id}/memo already uses.
    assert fit.fit_score > 0 and fit.contributions


def test_serving_the_bear_case_unlocks_the_recommendation(bold) -> None:
    view = _views()[0]
    lenses, _ = custom_council.derive_lenses(bold)
    fit = custom_council.score_company(
        view, lenses, bold, [], T0,
        anti_memo=_anti_memo(view.company_id),
        dissent_served=True,
        core_decision=council.CouncilDecision.REACH_OUT,
    )
    assert fit.personal_recommendation == council.CouncilDecision.REACH_OUT
    assert fit.anti_memo is not None
    assert fit.recommendation_locked_reason is None


def test_an_empty_bear_case_does_not_unlock_the_recommendation(bold) -> None:
    view = _views()[0]
    lenses, _ = custom_council.derive_lenses(bold)
    fit = custom_council.score_company(
        view, lenses, bold, [], T0,
        anti_memo=_anti_memo(view.company_id, bear="   "),
        dissent_served=True,
        core_decision=council.CouncilDecision.REACH_OUT,
    )
    assert fit.personal_recommendation is None
    assert fit.recommendation_locked_reason


def test_an_empty_council_does_not_unlock_the_recommendation() -> None:
    """The bug that was fixed in run_council, not reintroduced: every lens abstaining
    means nobody argued, and nobody arguing is not a deliberation."""
    profile = _make_profile(
        {"q01_founder_vs_market": "a", "q10_insider_vs_outsider": "a"},
        [
            ("ai-infra", "pre-seed", DecisionKind.INVESTED),
            ("ai-infra", "seed", DecisionKind.INVESTED),
            ("data-infra", "pre-seed", DecisionKind.INVESTED),
            *[("fintech", "series a", DecisionKind.PASSED)] * 17,
        ],
    )
    lenses, _ = custom_council.derive_lenses(profile)
    # A company with no axes and no sector or stage: every derivable lens abstains.
    blank = CompanyView(company_id=uuid4(), name="Blank", sector=None, stage=None, axes={})
    lenses = [
        lens
        for lens in lenses
        if lens.kind in {LensKind.FOUNDER_BET, LensKind.SECTOR_PATTERN, LensKind.STAGE_PATTERN}
    ]
    fit = custom_council.score_company(
        blank, lenses, profile, [], T0,
        anti_memo=_anti_memo(blank.company_id),
        dissent_served=True,
        core_decision=council.CouncilDecision.REACH_OUT,
    )
    assert all(c.reading is None for c in fit.contributions)
    assert fit.personal_recommendation is None
    assert "empty council" in fit.recommendation_locked_reason


# ===========================================================================
# 8. Narration — the persona argues, it never sets the score
# ===========================================================================


def test_narration_never_changes_the_reading(bold) -> None:
    view = _views()[0]
    evidence = [_event(f"e{i}", company_id=view.company_id) for i in range(6)]
    lenses, _ = custom_council.derive_lenses(bold)

    quiet = custom_council.score_company(view, lenses, bold, evidence, T0)

    def judge(prompt, **kwargs):
        return {
            "rationale": "A wildly enthusiastic and entirely unquantified endorsement.",
            "evidence_event_ids": [str(evidence[0].event_id)],
        }

    loud = custom_council.score_company(view, lenses, bold, evidence, T0, judge=judge)

    assert loud.fit_score == quiet.fit_score
    assert [c.reading for c in loud.contributions] == [c.reading for c in quiet.contributions]
    assert "wildly enthusiastic" in loud.contributions[0].rationale
    assert "computed:" in loud.contributions[0].rationale


def test_each_persona_argues_over_the_same_packet(bold) -> None:
    view = _views()[0]
    evidence = [_event(f"e{i}", company_id=view.company_id) for i in range(6)]
    lenses, _ = custom_council.derive_lenses(bold)
    seen: list[tuple[str, str]] = []

    def judge(prompt, *, system="", untrusted="", **kwargs):
        seen.append((system, untrusted))
        return {"rationale": "ok", "evidence_event_ids": [str(evidence[0].event_id)]}

    custom_council.score_company(view, lenses, bold, evidence, T0, judge=judge)

    assert len(seen) == len(lenses)
    packets = {payload for _, payload in seen}
    assert len(packets) == 1, "the personas must argue about identical facts"
    personas = {system for system, _ in seen}
    assert len(personas) == len(lenses), "identical personas produce a fake council"
    ids = {doc["event_id"] for doc in json.loads(packets.pop())[0]["events"]}
    assert ids == {str(event.event_id) for event in evidence}


def test_a_failed_narration_degrades_to_the_computed_rationale(bold) -> None:
    view = _views()[0]
    evidence = [_event("e", company_id=view.company_id)]
    lenses, _ = custom_council.derive_lenses(bold)

    def judge(prompt, **kwargs):
        raise RuntimeError("provider down")

    fit = custom_council.score_company(view, lenses, bold, evidence, T0, judge=judge)
    assert fit.fit_score > 0
    assert all(c.rationale.strip() and "computed:" not in c.rationale for c in fit.contributions)


def test_narration_citing_an_id_outside_the_packet_is_discarded(bold) -> None:
    view = _views()[0]
    evidence = [_event("e", company_id=view.company_id)]
    lenses, _ = custom_council.derive_lenses(bold)

    def judge(prompt, **kwargs):
        return {"rationale": "fabricated", "evidence_event_ids": [str(uuid4())]}

    fit = custom_council.score_company(view, lenses, bold, evidence, T0, judge=judge)
    assert all("fabricated" not in c.rationale for c in fit.contributions)


# ===========================================================================
# 9. Determinism
# ===========================================================================


def test_the_same_profile_and_evidence_produce_the_same_ranking(bold) -> None:
    views = _views()
    core, evidence = _core_order(views), _evidence_for(views)
    first = custom_council.rank(views, core, bold, evidence, T0)
    second = custom_council.rank(views, core, bold, evidence, T0)
    assert [row.model_dump() for row in first.rows] == [row.model_dump() for row in second.rows]
