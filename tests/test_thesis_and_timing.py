"""S0 thesis engine and signal-to-decision instrumentation.

Both were named in the spec and neither existed. The thesis file was served,
rendered and editable while being read by nothing — a picture of a control panel —
and signal-to-decision time returned zero grep hits across the repo.

These assert BEHAVIOUR: that changing the config changes a decision, and that the
clocks measure something. Two audits of this codebase found the same failure over
and over — a function existing while measuring nothing — so a test that only checks
a field is present would be repeating the mistake.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from core import thesis as thesis_mod
from core.timing import Stages, measure, signal_age_days

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _write(tmp_path, blob: dict):
    p = tmp_path / "thesis.json"
    p.write_text(json.dumps(blob))
    return p


# --- the thesis must actually govern something --------------------------------


def test_out_of_scope_sector_is_excluded_not_scored_down(tmp_path) -> None:
    """A fund that does not invest in a sector does not rank it lower — it does not
    look at it. Scoring it down would let a strong out-of-scope company outrank a
    weak in-scope one, which is not what a thesis means."""
    p = _write(tmp_path, {"sectors": [{"id": "ai-infra", "include": True}]})
    t = thesis_mod.load(p)

    ok, why = thesis_mod.in_scope(sector="ai-infra", thesis=t)
    assert ok and why is None

    ok, why = thesis_mod.in_scope(sector="fintech", thesis=t)
    assert not ok
    assert "outside the thesis" in why


def test_unknown_metadata_stays_in_scope(tmp_path) -> None:
    """Absent data is not disqualifying. Dropping a company for having no recorded
    sector is exactly how the Type 6 founder disappears — the same reasoning as the
    gate's absence classifier."""
    t = thesis_mod.load(_write(tmp_path, {"sectors": [{"id": "ai-infra", "include": True}]}))
    ok, why = thesis_mod.in_scope(sector=None, stage=None, geo=None, thesis=t)
    assert ok and why is None


def test_geo_is_unrestricted_by_default() -> None:
    """The shipped config leaves geography open and says why: a geographic filter is
    the cheapest way to systematically miss the founder this thesis exists to find."""
    t = thesis_mod.load()
    for region in ("eastern-europe", "south-asia", "west-africa", "south-america"):
        ok, _ = thesis_mod.in_scope(geo=region, thesis=t)
        assert ok, f"{region} must not be filtered out by default"


def test_risk_appetite_moves_the_evidence_bar_not_the_score(tmp_path) -> None:
    """Higher appetite means proceeding on THINNER evidence — a wider acceptable
    band. It must never move the score, or the same founder would be more capable at
    a bolder fund."""
    cautious = thesis_mod.load(_write(tmp_path, {"risk_appetite": {"value": 0.0}}))
    bold = thesis_mod.load(_write(tmp_path, {"risk_appetite": {"value": 1.0}}))

    assert thesis_mod.evidence_bar(bold) > thesis_mod.evidence_bar(cautious)
    assert thesis_mod.evidence_bar(cautious) > 0  # never zero, or nothing clears


def test_missing_thesis_is_permissive_not_empty(tmp_path) -> None:
    """A missing config must not silently filter the pipeline to nothing."""
    t = thesis_mod.load(tmp_path / "absent.json")
    ok, _ = thesis_mod.in_scope(sector="anything", stage="seed", geo="mars", thesis=t)
    assert ok
    assert thesis_mod.check_size(t)["min"] > 0


# --- signal-to-decision time --------------------------------------------------


def test_both_clocks_are_reported() -> None:
    """Reporting compute time alone is how a system claims to be fast. A 400ms
    decision over evidence that sat unexamined for two years is not speed."""
    stages = Stages()
    with stages.stage("read"):
        pass
    out = measure("c1", T0, stages, events=[])
    assert "compute_ms" in out
    assert "signal_age_days" in out
    assert out["stages_ms"], "per-stage breakdown is what makes the total actionable"


def test_no_evidence_reports_none_not_zero() -> None:
    """Zero would read as 'instant'. The truthful reading for a cold-start founder is
    'we have nothing to go on', which is None."""
    assert signal_age_days(None, T0) is None
    assert signal_age_days(T0 - timedelta(days=30), T0) == 30.0


def test_a_stage_entered_twice_accumulates() -> None:
    """Overwriting would report only the last visit and understate the total."""
    stages = Stages()
    for _ in range(3):
        with stages.stage("score"):
            sum(range(10_000))
    assert len(stages.marks) == 1
    assert stages.marks["score"] > 0
    assert stages.total_ms == round(stages.marks["score"], 1)
