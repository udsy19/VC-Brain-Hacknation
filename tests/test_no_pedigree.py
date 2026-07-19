"""Invariant #3, enforced by construction rather than intention. Owner: C.

Greps every prompt string and identifier in the codebase against the banned list.
This is the Type 6 guarantee and it is a hard CI fail.

SCOPE NOTE -- read before changing anything here.

Invariant #3 was weakened, deliberately and at the product owner's direction, by
one opt-in flag: `career_history_signals_enabled` (data/sources.json
feature_flags, default false). It admits self-reported career-history signals
from the `linkedin` source into scoring.

This file was NOT relaxed to accommodate that. The banned-term grep below still
runs over every scanned directory unconditionally, including the new
sourcing/linkedin.py, because that module reads only durations and counts and
never an organisation or school NAME -- so it passes the grep on its merits
rather than by exemption. If someone adds a brand name to it, this test fails,
which is the intended outcome.

What the flag changes is which RULES exist, so two tests below pin that down:

  test_career_history_rules_absent_when_flag_off -- with the flag off, the rule
      set is byte-identical to RULES and nothing career-derived can reach y_t.
  test_career_history_rules_require_the_flag -- turning the flag on is the ONLY
      way those rules appear, and even then they are skipped for any founder
      with no supplied profile, so absence of a profile costs nothing.

Passing the grep is not the same as being brand-neutral. Tenure length and
title-ladder density correlate with organisation size regardless of whether a
brand name appears in the source. That limitation is documented in
data/sources.json under the linkedin entry's failure_modes and is not something
this test can check.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from intelligence.banned import BANNED_TERMS

ROOT = Path(__file__).resolve().parent.parent
SCANNED_DIRS = ["core", "memory", "sourcing", "intelligence", "api", "backtest"]

# banned.py is the list itself; this test file quotes it too.
EXEMPT = {"intelligence/banned.py", "tests/test_no_pedigree.py"}


def _python_files() -> list[Path]:
    return [
        p
        for d in SCANNED_DIRS
        for p in (ROOT / d).rglob("*.py")
        if p.relative_to(ROOT).as_posix() not in EXEMPT  # as_posix so '/' matches on Windows
    ]


@pytest.mark.parametrize("term", BANNED_TERMS)
def test_no_pedigree_term_in_source(term: str) -> None:
    """No feature, prompt, or rule may reference school, employer brand, or investor."""
    # Word-boundary anchored: "mit" must not match "commit"/"limit"/"submit".
    pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    offenders = [
        f"{p.relative_to(ROOT).as_posix()}:{i}"
        for p in _python_files()
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert not offenders, (
        f"Pedigree term {term!r} found at {offenders}. Scoring must be substance-only "
        f"by construction — see SHARED.md Invariant #3."
    )


# ---------------------------------------------------------------------------
# The invariant is now conditional on one flag. These tests pin the condition,
# so the guarantee is either true or visibly false — never quietly weakened.
# ---------------------------------------------------------------------------


CAREER_RULE_IDS = {"role_tenure_duration", "role_progression", "self_described_scope"}


@pytest.fixture
def career_flag(monkeypatch):
    """Toggle the flag via its env override, with the registry left untouched."""

    def _set(enabled: bool) -> None:
        monkeypatch.setenv("VCBRAIN_CAREER_HISTORY_SIGNALS", "1" if enabled else "0")

    return _set


def test_flag_defaults_to_off_in_the_committed_registry() -> None:
    """The shipped default must be OFF. If this fails, the guarantee shipped broken."""
    import json

    blob = json.loads((ROOT / "data" / "sources.json").read_text(encoding="utf-8"))
    entry = blob["feature_flags"]["career_history_signals_enabled"]
    assert entry["value"] is False, (
        "career_history_signals_enabled must ship false. It weakens SHARED.md "
        "Invariant #3 and is opt-in by design."
    )


def test_career_history_rules_absent_when_flag_off(career_flag) -> None:
    """Flag off => the active rule set IS RULES. Not a superset, not a filtered copy."""
    from intelligence import flags

    career_flag(False)
    active = flags._active_rules()
    assert active is flags.RULES
    assert not (CAREER_RULE_IDS & {r.rule_id for r in active})


def test_career_history_rules_require_the_flag(career_flag) -> None:
    """Turning the flag ON is the ONLY way career history reaches scoring.

    This is the test the scope note promises. It asserts both halves: the rules
    are unreachable while the flag is off, and reachable only while it is on.
    """
    from intelligence import flags

    career_flag(False)
    assert not (CAREER_RULE_IDS & {r.rule_id for r in flags._active_rules()})

    career_flag(True)
    assert CAREER_RULE_IDS <= {r.rule_id for r in flags._active_rules()}

    career_flag(False)
    assert not (CAREER_RULE_IDS & {r.rule_id for r in flags._active_rules()}), (
        "Turning the flag back off must fully restore the invariant. If this "
        "fails, the weakening is not reversible and the guarantee is gone."
    )


def test_ingestion_refuses_while_the_flag_is_off(career_flag) -> None:
    """The other half of the gate: no career-history events can even be created."""
    from sourcing.linkedin import CareerHistoryDisabled, Profile, Provenance, ingest_profile

    career_flag(False)
    profile = Profile(profile_url="https://example.invalid/in/x", provenance=Provenance.USER_PASTED)
    with pytest.raises(CareerHistoryDisabled):
        ingest_profile(profile)
