"""Attestation for Proof Protocol submissions. Owner: D.

The behavioural trace is the sharpest signal in the system — pushing back on the
planted bad constraint is worth half the behavioural score. It was also, until
this module, entirely client-supplied: a founder could POST
`{"pushed_back_on_constraint": true}` and buy the strongest signal we have.

The rule here is the one the validator already applies to claims: a self-reported
assertion is never evidence. So we split the trace in two —

  ATTESTED     fields the server observed itself (issue time, submission time,
               elapsed, and any commits fetched from a public repo)
  SELF_REPORTED everything the client asserted

— and mark the difference on the emitted events, so the score, the memo and the
UI all know which half they are looking at. We do not silently discard the
self-reported half; we refuse to let it masquerade as observed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from schema.events import utcnow

# Fields a founder cannot be trusted to report about themselves. Each is either
# replaced by a server observation or carried through explicitly marked.
SELF_REPORTABLE = (
    "pushed_back_on_constraint",
    "questions_asked",
    "commits",
    "started_at",
    "submitted_at",
    "iterations",
)

# Server-side record of when each challenge was issued AND who it was issued to.
# The point is that the anchor is OURS, not the client's. The company is recorded so a
# submission cannot be graded against a different company than the challenge was
# written for.
#
# IN POSTGRES (migration 008), not in process memory. Issue and submission are two
# separate HTTP requests, and on serverless they land on different lambdas — so an
# in-process dict meant the grader found no record at all. That is not a neutral
# failure: with no server observation of the issue time, `elapsed` falls back to the
# founder's own `started_at`, which is the exact substitution this module exists to
# prevent. The dict remains as a fallback for when the database cannot answer.
_ISSUED: dict[str, tuple[datetime, str | None]] = {}


def record_issue(
    challenge_id: str, issued_at: datetime | None = None, company_id: str | None = None
) -> None:
    from core import state

    when = issued_at or utcnow()
    company = str(company_id) if company_id else None
    _ISSUED[str(challenge_id)] = (when, company)
    state.write(
        "insert into proof_challenges (challenge_id, company_id, issued_at) "
        "values (?, ?, ?) on conflict (challenge_id) do nothing",
        (str(challenge_id), company, when.isoformat()),
    )


def _row(challenge_id: str) -> tuple[datetime, str | None] | None:
    """The issue record, preferring process memory (same lambda, no round trip) and
    falling back to the shared table (a different lambda, or a restarted process)."""
    rec = _ISSUED.get(str(challenge_id))
    if rec is not None:
        return rec

    from core import state
    from memory import db

    rows = state.fetch(
        "select company_id, issued_at from proof_challenges where challenge_id = ?",
        (str(challenge_id),),
    )
    if not rows:
        return None
    try:
        when = db.from_iso(rows[0]["issued_at"])
    except Exception:  # noqa: BLE001 - an unparseable anchor is no anchor
        return None
    company = rows[0].get("company_id")
    return (when, str(company) if company else None)


def issued_at(challenge_id: str) -> datetime | None:
    rec = _row(challenge_id)
    return rec[0] if rec else None


def issued_company(challenge_id: str) -> str | None:
    rec = _row(challenge_id)
    return rec[1] if rec else None


def reset() -> None:
    """Test/demo hook."""
    from core import state

    _ISSUED.clear()
    state.write("delete from proof_challenges")


def challenge_belongs_to(challenge_id: str, company_id: UUID | None) -> bool | None:
    """Does this challenge belong to this company? None when we cannot tell.

    Checked against our own issue record first, then the PROOF_CHALLENGE_ISSUED
    event. Without this, a submission for an easy company's challenge could be
    graded onto a different company's founder score.
    """
    if company_id is None:
        return None

    recorded = issued_company(challenge_id)
    if recorded is not None:
        return recorded == str(company_id)

    try:
        from memory import store
        from schema.events import EventKind

        cid = uuid_or_none(challenge_id)
        for ev in store.events(as_of=utcnow(), kind=str(EventKind.PROOF_CHALLENGE_ISSUED)):
            if str(ev.payload.get("challenge_id")) == str(challenge_id) or ev.event_id == cid:
                return ev.company_id == company_id
    except Exception:  # noqa: BLE001 - unknown provenance is not a mismatch
        return None
    return None


def attest(
    challenge_id: str,
    trace: dict,
    *,
    repo_url: str | None = None,
    demo: bool = False,
) -> tuple[dict, dict]:
    """Return (trace_for_grading, attestation).

    trace_for_grading keeps the client's fields — the grader still needs them —
    but every server-observable value is overwritten with what we actually saw,
    and the attestation travels INSIDE the trace so the grader can weight
    self-reported behaviour differently rather than only having it corrected
    after the fact.
    """
    now = utcnow()
    issued = issued_at(challenge_id)

    observed: dict[str, Any] = {"submitted_at": now.isoformat()}
    attested: list[str] = ["submitted_at"]

    if issued is not None:
        observed["started_at"] = issued.isoformat()
        observed["elapsed_seconds"] = round((now - issued).total_seconds(), 1)
        attested += ["started_at", "elapsed_seconds"]

    fetched, repo_status = _fetch_commits(repo_url) if repo_url else (None, "no_repo_given")
    if fetched is not None:
        observed["commits"] = fetched
        attested.append("commits")

    merged = {**(trace or {}), **observed}
    self_reported = [f for f in SELF_REPORTABLE if f in (trace or {}) and f not in attested]

    attestation = {
        "challenge_anchored": issued is not None,
        "attested_fields": attested,
        "self_reported_fields": self_reported,
        # Why there are no attested commits, when there are none. A repo that genuinely
        # does not exist and a repo we failed to ask about correctly are different
        # findings, and reading identically is what let the wrong-endpoint bug survive.
        "repo_status": repo_status,
        # A trace with no server-side anchor cannot be placed in time at all. It is
        # still graded — refusing outright would punish a founder for our plumbing —
        # but it never counts as observed behaviour.
        "trust": _trust(issued is not None, bool(fetched), self_reported, demo),
        "demo_seeded": demo,
        "note": _note(issued is not None, repo_status, self_reported, demo),
    }
    # The grader needs this BEFORE scoring, not just afterwards: a self-reported
    # pushback must not earn the same scalar as an observed one. Post-grade
    # confidence scaling stays as a second line of defence.
    merged["attestation"] = attestation
    return merged, attestation


def _trust(anchored: bool, has_repo: bool, self_reported: list[str], demo: bool) -> float:
    if demo:
        return 0.5  # seeded for the stage; honest about it rather than scored as real
    trust = 0.35
    if anchored:
        trust += 0.25
    if has_repo:
        trust += 0.3
    if not self_reported:
        trust += 0.1
    return round(min(trust, 1.0), 2)


# Why the commit evidence is missing, in the words the memo and the UI will show.
_REPO_NOTE = {
    "attested": "Commits fetched independently from the public repository.",
    "no_commits": "The repository is reachable but has no commits we could read.",
    "repo_not_found": "The repository does not exist or is not public — no commit evidence.",
    "rate_limited": "GitHub rate limit reached before we could read the repository; "
    "commit evidence is missing on our side, not the founder's.",
    "fetch_failed": "We could not reach GitHub; commit evidence is missing on our side, "
    "not the founder's.",
    "not_a_repo_url": "The submitted repository reference is not a GitHub repository URL.",
    "no_repo_given": "",
}


def _note(anchored: bool, repo_status: str, self_reported: list[str], demo: bool) -> str:
    if demo:
        return "Seeded demonstration completion. The machinery is real; this run was pre-recorded."
    parts = []
    parts.append(
        "Timing observed server-side."
        if anchored
        else "No server-side issue record — timing is unverified."
    )
    if note := _REPO_NOTE.get(repo_status, ""):
        parts.append(note)
    if self_reported:
        parts.append(
            "Self-reported and not independently observed: " + ", ".join(self_reported) + "."
        )
    return " ".join(parts)


def _fetch_commits(repo_url: str) -> tuple[list[dict] | None, str]:
    """Commits read from the public repo by us, not asserted by the submitter.

    Returns (commits, status). The status exists because the previous `None` meant four
    different things at once — repo does not exist, network down, we built the request
    wrong, repo exists but is empty — and the third of those was live in production:
    this called `github.scan(repo_url)`, which treats its argument as a LOGIN, so every
    attestation asked `GET /users/https://github.com/owner/repo`, took the 404 as
    "unreachable repo", and dropped its commit evidence silently.

    Commit MESSAGES arrive in RawSignal.content, not meta. Message text is
    founder-controlled and lands in a trace the memo can quote, so it goes through
    `bus.prepare()` — the same sanitizer every other ingested string takes (Invariant #4).
    """
    from sourcing import bus
    from sourcing.scanners import github

    try:
        signals = github.scan_repo(repo_url, limit=50)
    except ValueError:
        return None, "not_a_repo_url"
    except bus.RateLimited:
        return None, "rate_limited"
    except bus.FetchError as exc:
        # 404 is a fact about the submission. Anything else is a fact about us.
        return None, "repo_not_found" if exc.status == 404 else "fetch_failed"
    except Exception:  # noqa: BLE001 - still never fatal to a submission
        return None, "fetch_failed"

    commits = [
        {
            # The author date — when the work was done, not when we asked (Invariant #1).
            # Absent rather than backfilled with ingestion time.
            "at": s.meta.get("observed_at"),
            "message": bus.prepare(s).clean_text[:500],
            "sha": s.meta.get("sha"),
            "url": s.source_url,
        }
        for s in signals
    ]
    # `files` used to be read here and the scanner never emitted it, so it was 0 forever.
    # Populating it costs one GET per commit against a 60/hr budget; the key is dropped
    # rather than carried as a permanent zero that reads like a measurement.
    return (commits or None), "attested" if commits else "no_commits"


def apply(events: list, attestation: dict) -> list:
    """Stamp attestation onto graded events and scale their confidence by its trust.

    The filter treats PROOF_* events as low-noise, high-weight observations. An
    unattested trace must not buy that weight, so trust multiplies confidence
    rather than sitting beside it as a decorative field.
    """
    trust = float(attestation.get("trust", 0.35))
    for ev in events or []:
        ev.payload = {**(ev.payload or {}), "attestation": attestation}
        ev.confidence = round(min(1.0, max(0.0, ev.confidence * trust)), 3)
        if attestation.get("self_reported_fields") and "unattested_trace" not in ev.integrity_flags:
            ev.integrity_flags = [*ev.integrity_flags, "unattested_trace"]
    return events


def uuid_or_none(v: str) -> UUID | None:
    try:
        return UUID(str(v))
    except (ValueError, AttributeError, TypeError):
        return None
