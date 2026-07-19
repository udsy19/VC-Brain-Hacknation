"""LinkedIn career-history ingestion. Owner: B. OPT-IN, OFF BY DEFAULT.

WHY THERE IS NO FETCHER IN THIS FILE
------------------------------------
There is no scraper here, no headless browser, no HTTP client, and no
`sourcing.bus.fetch_*` call. That is deliberate and it is not an oversight to be
fixed later.

LinkedIn's robots.txt states: "The use of robots or other automated means to
access LinkedIn without the express permission of LinkedIn is strictly
prohibited." `/profile/`, `/me/` and `/connections*` are disallowed outright and
only LinkedInBot is granted `Allow: /`. The hiQ litigation concerned CFAA
liability for public data; it did not make automated access contract-permissible.
The product owner's decision to score this source did not, and could not, grant
a right to crawl it.

So every byte that enters through this module arrives because a human who was
entitled to it handed it over. Three paths, and `Provenance` admits no fourth:

  USER_PASTED    an analyst or the founder pastes a profile URL and types or
                 confirms the fields. A person read the page under their own
                 credentials; we store what they told us, not what we took.
  OFFICIAL_API   LinkedIn's Marketing/Talent APIs, where the org holds
                 credentials and the member consented. `fetch_via_official_api`
                 is the seam a token slots into. It raises today because no
                 credentials exist; it does not fall back to scraping, and if it
                 ever appears to, that is the bug.
  FOUNDER_EXPORT the founder's own "Get a copy of your data" archive, supplied
                 by them.

`profile_from_payload` is provenance-agnostic on purpose: all three paths produce
the same `Profile`, so adding real API credentials later is a config change
rather than a parsing rewrite. The module simply has no automated fetcher, and
`_assert_no_fetcher_exists` is asserted by the tests so that "no scraper" is a
property under test rather than a promise in a docstring.

WHAT THIS WEAKENS
-----------------
SHARED.md Invariant #3 says no institution allowlists anywhere in scoring,
substance-only by construction. This module is the one exception, it is off by
default, and `career_history_signals_enabled()` is the only switch. With the flag
false, `ingest_profile` refuses and the rules in intelligence/flags.py are not
appended to the rule set, so behaviour is byte-identical to the pre-flag build.

The objection is on the record in data/sources.json under `feature_flags` and in
docs/SOURCES.md section 6, and it has not been withdrawn: every field here is
self-authored, contradicted by nobody, editable after the founder learns they are
being assessed, and correlated with the coverage gaps this registry already
documents. The mitigations below are real but partial.

WHAT THIS MODULE REFUSES TO READ
--------------------------------
Organisation and school NAMES are never returned by `signal_payload` and never
reach a rule, a feature name, or a prompt. They are carried as `PROFILE_FACT`
display material only, exactly as arXiv affiliation already is. Only durations
and counts are scored. That keeps the banned-term grep green, and passing that
grep is emphatically NOT the same as being brand-neutral -- tenure length and
title-ladder density both correlate with organisation size and with the same
underlying advantage. See the `failure_modes` in the registry entry.

Self-reported evidence carries higher observation noise: events are ingested at
SELF_REPORTED_CONFIDENCE, which widens r_t in intelligence.flags.observation()
rather than pretending to a precision this source does not have.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

from schema.events import Event, RawSignal, Source
from sourcing.bus import ingest, parse_ts

SOURCE_ID = "linkedin"
FLAG_NAME = "career_history_signals_enabled"

# Self-authored, third-party-unverified, retroactively editable. The registry's
# standing rule on self-reported evidence: it does not get to look precise.
SELF_REPORTED_CONFIDENCE = 0.4

SELF_REPORTED_FLAG = "self_reported"
DATE_INFERRED_FLAG = "date_inferred"

_REGISTRY_PATH = Path("data/sources.json")
_ENV_OVERRIDE = "VCBRAIN_CAREER_HISTORY_SIGNALS"


class Provenance(StrEnum):
    """How a human handed us this data. There is no automated fourth option."""

    USER_PASTED = "user_pasted"
    OFFICIAL_API = "official_api"
    FOUNDER_EXPORT = "founder_export"


@dataclass(frozen=True)
class Role:
    """One position, as the founder described it.

    `organisation` is display-only and is never scored -- see the module
    docstring. `started_at`/`ended_at` are month-granular and self-reported;
    `ended_at=None` means current as of `Profile.supplied_at`.
    """

    title: str
    organisation: str
    started_at: datetime
    ended_at: datetime | None = None
    description: str = ""

    @property
    def tenure_months(self) -> int:
        end = self.ended_at or datetime.now(timezone.utc)
        return max(
            0, (end.year - self.started_at.year) * 12 + (end.month - self.started_at.month)
        )


@dataclass(frozen=True)
class Profile:
    """A career history supplied by a human, with its provenance attached."""

    profile_url: str
    provenance: Provenance
    roles: list[Role] = field(default_factory=list)
    entity_id: str | None = None
    company_id: str | None = None
    supplied_at: datetime | None = None

    def role_steps(self) -> int:
        """Distinct role steps within a single organisation, summed, brand-blind.

        Counts transitions, not positions: two roles at one organisation is one
        step. Organisation identity is used only to group; the name is discarded.
        """
        by_org: dict[str, int] = {}
        for role in self.roles:
            key = role.organisation.strip().casefold()
            by_org[key] = by_org.get(key, 0) + 1
        return sum(max(0, n - 1) for n in by_org.values())

    def longest_tenure_months(self) -> int:
        return max((r.tenure_months for r in self.roles), default=0)


# ---------------------------------------------------------------------------
# The flag. One switch, read from the registry, default false.
# ---------------------------------------------------------------------------


def career_history_signals_enabled() -> bool:
    """True only if the registry (or an explicit env override) turns it on.

    Default is FALSE and a missing, unreadable or malformed registry reads as
    FALSE. Failing closed matters here: the failure mode of failing open is that
    a parse error silently switches an invariant off, which is precisely the
    "guarantee that stopped holding without anyone noticing" this flag exists to
    prevent.
    """
    override = os.getenv(_ENV_OVERRIDE)
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes", "on"}
    return _flag_from_registry(_REGISTRY_PATH.as_posix(), _mtime(_REGISTRY_PATH))


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@lru_cache(maxsize=4)
def _flag_from_registry(path: str, _mtime_key: float) -> bool:
    """Cached on (path, mtime) so editing the registry invalidates it."""
    try:
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    entry = (blob.get("feature_flags") or {}).get(FLAG_NAME)
    if isinstance(entry, dict):
        return entry.get("value") is True
    return entry is True


class CareerHistoryDisabled(RuntimeError):
    """Raised when ingestion is attempted with the flag off."""


# ---------------------------------------------------------------------------
# Parsing. Provenance-agnostic: one shape, three legitimate suppliers.
# ---------------------------------------------------------------------------


def profile_from_payload(payload: dict[str, Any], provenance: Provenance) -> Profile:
    """Build a Profile from any of the three permitted paths.

    Deliberately source-agnostic. A pasted form, an official API response and a
    founder's data export all reduce to the same fields, so wiring a real API
    token later needs no new parser.
    """
    roles: list[Role] = []
    for raw in payload.get("roles") or payload.get("positions") or []:
        if not isinstance(raw, dict):
            continue
        started = parse_ts(raw.get("started_at") or raw.get("start_date"))
        if started is None:
            # No defensible start date means no tenure claim. Skipped, not guessed:
            # a guessed date would manufacture tenure the founder never claimed.
            continue
        roles.append(
            Role(
                title=str(raw.get("title") or "").strip(),
                organisation=str(raw.get("organisation") or raw.get("company") or "").strip(),
                started_at=started,
                ended_at=parse_ts(raw.get("ended_at") or raw.get("end_date")),
                description=str(raw.get("description") or "").strip(),
            )
        )
    return Profile(
        profile_url=str(payload.get("profile_url") or payload.get("url") or "").strip(),
        provenance=provenance,
        roles=roles,
        entity_id=payload.get("entity_id"),
        company_id=payload.get("company_id"),
        supplied_at=parse_ts(payload.get("supplied_at")),
    )


def fetch_via_official_api(member_urn: str, *, access_token: str | None = None) -> Profile:
    """Seam for official API access. Raises until credentials exist.

    This is the ONLY function in the module that would ever perform a network
    read, and it performs none today. It must never acquire an HTML fallback:
    the whole point of the seam is that the permitted path either works through
    LinkedIn's own consented API or does not happen at all.
    """
    token = access_token or os.getenv("LINKEDIN_ACCESS_TOKEN")
    if not token:
        raise CareerHistoryDisabled(
            "No LinkedIn API credentials configured. There is no fallback path: "
            "automated access without express permission is prohibited by "
            "LinkedIn's robots.txt, so the correct behaviour is to stop here and "
            "have a human supply the profile instead."
        )
    raise NotImplementedError(
        "Official LinkedIn API access is not wired up. Implement it against the "
        f"consented Talent/Marketing endpoints for {member_urn!r} and return a "
        "Profile via profile_from_payload(..., Provenance.OFFICIAL_API). Do not "
        "implement it by fetching profile HTML."
    )


# ---------------------------------------------------------------------------
# Ingestion -- through the normal bus, never around it.
# ---------------------------------------------------------------------------


def signal_payload(profile: Profile) -> dict[str, Any]:
    """The scored fields. Brand-blind by construction.

    Only durations, counts and falsifiability of prose. No organisation name, no
    school name, no title string reaches this dict, so nothing downstream can
    read one even by accident.
    """
    return {
        "source_id": SOURCE_ID,
        "self_reported": True,
        "tenure_months_longest": profile.longest_tenure_months(),
        "role_steps": profile.role_steps(),
        "roles_count": len(profile.roles),
        "scope_claims_with_specifics": sum(
            1 for r in profile.roles if _has_checkable_particular(r.description)
        ),
    }


_UNITS = (
    "ms",
    "qps",
    "rps",
    "gb",
    "tb",
    "req/s",
    "tokens/s",
    "%",
    "x faster",
    "users",
    "customers",
)


def _has_checkable_particular(description: str) -> bool:
    """True only if the prose contains something falsifiable against an artifact.

    Unbacked self-description scores zero. Same standard the intellectual_honesty
    signal already applies: an assertion that cannot be checked against another
    artifact does not count, however impressive it reads.
    """
    text = (description or "").casefold()
    if not text:
        return False
    has_number = any(ch.isdigit() for ch in text)
    return has_number and any(unit in text for unit in _UNITS)


def to_raw_signals(profile: Profile) -> list[RawSignal]:
    """One RawSignal per role. Untrusted founder-authored text, normal funnel.

    `observed_at` is the role's self-reported start month -- the closest thing to
    a world-produced timestamp this source has, and not close. It is flagged
    `date_inferred` alongside `self_reported` so the backtest can see exactly
    which signals rest on a clock the subject controls.
    """
    signals: list[RawSignal] = []
    payload = signal_payload(profile)
    for index, role in enumerate(profile.roles):
        signals.append(
            RawSignal(
                source=Source.WEB,  # registry catch-all; source_id carries the identity
                source_url=profile.profile_url or None,
                # Founder-authored. Sanitized and <untrusted_content>-wrapped by the bus.
                content=role.description,
                fetched_at=profile.supplied_at or datetime.now(timezone.utc),
                meta={
                    "kind": "profile_fact",
                    "observed_at": role.started_at.isoformat(),
                    "entity_id": profile.entity_id,
                    "company_id": profile.company_id,
                    "confidence": SELF_REPORTED_CONFIDENCE,
                    "integrity_flags": [SELF_REPORTED_FLAG, DATE_INFERRED_FLAG],
                    "evidence_span": role.description[:240] or None,
                    "provenance": profile.provenance.value,
                    "role_index": index,
                    "tenure_months": role.tenure_months,
                    "is_current": role.ended_at is None,
                    **payload,
                },
            )
        )
    return signals


def ingest_profile(profile: Profile) -> list[Event]:
    """Turn a human-supplied profile into Events. Refuses when the flag is off.

    Refuses rather than returning [] so that a caller who has not noticed the
    flag gets an error instead of silently-empty results that look like "this
    founder has no career history".
    """
    if not career_history_signals_enabled():
        raise CareerHistoryDisabled(
            f"{FLAG_NAME} is false. LinkedIn ingestion is opt-in and off by default "
            "-- see SHARED.md Invariant #3 and data/sources.json feature_flags."
        )
    if profile.provenance not in set(Provenance):
        raise ValueError(f"Unrecognised provenance {profile.provenance!r}.")
    events: list[Event] = []
    for raw in to_raw_signals(profile):
        events.extend(ingest(raw))
    return events


def _assert_no_fetcher_exists() -> None:
    """Asserted by tests/test_sourcing_linkedin.py -- 'no scraper' is under test.

    A docstring promising no scraper is worth nothing once someone is in a hurry.
    This module must import no HTTP client and reference no fetch helper, and the
    test reads this file's source to confirm it.
    """
