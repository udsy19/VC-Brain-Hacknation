"""VC accounts and profile storage, plus the derivation that reads it.

Owner: personalisation layer (docs/DIFFERENTIATOR.md §1-§2).

Three things live here:

1. STORAGE — users, sessions, login throttle, and the two preference sources in two
   separate tables (`vc_survey_answers`, `vc_decisions`). Written in the SQLite dialect
   and translated centrally by memory/db.py, so this file never branches on backend.

2. PARSING — a defensive reader for uploaded CSV/JSON decision histories. Rows it cannot
   read are REPORTED, never dropped: a parser that silently discards a third of an upload
   produces a confident profile of nothing.

3. DERIVATION — axis weights, conviction style, sector/stage priors, red lines, and the
   stated-vs-revealed gap. Derived on READ from the raw tables rather than stored as a
   merged blob, so the two preference sources stay independently recomputable (§0).

THE RULE THIS FILE IS BUILT AROUND: nothing is inferred that the submitted data does not
support. Where the evidence is too thin, the field comes back ABSENT with a reason
recorded in `DerivedProfile.not_inferred` — never filled with a plausible default. A
profile built from four decisions has to say it was built from four decisions, which is
the same discipline the founder scorer already applies to thin evidence.
"""

from __future__ import annotations

import csv
import io
import json
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable
from uuid import UUID, uuid4

from memory import db
from schema.vc import (
    AXIS_SIGNALS,
    SURVEY_BY_ID,
    SURVEY_QUESTIONS,
    AuthoredLens,
    AuthoredLensPatch,
    AuthoredLensWrite,
    AxisWeights,
    Choice,
    ConvictionStyle,
    DecisionKind,
    DecisionUploadResult,
    DerivedProfile,
    GapFinding,
    GapReport,
    GapUncomputable,
    LensOrigin,
    NotInferred,
    PastDecision,
    Prior,
    Provenance,
    RedLine,
    RejectedRow,
    SurveyAnswer,
    User,
    VCProfile,
)

# ---------------------------------------------------------------------------
# Tunable thresholds. Named and gathered, because every one of them is a judgement
# call that a reader is entitled to disagree with, and a magic number buried in a
# branch is a judgement call nobody can find.
# ---------------------------------------------------------------------------

#: Decisions at which the revealed side is considered fully evidenced. Below this the
#: revealed confidence scales linearly, so a 4-row history reports 0.2, not 1.0.
DECISIONS_FOR_FULL_CONFIDENCE = 20

#: Minimum invested rows before any revealed prior or revealed conviction style is
#: computed at all. Two investments are an anecdote, not a concentration.
MIN_INVESTED_FOR_PRIORS = 3

#: A sector/stage must have at least this many decisions, ALL of them passes, before we
#: will even raise it as a *candidate* red line for the user to confirm.
MIN_ROWS_FOR_RED_LINE_CANDIDATE = 4

#: Below this overall confidence, personalisation is OFF and the API says why (§"What
#: could make this worse": a profile from too little data runs core rank only).
PERSONALISATION_MIN_CONFIDENCE = 0.35

#: How far stated and revealed must diverge before it is reported as a finding rather
#: than an agreement. Both are on a -1..1 scale, so magnitude is the gap halved.
GAP_REPORT_THRESHOLD = 0.25

SESSION_TTL = timedelta(days=7)

#: Login throttle: failures allowed per email before a lockout, and how long it lasts.
MAX_LOGIN_FAILURES = 5
LOCKOUT = timedelta(minutes=15)
FAILURE_WINDOW = timedelta(minutes=15)


# ---------------------------------------------------------------------------
# Schema. Postgres gets it from schema/migrations/002_vc_profiles.sql (applied on
# connect); SQLite gets it here. Two dialects of one schema is the pattern already
# established by db.SCHEMA / 001_init.sql, not a new one invented for this file.
# ---------------------------------------------------------------------------

SQLITE_SCHEMA = """
create table if not exists users (
    user_id       text primary key,
    email         text not null unique,
    password_hash text not null,
    created_at    text not null
);

create table if not exists sessions (
    session_id text primary key,
    token_hash text not null unique,
    user_id    text not null references users(user_id),
    created_at text not null,
    expires_at text not null
);

create index if not exists idx_sessions_user on sessions (user_id);
create index if not exists idx_sessions_expires on sessions (expires_at);

create table if not exists login_attempts (
    email            text primary key,
    failures         integer not null default 0,
    first_failure_at text not null,
    locked_until     text
);

create table if not exists vc_profiles (
    profile_id       text primary key,
    user_id          text not null unique references users(user_id),
    fund_name        text,
    focus_sectors    text not null default '[]',
    stated_red_lines text not null default '[]',
    created_at       text not null,
    updated_at       text not null
);

create table if not exists vc_survey_answers (
    profile_id  text not null references vc_profiles(profile_id),
    question_id text not null,
    choice      text not null check (choice in ('a', 'b')),
    answered_at text not null,
    primary key (profile_id, question_id)
);

create table if not exists vc_decisions (
    decision_id text primary key,
    profile_id  text not null references vc_profiles(profile_id),
    company     text not null,
    sector      text,
    stage       text,
    decision    text not null check (decision in ('invested', 'passed', 'watched')),
    decided_on  text,
    rationale   text,
    outcome     text,
    source_row  integer,
    uploaded_at text not null
);

create index if not exists idx_vc_decisions_profile on vc_decisions (profile_id);

create table if not exists vc_authored_lenses (
    lens_id    text primary key,
    profile_id text not null references vc_profiles(profile_id),
    name       text not null,
    quality    text not null,
    persona    text not null,
    weight     real not null check (weight >= 0.01 and weight <= 1.0),
    origin     text not null check (origin in ('authored', 'template')),
    created_at text not null,
    updated_at text not null
);

create unique index if not exists idx_vc_authored_lenses_profile_name
    on vc_authored_lenses (profile_id, lower(trim(name)));

create index if not exists idx_vc_authored_lenses_profile
    on vc_authored_lenses (profile_id, created_at);
"""

_ensured: dict[int, Any] = {}


def conn() -> Any:
    """The shared connection, with the profile tables guaranteed to exist.

    On Postgres they arrive via migration 002, which db.connect() applies on first
    connect. On SQLite there are no migrations, so the DDL runs here — once per
    connection object, tracked by identity so a test that repoints VCBRAIN_DB_PATH
    re-ensures against the new file.
    """
    c = db.connect()
    if _ensured.get(id(c)) is not c and db.backend() == db.SQLITE:
        c.executescript(SQLITE_SCHEMA)
        c.commit()
        # Keyed by id() but holding the connection itself, because CPython reuses the
        # id of a freed object: a bare set of ints would report "already ensured" for a
        # brand-new connection that happened to land on a dead one's address, and the
        # tables would silently not exist.
        _ensured[id(c)] = c
    return c


def write(sql: str, args: tuple | list = ()) -> None:
    """Every INSERT/UPDATE/DELETE goes through here, because the two backends disagree
    about transactions and only one of them is forgiving.

    The Postgres wrapper is autocommit and its `commit()` is a documented no-op. Python's
    sqlite3 is NOT: a bare `execute` opens a transaction that is only visible to the
    connection that opened it, and is discarded when the process exits. Reads inside one
    process therefore see their own uncommitted writes, which is exactly why this was
    invisible to both the test suite and a single-process manual run — and why every
    account would have disappeared the first time the server restarted.
    """
    c = conn()
    c.execute(sql, args)
    c.commit()


def fetch(sql: str, args: tuple | list = ()) -> list[dict]:
    """Every read goes through here, because the two backends disagree on row type:
    SQLite hands back `sqlite3.Row` (indexable, but no `.get`) and the Postgres wrapper
    hands back plain dicts. Normalising once means no call site has to know which, and
    an optional column can be read with `.get()` on either."""
    return [dict(row) for row in conn().execute(sql, args).fetchall()]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return db.to_iso(dt)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


def create_user(email: str, password_hash: str) -> User:
    """Raises ValueError if the email is taken. The caller must NOT relay that to the
    client verbatim — see api/auth.register for why an existence oracle matters."""
    if get_user_by_email(email) is not None:
        raise ValueError("email already registered")
    user = User(user_id=uuid4(), email=email, created_at=_now())
    write(
        "insert into users (user_id, email, password_hash, created_at) values (?, ?, ?, ?)",
        (str(user.user_id), email, password_hash, _iso(user.created_at)),
    )
    return user


def _user_row(row: dict | None) -> User | None:
    if row is None:
        return None
    return User(
        user_id=UUID(str(row["user_id"])),
        email=row["email"],
        created_at=db.from_iso(row["created_at"]),
    )


def get_user_by_email(email: str) -> User | None:
    rows = fetch("select * from users where email = ?", (email,))
    return _user_row(rows[0] if rows else None)


def get_user(user_id: UUID) -> User | None:
    rows = fetch("select * from users where user_id = ?", (str(user_id),))
    return _user_row(rows[0] if rows else None)


def password_hash_for(email: str) -> str | None:
    """The ONLY read path for a hash. Isolated to one function so that grepping for
    where hashes leave the database returns exactly one answer."""
    rows = fetch("select password_hash from users where email = ?", (email,))
    return rows[0]["password_hash"] if rows else None


# ---------------------------------------------------------------------------
# Sessions — the token itself is never stored, only its sha256
# ---------------------------------------------------------------------------


def create_session(user_id: UUID, token_hash: str, ttl: timedelta = SESSION_TTL) -> datetime:
    now = _now()
    expires = now + ttl
    write(
        "insert into sessions (session_id, token_hash, user_id, created_at, expires_at) "
        "values (?, ?, ?, ?, ?)",
        (str(uuid4()), token_hash, str(user_id), _iso(now), _iso(expires)),
    )
    return expires


def user_for_session(token_hash: str) -> User | None:
    """None for unknown OR expired. An expired session is indistinguishable from no
    session, which is what makes the unauthenticated fallback path uniform."""
    rows = fetch("select * from sessions where token_hash = ?", (token_hash,))
    if not rows:
        return None
    row = rows[0]
    if db.from_iso(row["expires_at"]) <= _now():
        delete_session(token_hash)
        return None
    return get_user(UUID(str(row["user_id"])))


def delete_session(token_hash: str) -> None:
    write("delete from sessions where token_hash = ?", (token_hash,))


def purge_expired_sessions() -> None:
    write("delete from sessions where expires_at <= ?", (_iso(_now()),))


# ---------------------------------------------------------------------------
# Login throttle
# ---------------------------------------------------------------------------


def lockout_remaining(email: str) -> timedelta | None:
    rows = fetch("select * from login_attempts where email = ?", (email,))
    if not rows or not rows[0].get("locked_until"):
        return None
    remaining = db.from_iso(rows[0]["locked_until"]) - _now()
    return remaining if remaining > timedelta(0) else None


def record_login_failure(email: str) -> None:
    """Counts failures inside a rolling window and locks the email when it overflows.

    Keyed on email rather than IP: a conference network puts every attendee behind one
    address, so an IP-keyed limiter would lock the room out the moment one person
    fat-fingered a password."""
    now = _now()
    rows = fetch("select * from login_attempts where email = ?", (email,))
    if not rows:
        write(
            "insert into login_attempts (email, failures, first_failure_at) values (?, ?, ?)",
            (email, 1, _iso(now)),
        )
        return

    row = rows[0]
    started = db.from_iso(row["first_failure_at"])
    failures = int(row["failures"]) + 1 if now - started <= FAILURE_WINDOW else 1
    started_at = started if now - started <= FAILURE_WINDOW else now
    locked = _iso(now + LOCKOUT) if failures >= MAX_LOGIN_FAILURES else None
    write(
        "update login_attempts set failures = ?, first_failure_at = ?, locked_until = ? "
        "where email = ?",
        (failures, _iso(started_at), locked, email),
    )


def clear_login_failures(email: str) -> None:
    write("delete from login_attempts where email = ?", (email,))


# ---------------------------------------------------------------------------
# Profile record + the two preference sources
# ---------------------------------------------------------------------------


def _profile_row(user_id: UUID) -> dict | None:
    rows = fetch("select * from vc_profiles where user_id = ?", (str(user_id),))
    return rows[0] if rows else None


def ensure_profile(user_id: UUID) -> UUID:
    """Every user has exactly one profile row. Created empty — an empty profile is a
    real state (personalisation off), not an error."""
    row = _profile_row(user_id)
    if row:
        return UUID(str(row["profile_id"]))
    profile_id = uuid4()
    now = _iso(_now())
    write(
        "insert into vc_profiles (profile_id, user_id, fund_name, focus_sectors, "
        "stated_red_lines, created_at, updated_at) values (?, ?, ?, ?, ?, ?, ?)",
        (str(profile_id), str(user_id), None, "[]", "[]", now, now),
    )
    return profile_id


def update_profile(
    user_id: UUID,
    *,
    fund_name: str | None = None,
    focus_sectors: list[str] | None = None,
    stated_red_lines: list[str] | None = None,
) -> None:
    """Partial update: None means "not supplied", so a PUT that sets only fund_name
    does not wipe the sectors."""
    profile_id = ensure_profile(user_id)
    sets, args = [], []
    if fund_name is not None:
        sets.append("fund_name = ?")
        args.append(fund_name)
    if focus_sectors is not None:
        sets.append("focus_sectors = ?")
        args.append(json.dumps([s.strip().lower() for s in focus_sectors if s.strip()]))
    if stated_red_lines is not None:
        sets.append("stated_red_lines = ?")
        args.append(json.dumps([s.strip() for s in stated_red_lines if s.strip()]))
    sets.append("updated_at = ?")
    args.append(_iso(_now()))
    args.append(str(profile_id))
    write(f"update vc_profiles set {', '.join(sets)} where profile_id = ?", args)


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    try:
        parsed = json.loads(value or "[]")
    except (TypeError, ValueError):
        return []
    return [str(v) for v in parsed] if isinstance(parsed, list) else []


def save_survey(user_id: UUID, answers: Iterable[SurveyAnswer]) -> int:
    """Upsert answers. Returns how many were stored.

    Answers to unknown question ids are DISCARDED rather than stored, so a stale client
    cannot inject a question that the derivation has no signals for. An unanswered
    question keeps no row at all — absence is the honest record of "not answered".
    """
    profile_id = ensure_profile(user_id)
    stored = 0
    for answer in answers:
        if answer.question_id not in SURVEY_BY_ID:
            continue
        write(
            "delete from vc_survey_answers where profile_id = ? and question_id = ?",
            (str(profile_id), answer.question_id),
        )
        write(
            "insert into vc_survey_answers (profile_id, question_id, choice, answered_at) "
            "values (?, ?, ?, ?)",
            (str(profile_id), answer.question_id, str(answer.choice), _iso(_now())),
        )
        stored += 1
    return stored


def get_survey(user_id: UUID) -> list[SurveyAnswer]:
    row = _profile_row(user_id)
    if not row:
        return []
    rows = fetch(
        "select question_id, choice from vc_survey_answers where profile_id = ?",
        (str(row["profile_id"]),),
    )
    return [
        SurveyAnswer(question_id=r["question_id"], choice=Choice(r["choice"]))
        for r in rows
        if r["question_id"] in SURVEY_BY_ID
    ]


def save_decisions(user_id: UUID, decisions: Iterable[PastDecision], *, replace: bool) -> int:
    """`replace=True` swaps the whole history — a re-upload is a correction of the file,
    not an append, and appending would double every row on a second upload."""
    profile_id = ensure_profile(user_id)
    if replace:
        write("delete from vc_decisions where profile_id = ?", (str(profile_id),))
    n = 0
    for d in decisions:
        write(
            "insert into vc_decisions (decision_id, profile_id, company, sector, stage, "
            "decision, decided_on, rationale, outcome, source_row, uploaded_at) "
            "values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid4()),
                str(profile_id),
                d.company,
                d.sector,
                d.stage,
                str(d.decision),
                d.decided_on.isoformat() if d.decided_on else None,
                d.rationale,
                d.outcome,
                d.source_row,
                _iso(_now()),
            ),
        )
        n += 1
    return n


def get_decisions(user_id: UUID) -> list[PastDecision]:
    row = _profile_row(user_id)
    if not row:
        return []
    rows = fetch(
        "select * from vc_decisions where profile_id = ? order by source_row",
        (str(row["profile_id"]),),
    )
    out = []
    for r in rows:
        decided = r.get("decided_on")
        if isinstance(decided, str) and decided:
            decided = date.fromisoformat(decided[:10])
        elif isinstance(decided, datetime):
            decided = decided.date()
        elif not isinstance(decided, date):
            decided = None
        out.append(
            PastDecision(
                company=r["company"],
                sector=r.get("sector"),
                stage=r.get("stage"),
                decision=DecisionKind(r["decision"]),
                decided_on=decided,
                rationale=r.get("rationale"),
                outcome=r.get("outcome"),
                source_row=r.get("source_row"),
            )
        )
    return out


# ---------------------------------------------------------------------------
# §3 — user-authored council lenses
#
# These are the THIRD input, and they are deliberately not merged into either of the
# other two. The survey is what the VC said, the decision history is what the VC did,
# and an authored lens is what the VC *asked for* — a direct instruction, not an
# inference from either source. `derive()` does not read this table, which is what makes
# the guarantee below true by construction rather than by care:
#
#     RE-DERIVING A PROFILE NEVER TOUCHES AN AUTHORED LENS.
#
# A survey change recomputes the derived lenses and their weights. It cannot create,
# edit, reweight or delete an authored one, because derivation has no write path here
# and no read path either. What re-deriving CAN do is change how many derived lenses
# compete for the remaining seats under the ceiling — and `compose_council` reports that
# displacement in `not_derived` by name rather than letting a lens quietly vanish.
#
# NOTHING IN HERE CREATES A LENS IMPLICITLY. There is no ensure_*, no default set and no
# seeding. A profile with no authored lenses has none, and that is a real state.
# ---------------------------------------------------------------------------


def _authored_row(row: dict) -> AuthoredLens:
    return AuthoredLens(
        lens_id=UUID(str(row["lens_id"])),
        profile_id=UUID(str(row["profile_id"])),
        name=row["name"],
        quality=row["quality"],
        persona=row["persona"],
        weight=float(row["weight"]),
        origin=LensOrigin(str(row["origin"])),
        created_at=db.from_iso(row["created_at"]),
        updated_at=db.from_iso(row["updated_at"]),
    )


def list_authored_lenses(user_id: UUID) -> list[AuthoredLens]:
    """Every council agent this user has authored, oldest first.

    Creation order, not weight order: the council builder is a list the VC maintains,
    and reordering it under them on every weight tweak would make an edit feel like a
    reshuffle. Weight ordering is the scorer's business, not the editor's.
    """
    row = _profile_row(user_id)
    if not row:
        return []
    return [
        _authored_row(r)
        for r in fetch(
            "select * from vc_authored_lenses where profile_id = ? order by created_at, lens_id",
            (str(row["profile_id"]),),
        )
    ]


def get_authored_lens(user_id: UUID, lens_id: UUID) -> AuthoredLens | None:
    """Scoped to the owning profile, so an id guessed from another account reads as
    absent rather than as someone else's council agent."""
    row = _profile_row(user_id)
    if not row:
        return None
    rows = fetch(
        "select * from vc_authored_lenses where profile_id = ? and lens_id = ?",
        (str(row["profile_id"]), str(lens_id)),
    )
    return _authored_row(rows[0]) if rows else None


def _name_taken(profile_id: UUID, name: str, *, excluding: UUID | None = None) -> bool:
    rows = fetch(
        "select lens_id, name from vc_authored_lenses where profile_id = ?", (str(profile_id),)
    )
    target = name.strip().lower()
    return any(
        str(r["name"]).strip().lower() == target
        and (excluding is None or str(r["lens_id"]) != str(excluding))
        for r in rows
    )


def create_authored_lens(user_id: UUID, body: AuthoredLensWrite) -> AuthoredLens:
    """Store one agent. Raises ValueError on a duplicate name.

    Duplicate names are refused rather than de-duplicated with a suffix: the ranking
    explanation names the lens that moved a company, and two agents called the same
    thing make that explanation unusable.
    """
    profile_id = ensure_profile(user_id)
    if _name_taken(profile_id, body.name):
        raise ValueError(
            f"you already have a council agent called {body.name!r} — the ranking "
            "explanation names the agent that moved a company, so names must be distinct"
        )
    now = _now()
    lens_id = uuid4()
    write(
        "insert into vc_authored_lenses (lens_id, profile_id, name, quality, persona, "
        "weight, origin, created_at, updated_at) values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            str(lens_id),
            str(profile_id),
            body.name,
            body.quality,
            body.persona,
            float(body.weight),
            str(body.origin),
            _iso(now),
            _iso(now),
        ),
    )
    return AuthoredLens(
        lens_id=lens_id,
        profile_id=profile_id,
        name=body.name,
        quality=body.quality,
        persona=body.persona,
        weight=float(body.weight),
        origin=body.origin,
        created_at=now,
        updated_at=now,
    )


def update_authored_lens(
    user_id: UUID, lens_id: UUID, patch: AuthoredLensPatch
) -> AuthoredLens | None:
    """Partial edit. None means "not supplied", so a PUT that only moves the weight
    slider does not blank the persona. Returns None if this user has no such lens."""
    existing = get_authored_lens(user_id, lens_id)
    if existing is None:
        return None
    if patch.name is not None and _name_taken(existing.profile_id, patch.name, excluding=lens_id):
        raise ValueError(f"you already have a council agent called {patch.name!r}")

    sets, args = [], []
    for field in ("name", "quality", "persona"):
        value = getattr(patch, field)
        if value is not None:
            sets.append(f"{field} = ?")
            args.append(value)
    if patch.weight is not None:
        sets.append("weight = ?")
        args.append(float(patch.weight))
    if patch.origin is not None:
        sets.append("origin = ?")
        args.append(str(patch.origin))
    now = _now()
    sets.append("updated_at = ?")
    args.extend([_iso(now), str(existing.profile_id), str(lens_id)])
    write(
        f"update vc_authored_lenses set {', '.join(sets)} where profile_id = ? and lens_id = ?",
        args,
    )
    return get_authored_lens(user_id, lens_id)


def delete_authored_lens(user_id: UUID, lens_id: UUID) -> bool:
    """True if a row was removed. A delete is permanent — there is no soft-delete flag,
    because a council agent the VC removed but that still scores is exactly the kind of
    invisible input this layer exists to rule out."""
    if get_authored_lens(user_id, lens_id) is None:
        return False
    write(
        "delete from vc_authored_lenses where profile_id = ? and lens_id = ?",
        (str(ensure_profile(user_id)), str(lens_id)),
    )
    return True


# ---------------------------------------------------------------------------
# Parsing an uploaded decision history
# ---------------------------------------------------------------------------

_HEADER_ALIASES = {
    "company": {"company", "company_name", "name", "startup", "target"},
    "sector": {"sector", "industry", "category", "vertical", "space"},
    "stage": {"stage", "round", "round_stage"},
    "decision": {"decision", "action", "verdict", "call"},
    "date": {"date", "decided_on", "decision_date", "decided_at", "when"},
    "rationale": {"rationale", "reason", "notes", "why", "comment"},
    "outcome": {"outcome", "result", "what_happened"},
}

_DECISION_WORDS = {
    DecisionKind.INVESTED: {"invested", "invest", "yes", "funded", "led", "participated", "y"},
    DecisionKind.PASSED: {"passed", "pass", "no", "declined", "decline", "rejected", "n"},
    DecisionKind.WATCHED: {"watched", "watch", "watching", "tracking", "track", "monitor", "maybe"},
}


def _canonical_header(raw: str) -> str | None:
    key = re.sub(r"[^a-z0-9]+", "_", (raw or "").strip().lower()).strip("_")
    for canonical, aliases in _HEADER_ALIASES.items():
        if key in aliases:
            return canonical
    return None


def _parse_decision_word(raw: Any) -> DecisionKind | None:
    word = str(raw or "").strip().lower()
    for kind, words in _DECISION_WORDS.items():
        if word in words:
            return kind
    return None


def _parse_date(raw: Any) -> tuple[date | None, str | None]:
    """Returns (date, warning). A date we cannot read is a WARNING, not a rejection —
    the row's decision and sector are still perfectly legible, and no derivation in this
    module weights by time, so discarding the whole row over its date would throw away
    good revealed preference to protect a field nothing reads.

    Ambiguous numeric formats (03/04/2024) are refused rather than guessed: picking a
    locale would silently invent a date that the user never supplied.
    """
    text = str(raw or "").strip()
    if not text:
        return None, None
    try:
        return date.fromisoformat(text[:10]), None
    except ValueError:
        pass
    for fmt in ("%Y/%m/%d", "%d %b %Y", "%b %d, %Y", "%d %B %Y", "%B %d, %Y", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(text, fmt).date(), None
        except ValueError:
            continue
    parts = re.split(r"[/\-.]", text)
    if len(parts) == 3 and all(p.strip().isdigit() for p in parts):
        a, b, y = (int(p) for p in parts)
        if a > 12 and b <= 12:
            return date(y, b, a), None
        if b > 12 and a <= 12:
            return date(y, a, b), None
        return None, f"ambiguous date {text!r} (day/month order undetermined) — left empty"
    return None, f"unreadable date {text!r} — left empty"


def _row_to_decision(
    raw: dict, row_number: int
) -> tuple[PastDecision | None, RejectedRow | None, RejectedRow | None]:
    """One parsed row -> (decision, rejection, warning). Exactly one of decision or
    rejection is non-None."""
    blob = json.dumps(raw, default=str)[:500]
    fields: dict[str, Any] = {}
    for key, value in raw.items():
        canonical = _canonical_header(str(key))
        if canonical and fields.get(canonical) in (None, ""):
            fields[canonical] = value

    company = str(fields.get("company") or "").strip()
    if not company:
        return None, RejectedRow(row_number=row_number, reason="no company name", raw=blob), None

    decision = _parse_decision_word(fields.get("decision"))
    if decision is None:
        supplied = str(fields.get("decision") or "").strip()
        reason = (
            f"unrecognised decision {supplied!r} — expected invested / passed / watched"
            if supplied
            else "no decision column (expected invested / passed / watched)"
        )
        return None, RejectedRow(row_number=row_number, reason=reason, raw=blob), None

    decided_on, date_warning = _parse_date(fields.get("date"))
    warning = (
        RejectedRow(row_number=row_number, reason=date_warning, raw=blob) if date_warning else None
    )

    def opt(key: str) -> str | None:
        value = str(fields.get(key) or "").strip()
        return value or None

    return (
        PastDecision(
            company=company,
            sector=(opt("sector") or "").lower() or None,
            stage=(opt("stage") or "").lower() or None,
            decision=decision,
            decided_on=decided_on,
            rationale=opt("rationale"),
            outcome=opt("outcome"),
            source_row=row_number,
        ),
        None,
        warning,
    )


def parse_decisions(content: bytes | str) -> tuple[list[PastDecision], DecisionUploadResult]:
    """Read a CSV or JSON decision history. Format is detected from the content itself
    rather than a filename, because an upload's extension is a claim, not a fact.

    Every row is accounted for: accepted, rejected with a reason, or accepted with a
    warning. `accepted + len(rejected) == total_rows` always holds.
    """
    text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
    stripped = text.strip()
    if not stripped:
        return [], DecisionUploadResult(accepted=0, rejected=[], total_rows=0)

    rows: list[dict]
    if stripped[0] in "[{":
        try:
            parsed = json.loads(stripped)
        except ValueError as exc:
            return [], DecisionUploadResult(
                accepted=0,
                total_rows=1,
                rejected=[RejectedRow(row_number=1, reason=f"invalid JSON: {exc}", raw=blob_of(stripped))],
            )
        if isinstance(parsed, dict):
            parsed = parsed.get("decisions") or parsed.get("rows") or [parsed]
        if not isinstance(parsed, list):
            return [], DecisionUploadResult(
                accepted=0,
                total_rows=1,
                rejected=[
                    RejectedRow(
                        row_number=1,
                        reason="JSON must be a list of decision objects, or an object "
                        "with a 'decisions' list",
                        raw=blob_of(stripped),
                    )
                ],
            )
        rows = [r if isinstance(r, dict) else {"_raw": r} for r in parsed]
    else:
        rows = list(csv.DictReader(io.StringIO(text)))

    decisions: list[PastDecision] = []
    rejected: list[RejectedRow] = []
    warnings: list[RejectedRow] = []
    for i, raw in enumerate(rows, start=1):
        clean = {k: v for k, v in raw.items() if k is not None}
        decision, rejection, warning = _row_to_decision(clean, i)
        if decision is not None:
            decisions.append(decision)
        if rejection is not None:
            rejected.append(rejection)
        if warning is not None:
            warnings.append(warning)

    return decisions, DecisionUploadResult(
        accepted=len(decisions), rejected=rejected, warnings=warnings, total_rows=len(rows)
    )


def blob_of(text: str) -> str:
    return text[:500]


# ---------------------------------------------------------------------------
# Derivation — stated side
# ---------------------------------------------------------------------------


def _answered(answers: list[SurveyAnswer]) -> list[tuple[SurveyAnswer, Any]]:
    """(answer, chosen option) pairs for answers to questions we actually know."""
    out = []
    for a in answers:
        question = SURVEY_BY_ID.get(a.question_id)
        if question is not None:
            out.append((a, question.option(a.choice)))
    return out


def _stated_axis_weights(answers: list[SurveyAnswer]) -> AxisWeights | None:
    """Sum the axis signals of the chosen options and normalise.

    Only ANSWERED questions contribute. An unanswered question is not treated as a
    neutral vote — it is simply absent, and it lowers the confidence instead.
    """
    pairs = _answered(answers)
    totals = {axis: 0.0 for axis in AXIS_SIGNALS}
    contributing: list[str] = []
    for answer, option in pairs:
        if any(option.signals.get(axis) for axis in AXIS_SIGNALS):
            contributing.append(answer.question_id)
        for axis in AXIS_SIGNALS:
            totals[axis] += float(option.signals.get(axis, 0.0))

    total = sum(totals.values())
    if not contributing or total <= 0:
        return None
    return AxisWeights(
        founder=round(totals["founder"] / total, 4),
        market=round(totals["market"] / total, 4),
        idea_vs_market=round(totals["idea_vs_market"] / total, 4),
        provenance=Provenance(
            basis="survey",
            method=(
                "summed the founder/market/idea_vs_market signals of each chosen option "
                "and normalised to sum to 1"
            ),
            question_ids=sorted(contributing),
            n=len(contributing),
        ),
        confidence=round(len(pairs) / len(SURVEY_QUESTIONS), 3),
    )


def _mean_signal(answers: list[SurveyAnswer], key: str) -> tuple[float, list[str]] | None:
    """Mean of one scalar signal over the answers that carry it, with the question ids."""
    values, ids = [], []
    for answer, option in _answered(answers):
        if key in option.signals:
            values.append(float(option.signals[key]))
            ids.append(answer.question_id)
    if not values:
        return None
    return sum(values) / len(values), sorted(ids)


def _capable(key: str) -> int:
    """How many questions could contribute to a scalar signal — the denominator for
    that signal's confidence."""
    return sum(1 for q in SURVEY_QUESTIONS if key in q.option_a.signals or key in q.option_b.signals)


def _conviction_label(score: float) -> str:
    if score <= -0.4:
        return "evidence-heavy"
    if score >= 0.4:
        return "conviction-heavy"
    return "balanced"


def _stated_conviction(answers: list[SurveyAnswer]) -> ConvictionStyle | None:
    result = _mean_signal(answers, "conviction")
    if result is None:
        return None
    score, ids = result
    return ConvictionStyle(
        score=round(max(-1.0, min(1.0, score)), 3),
        label=_conviction_label(score),
        provenance=Provenance(
            basis="survey",
            method=(
                "mean of the conviction signal (-1 evidence-heavy .. +1 conviction-heavy) "
                "over the answered questions that carry one"
            ),
            question_ids=ids,
            n=len(ids),
        ),
        confidence=round(len(ids) / max(1, _capable("conviction")), 3),
    )


# ---------------------------------------------------------------------------
# Derivation — revealed side
# ---------------------------------------------------------------------------

#: Where a stage sits on the evidence-available scale: -1 = almost nothing to inspect,
#: +1 = a fully evidenced business. This is the ONLY interpretation this module places
#: on submitted data, and it is stated here rather than buried so it can be argued with.
_STAGE_SCALE = {
    "pre-seed": -1.0,
    "preseed": -1.0,
    "pre seed": -1.0,
    "angel": -1.0,
    "seed": -0.5,
    "seed+": -0.25,
    "series a": 0.25,
    "a": 0.25,
    "series b": 0.75,
    "b": 0.75,
    "series c": 1.0,
    "c": 1.0,
    "series d": 1.0,
    "growth": 1.0,
    "late": 1.0,
}


def _stage_value(stage: str | None) -> float | None:
    return _STAGE_SCALE.get((stage or "").strip().lower())


def _invested(decisions: list[PastDecision]) -> list[PastDecision]:
    return [d for d in decisions if d.decision == DecisionKind.INVESTED]


def _priors(decisions: list[PastDecision], attr: str) -> list[Prior]:
    """Concentration over INVESTED rows only. A pass is not a preference for the thing
    passed on, so passes are excluded from a prior entirely."""
    invested = _invested(decisions)
    if len(invested) < MIN_INVESTED_FOR_PRIORS:
        return []
    counts: dict[str, list[int]] = {}
    for d in invested:
        key = (getattr(d, attr) or "").strip().lower()
        if not key:
            continue
        counts.setdefault(key, []).append(d.source_row or 0)
    total = sum(len(v) for v in counts.values())
    if not total:
        return []
    return sorted(
        (
            Prior(
                key=key,
                count=len(rows),
                share=round(len(rows) / total, 4),
                provenance=Provenance(
                    basis="decisions",
                    method=f"share of invested rows with this {attr}",
                    decision_rows=sorted(rows),
                    n=len(rows),
                ),
            )
            for key, rows in counts.items()
        ),
        key=lambda p: (-p.count, p.key),
    )


def _revealed_conviction(decisions: list[PastDecision]) -> ConvictionStyle | None:
    """Read off the STAGE of actual investments: committing earlier means committing
    with less evidence available, which is what conviction-heavy means operationally.

    Requires MIN_INVESTED_FOR_PRIORS investments with a recognisable stage. Below that
    it returns None and the caller records why — a conviction style read off one cheque
    is a number with the shape of a finding and none of the content.
    """
    scored = [
        (d, _stage_value(d.stage)) for d in _invested(decisions) if _stage_value(d.stage) is not None
    ]
    if len(scored) < MIN_INVESTED_FOR_PRIORS:
        return None
    mean_stage = sum(v for _, v in scored) / len(scored)
    score = max(-1.0, min(1.0, -mean_stage))
    invested_total = len(_invested(decisions))
    return ConvictionStyle(
        score=round(score, 3),
        label=_conviction_label(score),
        provenance=Provenance(
            basis="decisions",
            method=(
                "mean stage of actual investments on an evidence-available scale "
                "(pre-seed -1 .. growth +1), negated: investing earlier means committing "
                "with less to inspect"
            ),
            decision_rows=sorted(d.source_row or 0 for d, _ in scored),
            n=len(scored),
        ),
        # Discounted by the share of investments whose stage we could actually read, so
        # a history of mostly unrecognised stage labels reports low confidence.
        confidence=round(
            min(1.0, len(scored) / DECISIONS_FOR_FULL_CONFIDENCE)
            * (len(scored) / max(1, invested_total)),
            3,
        ),
    )


def _red_lines(decisions: list[PastDecision], stated: list[str]) -> list[RedLine]:
    """Stated red lines are taken at face value. Revealed ones are only ever raised as
    CANDIDATES, and only on a unanimous pass pattern over enough rows to mean something.

    We do not get to invent a VC's disqualifiers. A sector they passed on four times out
    of four is worth showing them; it is not worth asserting as a rule they hold.
    """
    out = [
        RedLine(
            statement=text,
            source="stated",
            provenance=Provenance(
                basis="profile_field", method="entered directly by the user", n=1
            ),
            confidence=1.0,
        )
        for text in stated
    ]

    for attr, noun in (("sector", "sector"), ("stage", "stage")):
        groups: dict[str, list[PastDecision]] = {}
        for d in decisions:
            key = (getattr(d, attr) or "").strip().lower()
            if key:
                groups.setdefault(key, []).append(d)
        for key, rows in sorted(groups.items()):
            if len(rows) < MIN_ROWS_FOR_RED_LINE_CANDIDATE:
                continue
            if any(d.decision != DecisionKind.PASSED for d in rows):
                continue
            out.append(
                RedLine(
                    statement=f"possible red line: passed on every {noun} == {key!r} seen so far",
                    source="revealed_candidate",
                    provenance=Provenance(
                        basis="decisions",
                        method=(
                            f"all {len(rows)} decisions with {noun} {key!r} were passes "
                            f"(threshold {MIN_ROWS_FOR_RED_LINE_CANDIDATE})"
                        ),
                        decision_rows=sorted(d.source_row or 0 for d in rows),
                        n=len(rows),
                    ),
                    # Grows with the run length but never reaches certainty: a unanimous
                    # pass streak is evidence of a pattern, not proof of a rule.
                    confidence=round(min(0.8, len(rows) / 10.0), 3),
                )
            )
    return out


# ---------------------------------------------------------------------------
# The whole derived profile
# ---------------------------------------------------------------------------


def derive(user_id: UUID) -> DerivedProfile:
    """Recompute the derived profile from the raw tables. Cheap, and deliberately not
    cached: a cached derivation is a merged blob with extra steps, and it would drift
    out of step with the two sources the gap analysis compares."""
    row = _profile_row(user_id)
    answers = get_survey(user_id)
    decisions = get_decisions(user_id)
    stated_red = _json_list(row.get("stated_red_lines")) if row else []

    not_inferred: list[NotInferred] = []
    axis_stated = _stated_axis_weights(answers)
    if axis_stated is None:
        not_inferred.append(
            NotInferred(
                field_name="axis_weights_stated",
                reason=(
                    "no survey answers carrying an axis signal were submitted — "
                    "weights would be invented, not derived"
                ),
            )
        )

    # THE ONE WE REFUSE TO DERIVE.
    #
    # Revealed axis weights would say how much this VC weighted founder vs market vs
    # idea-vs-market in decisions they actually made. Computing that needs a per-axis
    # score for each historical company, and an uploaded decision history contains no
    # such thing — it has a name, a sector, a stage, a verdict and free text. Any number
    # here would come from scoring companies we have never seen on evidence we do not
    # hold. So the field stays empty and says so, and the gap analysis compares the
    # dimensions where both sides genuinely have data instead.
    not_inferred.append(
        NotInferred(
            field_name="axis_weights_revealed",
            reason=(
                "an uploaded decision history carries no per-axis scores for the companies "
                "in it, so revealed axis weights cannot be computed from it without scoring "
                "companies we have no evidence about. Compare conviction style, stage and "
                "sector instead — those are present on both sides."
            ),
        )
    )

    conviction_stated = _stated_conviction(answers)
    if conviction_stated is None:
        not_inferred.append(
            NotInferred(
                field_name="conviction_style_stated",
                reason="no answered survey question carried a conviction signal",
            )
        )

    conviction_revealed = _revealed_conviction(decisions)
    if conviction_revealed is None:
        invested = len(_invested(decisions))
        recognised = sum(1 for d in _invested(decisions) if _stage_value(d.stage) is not None)
        not_inferred.append(
            NotInferred(
                field_name="conviction_style_revealed",
                reason=(
                    f"needs at least {MIN_INVESTED_FOR_PRIORS} investments with a recognisable "
                    f"stage; found {invested} investment(s), {recognised} with a readable stage"
                ),
            )
        )

    sector_priors = _priors(decisions, "sector")
    stage_priors = _priors(decisions, "stage")
    if not sector_priors:
        not_inferred.append(
            NotInferred(
                field_name="sector_priors",
                reason=(
                    f"needs at least {MIN_INVESTED_FOR_PRIORS} invested rows with a sector; "
                    f"found {len(_invested(decisions))} investment(s)"
                ),
            )
        )
    if not stage_priors:
        not_inferred.append(
            NotInferred(
                field_name="stage_priors",
                reason=(
                    f"needs at least {MIN_INVESTED_FOR_PRIORS} invested rows with a stage; "
                    f"found {len(_invested(decisions))} investment(s)"
                ),
            )
        )

    survey_confidence = len(answers) / len(SURVEY_QUESTIONS)
    decisions_confidence = min(1.0, len(decisions) / DECISIONS_FOR_FULL_CONFIDENCE)
    # Both halves count equally: the feature this profile exists to serve is the
    # comparison between them, and a profile strong on one side and empty on the other
    # cannot produce it. Averaging keeps a one-sided profile honestly below threshold.
    confidence = round((survey_confidence + decisions_confidence) / 2, 3)

    enabled = confidence >= PERSONALISATION_MIN_CONFIDENCE
    if enabled:
        reason = (
            f"profile confidence {confidence} at or above the "
            f"{PERSONALISATION_MIN_CONFIDENCE} threshold"
        )
    else:
        missing = []
        if len(answers) < len(SURVEY_QUESTIONS):
            missing.append(f"{len(SURVEY_QUESTIONS) - len(answers)} unanswered survey question(s)")
        if len(decisions) < DECISIONS_FOR_FULL_CONFIDENCE:
            missing.append(
                f"{len(decisions)} past decision(s) uploaded, "
                f"{DECISIONS_FOR_FULL_CONFIDENCE} for full confidence"
            )
        reason = (
            f"personalisation is OFF: profile confidence {confidence} is below the "
            f"{PERSONALISATION_MIN_CONFIDENCE} threshold ({'; '.join(missing)}). "
            "The core objective ranking is unaffected and continues to work."
        )

    return DerivedProfile(
        axis_weights_stated=axis_stated,
        axis_weights_revealed=None,
        conviction_style_stated=conviction_stated,
        conviction_style_revealed=conviction_revealed,
        sector_priors=sector_priors,
        stage_priors=stage_priors,
        red_lines=_red_lines(decisions, stated_red),
        survey_answered=len(answers),
        decisions_count=len(decisions),
        invested_count=len(_invested(decisions)),
        confidence=confidence,
        personalisation_enabled=enabled,
        personalisation_reason=reason,
        not_inferred=not_inferred,
    )


def get_profile(user_id: UUID) -> VCProfile:
    ensure_profile(user_id)
    row = _profile_row(user_id)
    assert row is not None  # ensure_profile just created it
    return VCProfile(
        profile_id=UUID(str(row["profile_id"])),
        user_id=user_id,
        fund_name=row.get("fund_name"),
        focus_sectors=_json_list(row.get("focus_sectors")),
        stated_red_lines=_json_list(row.get("stated_red_lines")),
        updated_at=db.from_iso(row["updated_at"]),
        derived=derive(user_id),
    )


# ---------------------------------------------------------------------------
# §2.3 — the stated-vs-revealed gap
# ---------------------------------------------------------------------------


def gap(user_id: UUID) -> GapReport:
    """Compare what the VC SAID against what the VC DID.

    Only comparable dimensions produce findings; anything with a missing side is listed
    in `uncomputable` naming which side was missing. A gap report that silently omits
    the dimensions it lacked data for reads as agreement, which would be a lie about the
    user rather than a finding about them.
    """
    profile = _profile_row(user_id)
    answers = get_survey(user_id)
    decisions = get_decisions(user_id)
    focus = _json_list(profile.get("focus_sectors")) if profile else []

    findings: list[GapFinding] = []
    uncomputable: list[GapUncomputable] = []
    agreements: list[str] = []

    _gap_conviction(answers, decisions, findings, uncomputable, agreements)
    _gap_stage(answers, decisions, findings, uncomputable, agreements)
    _gap_sector(focus, decisions, findings, uncomputable, agreements)

    return GapReport(
        findings=findings,
        uncomputable=uncomputable,
        agreements=agreements,
        computed_at=_now(),
    )


def _gap_conviction(answers, decisions, findings, uncomputable, agreements) -> None:
    stated = _stated_conviction(answers)
    revealed = _revealed_conviction(decisions)
    if stated is None or revealed is None:
        uncomputable.append(
            GapUncomputable(
                dimension="conviction_style",
                missing="stated" if stated is None else "revealed",
                reason=(
                    "no answered survey question carried a conviction signal"
                    if stated is None
                    else f"fewer than {MIN_INVESTED_FOR_PRIORS} investments with a readable stage"
                ),
            )
        )
        return

    magnitude = round(min(1.0, abs(stated.score - revealed.score) / 2.0), 3)
    confidence = round(min(stated.confidence, revealed.confidence), 3)
    if magnitude < GAP_REPORT_THRESHOLD:
        agreements.append(
            f"conviction style agrees: stated {stated.label} ({stated.score}), "
            f"revealed {revealed.label} ({revealed.score})"
        )
        return
    direction = "more" if revealed.score < stated.score else "less"
    findings.append(
        GapFinding(
            dimension="conviction_style",
            stated=f"{stated.label} ({stated.score})",
            revealed=f"{revealed.label} ({revealed.score})",
            finding=(
                f"You answered {stated.label} on {stated.provenance.n} trade-off(s), but the "
                f"{revealed.provenance.n} investment(s) you actually made are {direction} "
                f"evidence-driven than that — they cluster at "
                f"{'later' if direction == 'more' else 'earlier'} stages than your answers imply."
            ),
            magnitude=magnitude,
            provenance=Provenance(
                basis="survey+decisions",
                method=(
                    "stated conviction from survey signals vs revealed conviction from the "
                    "stage mix of actual investments; magnitude is the difference halved"
                ),
                question_ids=stated.provenance.question_ids,
                decision_rows=revealed.provenance.decision_rows,
                n=stated.provenance.n + revealed.provenance.n,
            ),
            confidence=confidence,
        )
    )


def _gap_stage(answers, decisions, findings, uncomputable, agreements) -> None:
    stated = _mean_signal(answers, "stage_lean")
    scored = [
        (d, _stage_value(d.stage)) for d in _invested(decisions) if _stage_value(d.stage) is not None
    ]
    if stated is None or len(scored) < MIN_INVESTED_FOR_PRIORS:
        uncomputable.append(
            GapUncomputable(
                dimension="stage",
                missing="stated" if stated is None else "revealed",
                reason=(
                    "no answered survey question carried a stage signal"
                    if stated is None
                    else f"fewer than {MIN_INVESTED_FOR_PRIORS} investments with a readable stage"
                ),
            )
        )
        return

    stated_lean, ids = stated
    revealed_lean = sum(v for _, v in scored) / len(scored)
    magnitude = round(min(1.0, abs(stated_lean - revealed_lean) / 2.0), 3)
    if magnitude < GAP_REPORT_THRESHOLD:
        agreements.append(
            f"stage appetite agrees: stated lean {round(stated_lean, 2)}, "
            f"revealed lean {round(revealed_lean, 2)} (-1 earliest .. +1 latest)"
        )
        return
    later = revealed_lean > stated_lean
    findings.append(
        GapFinding(
            dimension="stage",
            stated=f"lean {round(stated_lean, 2)} (-1 earliest .. +1 latest)",
            revealed=f"lean {round(revealed_lean, 2)} across {len(scored)} investment(s)",
            finding=(
                f"Your answers lean {'earlier' if later else 'later'} than your cheques. "
                f"You chose the {'earlier' if later else 'later'}-stage option on "
                f"{len(ids)} trade-off(s), but the {len(scored)} investment(s) you made sit "
                f"{'later' if later else 'earlier'} on the stage scale."
            ),
            magnitude=magnitude,
            provenance=Provenance(
                basis="survey+decisions",
                method=(
                    "mean stage_lean signal from the survey vs mean stage of actual "
                    "investments on the same -1..+1 scale"
                ),
                question_ids=ids,
                decision_rows=sorted(d.source_row or 0 for d, _ in scored),
                n=len(ids) + len(scored),
            ),
            # Both sides discount the finding, and the weaker one governs. The revealed
            # term was `len(scored) / len(scored)` — identically 1.0 — so a stage gap
            # drawn from six investments claimed full confidence while the conviction
            # and sector findings built on the SAME six rows correctly reported 0.3.
            confidence=round(
                min(
                    len(ids) / max(1, _capable("stage_lean")),
                    len(scored) / DECISIONS_FOR_FULL_CONFIDENCE,
                ),
                3,
            ),
        )
    )


def _gap_sector(focus, decisions, findings, uncomputable, agreements) -> None:
    """Stated focus sectors (typed into the profile) vs where the money actually went."""
    invested = _invested(decisions)
    with_sector = [d for d in invested if (d.sector or "").strip()]
    if not focus:
        uncomputable.append(
            GapUncomputable(
                dimension="sector",
                missing="stated",
                reason="no focus sectors set on the profile",
            )
        )
        return
    if len(with_sector) < MIN_INVESTED_FOR_PRIORS:
        uncomputable.append(
            GapUncomputable(
                dimension="sector",
                missing="revealed",
                reason=(
                    f"fewer than {MIN_INVESTED_FOR_PRIORS} investments carry a sector "
                    f"({len(with_sector)} found)"
                ),
            )
        )
        return

    focus_set = {s.strip().lower() for s in focus}
    outside = [d for d in with_sector if (d.sector or "").lower() not in focus_set]
    unfunded = sorted(
        s for s in focus_set if not any((d.sector or "").lower() == s for d in with_sector)
    )
    magnitude = round(len(outside) / len(with_sector), 3)

    if magnitude < GAP_REPORT_THRESHOLD and not unfunded:
        agreements.append(
            f"sector focus agrees: {len(with_sector) - len(outside)} of {len(with_sector)} "
            f"investments fall inside the stated focus {sorted(focus_set)}"
        )
        return

    parts = []
    if outside:
        parts.append(
            f"{len(outside)} of {len(with_sector)} investments are outside your stated focus "
            f"({', '.join(sorted({(d.sector or '') for d in outside}))})"
        )
    if unfunded:
        parts.append(f"you have never invested in {', '.join(unfunded)} despite listing it")
    findings.append(
        GapFinding(
            dimension="sector",
            stated=", ".join(sorted(focus_set)),
            revealed=", ".join(
                f"{p.key} ({p.count})" for p in _priors(decisions, "sector")
            )
            or "none",
            finding="; ".join(parts).capitalize() + ".",
            magnitude=max(magnitude, 0.25 if unfunded else 0.0),
            provenance=Provenance(
                basis="survey+decisions",
                method=(
                    "stated focus_sectors on the profile vs the sector of every invested row; "
                    "magnitude is the share of investments falling outside the stated focus"
                ),
                decision_rows=sorted(d.source_row or 0 for d in with_sector),
                n=len(with_sector),
            ),
            confidence=round(min(1.0, len(with_sector) / DECISIONS_FOR_FULL_CONFIDENCE), 3),
        )
    )
