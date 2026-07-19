"""Shared helpers for the routers. Owner: D.

Two jobs: load seed fixtures, and degrade gracefully when a teammate's module still
raises NotImplementedError. The app must ALWAYS run — a dead route at hour 23 is a
dead demo beat, so every live call has a fixture behind it.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, TypeVar
from uuid import UUID

from fastapi import HTTPException

log = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")


def seed_dir() -> Path:
    """VCBRAIN_SEED_DIR wins so tests can point at a tmp fixture set."""
    return Path(os.getenv("VCBRAIN_SEED_DIR", "data/seed"))


def seed(name: str) -> dict:
    path = seed_dir() / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, f"no seed fixture: {name}")
    return json.loads(path.read_text())


def seed_or(name: str, default: Any) -> Any:
    path = seed_dir() / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else default


def fixture_key(company_id: str) -> str:
    """Map a store UUID back to its fixture slug.

    /companies returns real UUIDs from the store, but the detail/memo/dissent
    fixtures are named by slug (company_vb-tensorpage.json). Without this the
    primary navigation — click a row in the ranked list — 404s on every company.
    Passes non-UUID input straight through, so slugs still work directly.

    Resolved through the company NAME rather than the generated _resolved_ids.json:
    UUIDs are minted fresh whenever the database is rebuilt, so any file mapping
    slug->uuid is stale the moment someone reseeds. Names are stable.
    """
    cid = as_uuid(company_id)
    if cid is None:
        return company_id

    from memory import store

    row = store.get_company(cid) or {}
    name = row.get("name")
    if not name:
        return company_id
    return _slug_by_name().get(name, company_id)


def company_uuid(company_id: str) -> UUID | None:
    """Accept either a UUID or a fixture slug and return the store UUID.

    The ranked list hands the client slugs as `id`, so every link the UI builds
    arrives here as a slug. Resolving only UUIDs 404s the entire navigation.
    """
    cid = as_uuid(company_id)
    if cid is not None:
        return cid

    name = {slug: n for n, slug in _slug_by_name().items()}.get(company_id)
    if not name:
        return None

    from memory import store

    for row in store.all_companies():
        if row.get("name") == name:
            return as_uuid(row.get("company_id"))
    return None


@lru_cache(maxsize=1)
def prior_company_names() -> frozenset[str]:
    """Companies a serial founder ran BEFORE the current one — history, not pipeline."""
    out: set[str] = set()
    for path in sorted(seed_dir().glob("archetype_*.json")):
        for profile in json.loads(path.read_text()).get("profiles", []):
            out |= {p["name"] for p in profile.get("prior_companies", [])}
    return frozenset(out)


@lru_cache(maxsize=1)
def backtest_cohort_names() -> frozenset[str]:
    """Companies that exist only to be REPLAYED, never to be invested in.

    The cohort became real entities with real events so the backtest could actually
    score them — which is what makes it a replay rather than a retelling. The side
    effect was that Docker, Supabase, Hugging Face and Vercel took ranks 1-4 of the
    investable pipeline: companies that broke out a decade ago, offered as deals to
    write a cheque into. Their events still feed the replay; they are not
    opportunities. Same reasoning as a serial founder's prior company.
    """
    blob = seed_or("backtest", {})
    out: set[str] = set()
    for key in ("cohort", "winners", "controls", "failures"):
        for m in blob.get(key) or []:
            for field in ("company_name", "name", "founder"):
                if isinstance(m.get(field), str):
                    out.add(m[field])
    failure = blob.get("correctly_deprioritized_failure")
    if isinstance(failure, dict):
        out |= {failure[f] for f in ("company_name", "name") if isinstance(failure.get(f), str)}
    return frozenset(out)


@lru_cache(maxsize=1)
def _slug_by_name() -> dict[str, str]:
    """company_name -> fixture slug, read from the archetype fixtures themselves."""
    out: dict[str, str] = {}
    for path in sorted(seed_dir().glob("archetype_*.json")):
        for profile in json.loads(path.read_text()).get("profiles", []):
            out[profile["company_name"]] = profile["company_id"]
            for prior in profile.get("prior_companies", []):
                out[prior["name"]] = prior["company_id"]
    return out


# Computed three-axis screenings, memoized per (company, as_of hour).
#
# market and idea_vs_market each cost an LLM call (~7s per company measured), so the
# ranked list must NEVER compute them inline — 13 companies is a 95s page load. The
# list reads whatever a detail view already computed and otherwise honestly reports no
# receipts; the detail view pays the cost once, for one company, and warms this.
_SCREENING: dict[tuple[str, str], Any] = {}


def _screening_bucket(as_of: datetime) -> str:
    """as_of defaults to now(), so an exact key would never hit. Bucketed to the hour:
    coarse enough that list-after-detail shares an entry, fine enough that a deliberate
    historical as_of query does not silently read a screening from a different era."""
    return as_of.astimezone(timezone.utc).strftime("%Y-%m-%dT%H")


def screening(company_id: UUID, as_of: datetime, *, compute: bool = False) -> Any | None:
    """The computed screening for a company, or None when we do not have one.

    None is a real answer and callers must render it as "no receipts computed", never
    as zero and never as a padded placeholder list.
    """
    key = (str(company_id), _screening_bucket(as_of))
    if key in _SCREENING:
        return _SCREENING[key]
    if not compute:
        return None
    try:
        from intelligence import screen

        result = screen.three_axis(company_id, as_of)
    except Exception as exc:  # noqa: BLE001 - an unscreened company still renders
        log.info("screening unavailable (%s): %s", type(exc).__name__, exc)
        return None
    _SCREENING[key] = result
    return result


def reset_screening_cache() -> None:
    """Test/demo-reset hook, mirroring reset_dissent_locks."""
    _SCREENING.clear()


# ---------------------------------------------------------------------------
# VIEWER SCOPE — who is asking, for the purpose of the dissent lock.
#
# The storage behind the lock is core.state (migration 008). What lives here is the
# only part that is HTTP: reading cookies to decide whose lock it is.
# ---------------------------------------------------------------------------

#: Anonymous viewer identity. Distinct from the auth session cookie: this one confers no
#: privilege at all and is never a credential — it exists ONLY so that "this browser has
#: been shown the bear case" is a statement about ONE browser.
VIEWER_COOKIE = "vcbrain_viewer"


def viewer_scope(request: Any, response: Any = None) -> str | None:
    """The scope the dissent lock is keyed by, or None when there is no per-viewer identity.

    Three cases, in order:

    1. SIGNED IN -> "user:<uuid>". The strongest scope available, and the one that makes
       the lock follow the person across browsers.
    2. ANONYMOUS WITH A VIEWER COOKIE -> "viewer:<sha256 of the token>". The demo is
       driven logged-out, so this is the path that actually runs on stage. The token is
       minted on the first request that unlocks something and is hashed before storage.
    3. NO IDENTITY AT ALL -> None, and the caller falls back to process memory.

    Case 3 is not a loophole, it is the honest floor. It happens for a non-browser client
    (curl, scripts/verify_demo.py) and for local dev, where the dashboard calls
    http://localhost:8000 cross-origin and fetch's default `credentials: "same-origin"`
    means no cookie is sent. Local behaviour is therefore EXACTLY what it was before this
    change — one process, one shared set. In production the frontend is same-origin
    behind the /api rewrite, so a browser always reaches case 2.

    `response` is passed only on the paths that RECORD an unlock. A read must never mint
    an identity: doing so would hand every memo request a fresh scope and the lock would
    read as permanently closed.
    """
    from api import auth

    user = auth.optional_user(request)  # documented never to raise
    if user is not None:
        return f"user:{user.user_id}"

    token = request.cookies.get(VIEWER_COOKIE)
    if token:
        return f"viewer:{auth.token_hash(token)}"
    if response is None:
        return None

    token = auth.new_token()
    response.set_cookie(
        VIEWER_COOKIE,
        token,
        httponly=True,
        samesite="lax",
        secure=auth._secure_cookies(),
        max_age=60 * 60 * 24 * 7,
        path="/",
    )
    return f"viewer:{auth.token_hash(token)}"


def degrade(live: Callable[[], T], fallback: Callable[[], U]) -> T | U:
    """Run the real module; fall back to fixtures on anything short of a 4xx.

    NotImplementedError is the expected case while branches A/B/C are still landing.
    """
    try:
        return live()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - degrading is the whole point
        log.info("degraded to fixture (%s): %s", type(exc).__name__, exc)
        return fallback()


def as_uuid(value: str | UUID | None) -> UUID | None:
    if value is None or isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


def now() -> datetime:
    return datetime.now(timezone.utc)


def resolve_as_of(as_of: datetime | None) -> datetime:
    """Naive datetimes silently break as_of comparisons — normalize at the boundary."""
    if as_of is None:
        return now()
    return as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)


def pick(d: dict, *keys: str, default: Any = None) -> Any:
    """Tolerant accessor — fixture key names are still settling across branches."""
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return default


def founder_entity_ids(company_id: UUID) -> list[UUID]:
    """Founder entities for a company, via the store; empty when unresolvable."""
    from memory import store

    row = store.get_company(company_id) or {}
    raw = row.get("founder_entity_ids") or "[]"
    ids = json.loads(raw) if isinstance(raw, str) else raw
    resolved = [u for u in (as_uuid(i) for i in ids) if u is not None]
    if resolved:
        return resolved

    # Fall back to the event log. The column is a denormalized convenience that
    # ingestion does not always populate, but events always carry entity_id —
    # so the log is the authority. Without this every founder scores at the
    # prior, because the scorer is never handed an entity to score.
    from schema.events import utcnow

    seen: dict[UUID, None] = {}
    for e in store.events(as_of=utcnow(), company_id=company_id):
        if e.entity_id is not None:
            seen.setdefault(e.entity_id, None)
    return list(seen)


@lru_cache(maxsize=1)
def _seed_files_cached(dir_key: str, stamp: float) -> tuple[dict, ...]:
    return tuple(json.loads(p.read_text()) for p in sorted(Path(dir_key).glob("*.json")))


def all_seed_blobs() -> list[dict]:
    d = seed_dir()
    if not d.exists():
        return []
    stamp = max((p.stat().st_mtime for p in d.glob("*.json")), default=0.0)
    return list(_seed_files_cached(str(d), stamp))


def find_seed_event(event_id: str) -> dict | None:
    """Scan every fixture for an event with this id. Fixture shapes are still moving,
    so walk the whole structure rather than assuming a key."""

    def walk(node: Any) -> dict | None:
        if isinstance(node, dict):
            if str(node.get("event_id", "")) == event_id:
                return node
            for v in node.values():
                if (hit := walk(v)) is not None:
                    return hit
        elif isinstance(node, list):
            for v in node:
                if (hit := walk(v)) is not None:
                    return hit
        return None

    for blob in all_seed_blobs():
        if (hit := walk(blob)) is not None:
            return hit
    return None
