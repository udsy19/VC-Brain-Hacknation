"""Per-company routes: scorecard, trace, score history, memo, dissent, proof. Owner: D."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request, Response
from pydantic import BaseModel

from api import attest
from api import memo as memo_mod
from core import state
from schema.events import ClaimStatus, utcnow
from api.routers.deps import (
    as_uuid,
    degrade,
    find_seed_event,
    company_uuid,
    fixture_key,
    founder_entity_ids,
    pick,
    reset_screening_cache,
    resolve_as_of,
    seed,
    seed_or,
    viewer_scope,
)

router = APIRouter(prefix="/companies", tags=["companies"])

# THE DISSENT LOCK.
#
# Server-side, because a query flag the frontend sets is not a lock — it is a
# suggestion, and it is bypassable live on stage with a URL edit. A company is unlocked
# only when GET /companies/{id}/dissent actually served the anti-memo. The
# dissent_viewed request flag alone can NEVER unlock the recommendation.
#
# SCOPED PER VIEWER, AND STORED IN POSTGRES (migration 008). This was a module-level
# `set[str]`, which has the same weakness across lambdas that a client flag has across a
# URL bar. Serverless broke it in both directions: an unlock recorded on one lambda was
# invisible to the next request, so the recommendation never opened; and on a warm
# lambda one visitor's unlock opened that company for EVERY OTHER VISITOR, who then read
# the cheque figure having never been shown the case against it. The second is the one
# that matters — it is the exact failure the lock exists to prevent — and it is why the
# key is (scope, company) and never company alone. See deps.viewer_scope for how an
# anonymous browser gets a scope of its own.
#
# The in-process set REMAINS, as the fallback for when there is no viewer identity or no
# reachable database. That is a narrower guarantee, not a bypass: it still requires the
# server to have served a bear case, and it is exactly the behaviour local dev has today.
_DISSENT_SERVED: set[tuple[str, str]] = set()

#: The scope used when deps.viewer_scope cannot identify a viewer. Process-local by
#: construction — it is NEVER written to the database, because a shared row under a
#: shared key is precisely the cross-user leak this table was created to end.
_LOCAL_SCOPE = "local"


def dissent_was_served(company_id: str, request: Any = None) -> bool:
    """Has THIS viewer been served the bear case for this company?"""
    scope = _scope(request)
    if scope is not None:
        rows = state.fetch(
            "select 1 from dissent_unlocks where scope = ? and company_id = ?",
            (scope, company_id),
        )
        # None means the store could not answer. Fall through to process memory rather
        # than reporting "not unlocked" — a database blip must not re-lock a
        # recommendation the user is part-way through reading.
        if rows is not None:
            return bool(rows)
    return (_LOCAL_SCOPE if scope is None else scope, company_id) in _DISSENT_SERVED


def record_dissent_served(company_id: str, request: Any = None, response: Any = None) -> None:
    """Called ONLY from a route that actually rendered a bear case."""
    scope = _scope(request, response)
    if scope is not None and state.write(
        "insert into dissent_unlocks (scope, company_id, served_at) values (?, ?, ?) "
        "on conflict (scope, company_id) do nothing",
        (scope, company_id, utcnow().isoformat()),
    ):
        return
    _DISSENT_SERVED.add((_LOCAL_SCOPE if scope is None else scope, company_id))


def _scope(request: Any, response: Any = None) -> str | None:
    if request is None:
        return None
    try:
        return viewer_scope(request, response)
    except Exception:  # noqa: BLE001 - an unidentifiable viewer is the local fallback
        return None


def reset_dissent_locks() -> None:
    """Test/demo-reset hook. Not routed — there is deliberately no HTTP way to unlock."""
    _DISSENT_SERVED.clear()
    state.reset()
    reset_screening_cache()


@router.get("/{company_id}")
def get_company(company_id: str, as_of: datetime | None = None) -> dict:
    """Detail view. Accepts either a store UUID or a fixture slug — the ranked list
    hands out UUIDs, so resolving them is what makes the list clickable."""
    key = fixture_key(company_id)
    detail = seed_or(f"company_{key}", None)
    if detail is None:
        # Only the five stage archetypes have hand-authored detail fixtures. Build
        # the rest from the event log so every row in the ranked list is clickable —
        # a demo that dead-ends on two thirds of its own list is worse than a plain page.
        detail = _detail_from_store(company_id, key)
        if detail is None:
            raise HTTPException(404, f"unknown company: {company_id}")

    cutoff = resolve_as_of(as_of)
    detail = _normalize_detail(detail, company_id, cutoff)

    # Overlay the live score so the detail page shows what the filter actually
    # computed rather than a number frozen into the fixture.
    cid = company_uuid(company_id) or as_uuid(detail.get("company_id"))
    if cid:
        detail["company_id"] = str(cid)
        try:
            from intelligence import team as team_mod

            if founder_entity_ids(cid):
                # `ents[0]` used to be the whole Founder axis for this page, so a
                # co-founder carrying the technical signal was invisible here. The team
                # score aggregates every resolved founder; for a solo founder it is
                # byte-identical to that founder's own FounderScore, so no single-founder
                # company's numbers move. `founder_score` keeps its existing shape for
                # the dashboard; `team_score` carries the composition detail.
                ts = team_mod.team_score(cid, cutoff)
                detail["founder_score"] = {
                    "mu": ts.mu,
                    "band": ts.band,
                    "trend": ts.trend,
                    "contributing_event_ids": [str(i) for i in ts.contributing_event_ids],
                    "as_of": cutoff.isoformat(),
                }
                detail["team_score"] = team_mod.as_dict(ts)
        except Exception:  # noqa: BLE001 - a fixture detail page still beats a 500
            pass
    return detail


def _normalize_detail(detail: dict, company_id: str, cutoff) -> dict:
    """Emit the client's CompanyDetail contract instead of the fixture's own shape.

    The detail payload disagreed with the list on three things at once: axes came back
    0..1 where the list sends 0..100, `gate` was an object where the list sends a
    string, and `events[]` was absent entirely. The dashboard papered over all of it in
    app/lib/adapt.ts. An adapter that translates between two of our own endpoints is a
    bug with a shim on top, so the translation belongs here, once.
    """
    # _rescale_axis is THE axis serializer, shared with the ranked list. This module used
    # to carry a second, divergent copy: the two disagreed on null (the list's coalesced
    # an unmeasured axis to a confident 0.0) and on how `trend` was scaled. One function,
    # one answer, so the detail page and the list cannot drift apart again.
    from api.main import _ranked_row, _rescale_axis

    out = dict(detail)

    # Same builder as the ranked list, so the two endpoints cannot drift again.
    # Resolved to a UUID first: _ranked_row looks the founder up by id, and handing it
    # a slug silently produced no live axis, so the fixture's authored score won and
    # the detail page disagreed with the list it was reached from.
    cid = company_uuid(company_id)
    row = {
        "company_id": str(cid) if cid else out.get("company_id"),
        "name": out.get("name"),
        "archetype": out.get("archetype"),
    }
    try:
        # compute=True: the detail view is ONE company, so it can afford the screening
        # LLM calls the ranked list cannot, and it is the page where the receipts are
        # actually drilled into. It also warms deps.screening for the list behind it.
        summary = _ranked_row(row, cutoff, compute=True)
        out |= {k: summary[k] for k in summary if k not in ("axes",)}
        merged = dict(summary.get("axes") or {})
        for name, axis in (out.get("axes") or {}).items():
            if name not in merged and isinstance(axis, dict):
                merged[name] = _rescale_axis(axis)
        out["axes"] = merged
    except Exception:  # noqa: BLE001 - a detail page still beats a 500
        from api.main import _rescale_axis

        out["axes"] = {
            k: _rescale_axis(v) for k, v in (out.get("axes") or {}).items() if isinstance(v, dict)
        }

    # The engine wins here too. This used to unconditionally overwrite the computed gate
    # that _ranked_row had just resolved with the fixture's authored one — the same
    # substitution as the ranked list, one line later, so the detail page and the list
    # agreed with each other and both disagreed with the decision engine.
    gate = detail.get("gate")
    seeded_gate = gate.get("outcome") if isinstance(gate, dict) else gate
    if out.get("gate_source") != "computed":
        out["gate"] = seeded_gate or out.get("gate")
        out["gate_source"] = (
            "seeded_fixture" if seeded_gate else out.get("gate_source", "unavailable")
        )
        out["gate_rationale"] = gate.get("rationale") if isinstance(gate, dict) else None

    # Provenance is ASSIGNED, not defaulted: the detail payload can arrive from a
    # hand-authored fixture, and a fixture describing its own evidence as sourced is
    # the exact failure this field exists to catch. The store row is authoritative,
    # and an unknown company is treated as constructed rather than sourced.
    out["provenance"] = _provenance_for(company_id) or out.get("provenance") or "constructed"

    out.setdefault("events", _events_for(company_id, cutoff))
    out.setdefault("claims", [])
    out.setdefault("integrity", _integrity_for(company_id, cutoff))
    out.setdefault("proof_protocol", None)
    out.setdefault("entity_resolution_note", None)
    return out


def _events_for(company_id: str, cutoff) -> list[dict]:
    from memory import store

    cid = company_uuid(company_id)
    if cid is None:
        return []
    return [
        {
            "event_id": str(e.event_id),
            "kind": str(e.kind),
            "source": str(e.source),
            "observed_at": e.observed_at.isoformat(),
            "evidence_span": e.evidence_span,
            "quoted_span": e.evidence_span,
            "source_url": e.source_url,
            "confidence": e.confidence,
            "integrity_flags": e.integrity_flags,
        }
        for e in sorted(
            store.events(as_of=cutoff, company_id=cid), key=lambda e: e.observed_at, reverse=True
        )[:80]
    ]


def _integrity_for(company_id: str, cutoff) -> list[dict]:
    """Surfaced, never silent — a provenance note the founder should not be punished for."""
    return [
        {"flag": f, "event_id": e["event_id"], "evidence_span": e.get("quoted_span")}
        for e in _events_for(company_id, cutoff)
        for f in (e.get("integrity_flags") or [])
    ]


def _provenance_for(company_id: str) -> str | None:
    """'sourced' | 'constructed' from the store row, or None if it cannot be resolved.

    None means "do not know", and every caller treats that as constructed rather than
    sourced. Silence here must never read as a vouch for the evidence.
    """
    from memory import store

    cid = company_uuid(company_id)
    if cid is None:
        return None
    row = store.get_company(cid)
    return (row or {}).get("provenance")


def _detail_from_store(company_id: str, slug: str) -> dict | None:
    """A detail page assembled from the event log, for companies without a fixture."""
    from memory import store

    cid = company_uuid(company_id)
    if cid is None:
        return None
    row = store.get_company(cid)
    if not row:
        return None

    events = store.events(as_of=resolve_as_of(None), company_id=cid)
    return {
        "company_id": str(cid),
        "slug": slug,
        "name": row.get("name"),
        "archetype": row.get("archetype"),
        "provenance": row.get("provenance"),
        "source": "event_log",
        "event_count": len(events),
        "events": [
            {
                "event_id": str(e.event_id),
                "kind": str(e.kind),
                "source": str(e.source),
                "observed_at": e.observed_at.isoformat(),
                "quoted_span": e.evidence_span,
                "source_url": e.source_url,
                "integrity_flags": e.integrity_flags,
            }
            for e in sorted(events, key=lambda e: e.observed_at, reverse=True)[:60]
        ],
        "integrity_flags": sorted({f for e in events for f in e.integrity_flags}),
    }


@router.get("/{company_id}/trace/{event_id}")
def get_trace(company_id: str, event_id: str) -> dict:
    """Score -> contributing events -> source span -> original URL/slide id.

    Judges will click this. It bottoms out in `quoted_span` — a real span of text, a
    commit sha or a slide id — never merely a source name.
    """

    def live() -> dict:
        from memory import store

        eid, cid = as_uuid(event_id), company_uuid(company_id)
        if eid is None:
            raise HTTPException(400, "event_id is not a uuid")
        match = next(
            (
                e
                for e in store.events(as_of=resolve_as_of(None), company_id=cid)
                if e.event_id == eid
            ),
            None,
        )
        if match is None:
            raise LookupError(f"event {event_id} not in the store")

        contributing = None
        for ent in founder_entity_ids(cid) if cid else []:
            from memory import score as score_mod

            fs = score_mod.founder(ent, resolve_as_of(None))
            if eid in fs.contributing_event_ids:
                contributing = {
                    "entity_id": str(ent),
                    "mu": fs.mu,
                    "band": fs.band,
                    "trend": fs.trend,
                }
                break

        underlying = _underlying_evidence(match)
        return _trace_payload(
            company_id=company_id,
            event_id=event_id,
            kind=str(match.kind),
            source=str(match.source),
            # A rollup carries no url of its own; the commits it summarizes do.
            source_url=match.source_url
            or next((u["source_url"] for u in underlying if u["source_url"]), None),
            observed_at=match.observed_at.isoformat(),
            quoted_span=match.evidence_span,
            confidence=match.confidence,
            integrity_flags=match.integrity_flags,
            payload=match.payload,
            contributing_to=contributing,
            underlying_evidence=underlying,
            span_is_generated=bool((match.payload or {}).get("rollup")),
            degraded=False,
        )

    def fallback() -> dict:
        ev = find_seed_event(event_id)
        if ev is None:
            raise HTTPException(404, f"no trace for event {event_id}")
        return _trace_payload(
            company_id=company_id,
            event_id=event_id,
            kind=pick(ev, "kind", default="unknown"),
            source=pick(ev, "source", default="unknown"),
            source_url=pick(ev, "source_url", "url"),
            observed_at=pick(ev, "observed_at"),
            quoted_span=pick(ev, "evidence_span", "quoted_span", "span"),
            confidence=pick(ev, "confidence", default=1.0),
            integrity_flags=pick(ev, "integrity_flags", default=[]),
            payload=pick(ev, "payload", default={}),
            contributing_to=pick(ev, "contributing_to"),
            degraded=True,
        )

    return degrade(live, fallback)


def _underlying_evidence(event) -> list[dict]:
    """Follow `source_evidence_event_ids` to the observations a rollup summarizes.

    A green-flag rollup's own span is a sentence the system wrote ABOUT itself —
    "1/24 applicable green flags fired" — with no source_url. Ending the drill-down
    there presents a generated summary as the receipt, which is exactly what SHARED.md
    forbids. The rollup does record which real events it was computed from; this is
    that hop, and it lands on things like `commit 4b91e0c "pagekv: block table with
    refcounted physical pages"` with the GitHub URL attached.
    """
    from memory import store

    out: list[dict] = []
    for raw in (event.payload or {}).get("source_evidence_event_ids") or []:
        eid = as_uuid(raw)
        src = store.get_event(eid) if eid else None
        if src is None:
            continue
        out.append(
            {
                "event_id": str(src.event_id),
                "kind": str(src.kind),
                "source": str(src.source),
                "source_url": src.source_url,
                "quoted_span": src.evidence_span,
                "observed_at": src.observed_at.isoformat(),
            }
        )
    return out


def _trace_payload(
    *,
    quoted_span: str | None,
    underlying_evidence: list[dict] | None = None,
    span_is_generated: bool = False,
    **kw,
) -> dict:
    """The drill-down chain, rendered top to bottom by the UI.

    It must bottom out in a QUOTED SPAN — a real span of text, a commit sha or a slide
    id — never merely a source name and never a summary we generated ourselves.
    """
    underlying = underlying_evidence or []
    cited = [u for u in underlying if (u.get("quoted_span") or "").strip()]

    # The receipt is the event's own span unless that span is something we generated,
    # in which case the real receipts are the ones the rollup was computed from.
    receipt = None if span_is_generated else (quoted_span or None)
    if receipt is None and cited:
        receipt = cited[0]["quoted_span"]

    chain = [
        {"step": "score", "detail": kw.get("contributing_to") or "not a scoring observation"},
        {"step": "event", "detail": f"{kw['kind']} via {kw['source']} at {kw['observed_at']}"},
    ]
    if span_is_generated:
        # Named as a summary so it cannot be read as evidence in its own right.
        chain.append(
            {
                "step": "rollup summary",
                "detail": f"{quoted_span} — GENERATED BY THIS SYSTEM, not a receipt",
            }
        )
    chain.append(
        {
            "step": "source span",
            # No span means we cannot show a receipt — say so rather than showing the
            # source name and letting it read as evidence.
            "detail": receipt or "NO QUOTED SPAN STORED — this event is not citable evidence",
        }
    )
    chain.append({"step": "original", "detail": kw.get("source_url") or "no url on file"})

    return {
        **kw,
        "quoted_span": quoted_span,
        # Whether a REAL receipt exists, following the rollup hop. A generated summary
        # with nothing behind it is has_span: false — that is the honest answer.
        "has_span": bool(receipt),
        "span_is_generated": span_is_generated,
        "underlying_evidence": underlying,
        "chain": chain,
    }


@router.get("/{company_id}/score-history")
def get_score_history(company_id: str, as_of: datetime | None = None, points: int = 12) -> dict:
    """The moving score line + tightening band. Scores at successive dates — every point
    is a real filter run at that cutoff, not an interpolation of the final value."""
    cutoff = resolve_as_of(as_of)

    def live() -> dict:
        from memory import score as score_mod, store

        cid = company_uuid(company_id)
        ents = founder_entity_ids(cid) if cid else []
        if not ents:
            raise LookupError("no resolved founder entity for this company")
        entity_id = ents[0]

        events = store.events(as_of=cutoff, entity_id=entity_id)
        if not events:
            raise LookupError("no events to build a history from")

        start = min(e.observed_at for e in events)
        span = (cutoff - start) or timedelta(days=1)
        step = span / max(points - 1, 1)
        series = []
        for i in range(points):
            at = start + step * i
            fs = score_mod.founder(entity_id, at)
            series.append(
                {
                    "as_of": at.isoformat(),
                    "mu": fs.mu,
                    "band": fs.band,
                    "trend": fs.trend,
                    "event_count": len([e for e in events if e.observed_at <= at]),
                }
            )
        return {
            "company_id": company_id,
            "entity_id": str(entity_id),
            "series": series,
            "degraded": False,
        }

    def fallback() -> dict:
        fixture = seed_or(f"score_history_{company_id}", None)
        if fixture:
            return {**fixture, "degraded": True}
        company = seed_or(f"company_{company_id}", {})
        return {
            "company_id": company_id,
            "entity_id": pick(company, "entity_id"),
            "series": pick(company, "score_history", "series", default=[]),
            "degraded": True,
        }

    return degrade(live, fallback)


@router.get("/{company_id}/standout")
def get_standout(company_id: str, as_of: datetime | None = None, refresh: bool = False) -> dict:
    """What stood out about this company RELATIVE TO THE REST OF THE CORPUS.

    This is the explicit call that populates the cache the ranked list reads. It is on
    the detail path, like the memo and the screening, for the same reason: it costs one
    LLM round-trip, and thirteen of those inline would be a 90-second list page.

    `refresh=true` rebuilds the corpus frame and regenerates. Nothing else invalidates:
    the cache key already carries this company's evidence digest AND the corpus digest,
    so new evidence — here or on any company it is compared against — produces a miss
    on its own.
    """
    from api import standout

    return degrade(
        lambda: standout.generate(company_id, resolve_as_of(as_of), refresh=refresh),
        lambda: standout.not_generated(company_id),
    )


@router.get("/{company_id}/memo")
def get_memo(
    company_id: str,
    request: Request,
    as_of: datetime | None = None,
    dissent_viewed: bool = False,
) -> dict:
    """Recommendation stays null until the dissent has actually been served.

    Enforced here against server state. `dissent_viewed=true` on its own does nothing —
    that flag is a UI hint, and trusting it would make the lock decorative.
    """
    cutoff = resolve_as_of(as_of)
    result = degrade(
        lambda: memo_mod.generate_memo(company_id, cutoff),
        lambda: seed(f"memo_{company_id}"),
    )

    # The live path emits every declared section; the seed passthrough returns whatever
    # the fixture happens to carry. That made a section's PRESENCE depend on which path
    # served the request, so a client reading body["founder"] worked live and raised on
    # the fallback. Missing sections are filled as explicitly empty — never invented —
    # so the shape is stable and an unwritten section still reads as unwritten.
    # A section we had to fill is a GAP, and it has to say so. Filling the shape while
    # leaving `gaps` empty is precisely the "memo that fabricates completeness" this
    # endpoint exists to prevent — the reader would see five headings and no admission
    # that three of them were never written.
    unwritten = [name for name in memo_mod.SECTIONS if name not in result]
    for name in unwritten:
        result[name] = {"summary": None, "not_written": True}
    if unwritten:
        gaps = result.get("gaps")
        result["gaps"] = (gaps if isinstance(gaps, list) else []) + [
            {
                "claim": f"the {name} section",
                "source_span": "",
                "status": str(ClaimStatus.NOT_ATTEMPTED),
                "why": "this section was never written for this company",
            }
            for name in unwritten
        ]

    # Both must hold: the server must have served the anti-memo, AND the client must
    # say it rendered it. The client half alone is not sufficient — that is the lock.
    #
    # `investment_recommendation` carries the cheque figure, so it is locked by the SAME
    # gate as the prose section — it is in fact the thing the lock exists to protect, and
    # leaving it readable while nulling the prose would make the lock decorative.
    if not (dissent_was_served(company_id, request) and dissent_viewed):
        result["recommendation"] = None
        result["investment_recommendation"] = None
        result["recommendation_locked_reason"] = "open the dissent view first"
    return result


@router.get("/{company_id}/dissent")
def get_dissent(
    company_id: str, request: Request, response: Response, as_of: datetime | None = None
) -> dict:
    """Serving the anti-memo is the ONLY thing that unlocks the recommendation."""
    cutoff = resolve_as_of(as_of)

    def live() -> dict:
        from intelligence import dissent

        cid = company_uuid(company_id)

        # Prefer C's combined dissent/council view when it exists — it is the
        # richer artifact. Falls back to the plain anti-memo, so this route keeps
        # working whether or not the council has shipped.
        combined = _council_view(cid, cutoff)
        if combined is not None:
            return {"company_id": company_id, **combined, "degraded": False}

        anti = dissent.generate(cid, cutoff)
        out = {
            "company_id": company_id,
            "bear_case": anti.bear_case,
            "weakest_evidence": anti.weakest_evidence,
            "load_bearing_claim": anti.load_bearing_claim,
            "axis_spreads": anti.axis_spreads,
            "degraded": False,
        }
        try:
            out["uncertainty"] = dissent.uncertainty_from_spread(anti)
        except (NotImplementedError, AttributeError):
            pass
        return out

    result = degrade(live, lambda: seed(f"dissent_{company_id}"))
    # Same substance test as the council route: reaching this line means SOMETHING
    # rendered, but a payload carrying no bear case has not shown anyone a dissent.
    if _rendered_bear_case(result):
        record_dissent_served(company_id, request, response)
    else:
        result["recommendation_locked_reason"] = (
            "no bear case could be produced for this company, so the recommendation "
            "stays locked"
        )
    return result


def _council_view(cid, cutoff) -> dict | None:
    """intelligence.council.view_dissent, when C has shipped it."""
    try:
        from intelligence import council
    except ImportError:
        return None
    try:
        view = council.view_dissent(cid, cutoff)
    except (AttributeError, NotImplementedError):
        return None
    return view if isinstance(view, dict) else view.model_dump(mode="json")


@router.post("/{company_id}/council")
def run_council(
    company_id: str, request: Request, response: Response, as_of: datetime | None = None
) -> dict:
    """Run C's AI Council. Like the dissent view, actually serving a council
    deliberation is what unlocks the recommendation — never a client boolean."""
    cutoff = resolve_as_of(as_of)

    def live() -> dict:
        from intelligence import council

        out = council.deliberate(company_uuid(company_id), cutoff)
        out = out if isinstance(out, dict) else out.model_dump(mode="json")
        return {"company_id": company_id, **out, "degraded": False}

    def fallback() -> dict:
        fixture = seed_or(f"council_{company_id}", None)
        if fixture is None:
            raise HTTPException(503, "the AI Council is not available yet")
        return {**fixture, "company_id": company_id, "degraded": True}

    result = degrade(live, fallback)
    # ONLY a council that actually argued a bear case unlocks the recommendation.
    #
    # council.deliberate() returns decision=None / anti_memo=None BY DESIGN — it is the
    # locked view, and its own payload says "open the dissent view first". Unlocking on
    # it unconditionally, which is what this did, meant the endpoint that represents the
    # lock was the one endpoint that bypassed it: POST /council then
    # GET /memo?dissent_viewed=true returned a recommendation having shown no dissent at
    # all. The lock is the product; this is the hole in it.
    if _rendered_bear_case(result):
        record_dissent_served(company_id, request, response)
    else:
        result["recommendation_locked_reason"] = (
            "this council returned no anti-memo, so it does not count as dissent — "
            "open the dissent view"
        )
    return result


def _rendered_bear_case(payload: dict) -> bool:
    """Did this payload actually put a bear case in front of the viewer?

    Substance, not shape: an anti_memo key holding None, or a bear_case holding an
    empty string, is exactly the empty deliberation that must not unlock anything.
    """
    if not isinstance(payload, dict):
        return False
    anti = payload.get("anti_memo")
    if isinstance(anti, dict) and str(anti.get("bear_case") or "").strip():
        return True
    return bool(str(payload.get("bear_case") or "").strip())


class ProofSubmission(BaseModel):
    artifact: str = ""
    trace: dict = {}
    demo: bool = False  # the seeded stage path
    # A public repo lets the server read the commit history itself instead of
    # taking the submitter's word for it. Optional, and the strongest attestation
    # available — see api/attest.py.
    repo_url: str | None = None


@router.post("/{company_id}/proof")
def issue_proof(company_id: str) -> dict:
    """Issue a challenge: one ambiguous requirement, one planted bad constraint."""

    def live() -> dict:
        from intelligence import proof

        ch = proof.generate(company_uuid(company_id))
        # The server's own record of when this went out. Without it a submitted
        # trace cannot be placed in time except by trusting the submitter.
        attest.record_issue(str(ch.challenge_id), ch.issued_at, str(company_uuid(company_id)))
        return {
            "challenge_id": str(ch.challenge_id),
            "company_id": company_id,
            "prompt": ch.prompt,
            "central_claim": ch.central_claim,
            "ambiguous_requirement": ch.ambiguous_requirement,
            "planted_bad_constraint": ch.planted_bad_constraint,
            "issued_at": ch.issued_at.isoformat(),
            "degraded": False,
        }

    def fallback() -> dict:
        fixture = seed_or(f"challenge_{company_id}", None) or seed_or("challenge", None)
        if fixture is None:
            raise HTTPException(503, "proof protocol unavailable and no challenge fixture seeded")
        return {**fixture, "company_id": company_id, "degraded": True}

    return degrade(live, fallback)


@router.post("/{company_id}/proof/{challenge_id}/grade")
def grade_proof(
    company_id: str, challenge_id: str, submission: ProofSubmission = Body(default=None)
) -> dict:
    """Grade a submission. Graded events are appended, so the score visibly moves —
    that re-entry into the gate is the demo."""
    sub = submission or ProofSubmission()

    def live() -> dict:
        from intelligence import proof
        from memory import store

        cid = company_uuid(company_id)

        # A challenge is written against ONE company's central technical claim.
        # Grading it onto another company's founder score would let a submission
        # for an easy challenge inflate an unrelated founder. Rejected before any
        # event is appended, because the log has no undo.
        if attest.challenge_belongs_to(challenge_id, cid) is False:
            raise HTTPException(
                409,
                f"challenge {challenge_id} was not issued for company {company_id}",
            )

        # Split the trace into what we observed and what we were told, BEFORE
        # grading. Pushing back on the planted constraint is worth half the
        # behavioural score, so accepting it on the client's word would hand the
        # sharpest signal in the system to anyone willing to assert it. The
        # attestation rides inside the trace so the grader can weight
        # self-reported behaviour down at scoring time, not merely afterwards.
        graded_trace, attestation = attest.attest(
            challenge_id, sub.trace, repo_url=sub.repo_url, demo=sub.demo
        )

        artifact, trace, cid_for_grade = sub.artifact, graded_trace, challenge_id
        if sub.demo:
            # seed_demo_completion returns the pre-run artifact + trace, not events.
            # Running it through the real grader is both more honest for the demo
            # and what keeps the seeded path on the same code path as a live one.
            seeded = proof.seed_demo_completion(cid)
            if isinstance(seeded, dict):
                artifact = seeded.get("artifact", artifact)
                trace = {**seeded.get("trace", {}), "attestation": attestation}
                cid_for_grade = seeded.get("challenge_id", challenge_id)
            else:
                events = attest.apply(list(seeded or []), attestation)
                for ev in events:
                    store.append(ev)
                return {
                    "company_id": company_id,
                    "challenge_id": challenge_id,
                    "graded_event_ids": [str(e.event_id) for e in events],
                    "attestation": attestation,
                    "degraded": False,
                }

        events = proof.grade(as_uuid(cid_for_grade), artifact, trace)
        events = attest.apply(events, attestation)
        appended = []
        for ev in events or []:
            store.append(ev)
            appended.append(str(ev.event_id))
        return {
            "company_id": company_id,
            "challenge_id": challenge_id,
            "graded_event_ids": appended,
            "attestation": attestation,
            "degraded": False,
        }

    def fallback() -> dict:
        fixture = seed_or(f"proof_result_{company_id}", None) or seed_or("proof_result", None)
        if fixture is None:
            # Say which of the two things went wrong. An unknown challenge is a
            # client error — we cannot grade a submission for a challenge we never
            # issued, and pretending otherwise would grade an unanchored trace.
            if not attest.issued_at(challenge_id):
                raise HTTPException(
                    422,
                    f"challenge {challenge_id} was not issued by this server, so the "
                    "submitted trace cannot be anchored in time or graded",
                )
            raise HTTPException(503, "grading unavailable and no proof fixture seeded")
        # The degraded path must still say what it is. Returning a graded-looking
        # result with no attestation block is how an unverified trace ends up read
        # as an observed one.
        _, attestation = attest.attest(
            challenge_id, sub.trace, repo_url=sub.repo_url, demo=sub.demo
        )
        return {
            **fixture,
            "company_id": company_id,
            "challenge_id": challenge_id,
            "attestation": attestation,
            "degraded": True,
        }

    return degrade(live, fallback)
