"""The conductor: raw observations -> derived observations.

Every stage module was built to read the event log and write back to it. Nothing
ran the derivation, so scores sat at the prior (mu=0.5, band=0.5, zero receipts)
no matter how rich a founder's history was. This is that missing step.

  raw events (scanners, deck)  ->  derive()  ->  GREEN_FLAG + VALIDATION_RESULT
                                                        |
                                        score / screen / gate / memo read these

Idempotent: derived events get deterministic uuid5 ids, so re-running appends
nothing. That matters because the append-only log has no undo.
"""

from __future__ import annotations

import logging
from datetime import datetime
from uuid import UUID, uuid5

from schema.events import Event, EventKind, utcnow

log = logging.getLogger(__name__)

# Stable namespace for derived events. Changing it re-derives everything.
DERIVED_NS = UUID("d1f0a7c2-0000-4000-8000-000000000001")


def _supersedes_existing(ev: Event, prior: dict[tuple, set[str]]) -> bool:
    """Would this event replace a previously derived reading at the same point?"""
    key = (ev.entity_id, ev.payload.get("rule_id", "rollup"), ev.observed_at)
    seen = prior.get(key)
    return bool(seen) and _reading_digest(ev.payload) not in seen


def _reading_digest(payload: dict) -> str:
    """What the rule concluded, not merely that it ran.

    Only the fields that constitute the reading — the scalar and which rules fired.
    Prose or ordering changes must not mint a new event, or every re-derive would
    append duplicates at the same timestamp.
    """
    import hashlib
    import json as _json

    fired = payload.get("flags")
    material = {
        "value": payload.get("value"),
        "fired": sorted(
            (f.get("id"), bool(f.get("fired")))
            for f in fired
            if isinstance(f, dict) and f.get("id")
        )
        if isinstance(fired, list)
        else payload.get("fired"),
    }
    return hashlib.sha256(_json.dumps(material, sort_keys=True).encode()).hexdigest()[:12]


def _stable_id(*parts: object) -> UUID:
    return uuid5(DERIVED_NS, "|".join(str(p) for p in parts))


def derive(
    company_id: UUID,
    as_of: datetime | None = None,
    *,
    validate: bool = True,
) -> dict:
    """Run flag evaluation and claim validation, appending what's missing.

    validate=False skips the validator, which makes live search + LLM calls.
    Use it when seeding in bulk or offline; the scores still work, claims just
    stay NOT_ATTEMPTED — which the memo reports honestly rather than hiding.
    """
    from api.routers.deps import founder_entity_ids
    from memory import store

    cutoff = as_of or utcnow()
    existing = {e.event_id for e in store.events(as_of=cutoff, company_id=company_id)}
    entities = founder_entity_ids(company_id)
    for entity_id in entities:
        existing |= {e.event_id for e in store.events(as_of=cutoff, entity_id=entity_id)}

    appended = {"green_flag": 0, "validation_result": 0, "stale_rollups": 0}

    # (entity, rule_id, observed_at) -> the digests already stored, so a changed
    # reading at an already-derived point is detectable rather than invisible.
    prior_readings: dict[tuple, set[str]] = {}
    for entity_id in entities:
        for ev in store.events(as_of=cutoff, entity_id=entity_id, kind="green_flag"):
            key = (entity_id, ev.payload.get("rule_id", "rollup"), ev.observed_at)
            prior_readings.setdefault(key, set()).add(_reading_digest(ev.payload))

    # 1. Green flags -> the sensor readings the filter consumes.
    #
    # Evaluated at SUCCESSIVE checkpoints, not once at `cutoff`. flags.evaluate()
    # summarizes everything up to a date into one rollup, so a single call yields
    # a single observation — and a filter given one point has no trend to estimate
    # and accumulates process noise across the whole gap to `as_of` (we measured a
    # band of 8.96 on a 0..1 scale). A trajectory needs a series.
    for entity_id in entities:
        checkpoints = _checkpoints(entity_id, cutoff)
        for i, at in enumerate(checkpoints):
            is_last = i == len(checkpoints) - 1
            for ev in _evaluate_flags(entity_id, at):
                is_rollup = "value" in ev.payload
                # Per-rule receipts carry no scalar and exist for the trace, so
                # emit them once at the final checkpoint instead of at every one.
                if not is_rollup and not is_last:
                    continue
                ev.observed_at = at if is_rollup else ev.observed_at
                # The derived READING is part of the identity, not just the rule and
                # the date. Keying on (rule, entity, observed_at) alone meant a changed
                # rule produced an identical id, `not in existing` skipped it, and the
                # store silently kept the OLD value — so adding or fixing a rule did
                # not propagate without a full rebuild, and nothing said so. A new
                # reading is a new observation; that is what append-only means.
                ev.event_id = _stable_id(
                    "flag",
                    entity_id,
                    ev.payload.get("rule_id", "rollup"),
                    ev.observed_at,
                    _reading_digest(ev.payload),
                )
                if ev.event_id in existing:
                    continue

                # A rollup already at this (entity, rule, date) but with a DIFFERENT
                # reading means the rules changed since it was derived. Appending the
                # new one would double-count at a single world-time; skipping it
                # silently — which is what happened before the digest was part of the
                # id — meant a rule change never reached the store and nothing said so.
                # So: refuse both, and COUNT it. A stale derivation is a rebuild
                # instruction, not something to paper over.
                if _supersedes_existing(ev, prior_readings):
                    appended["stale_rollups"] = appended.get("stale_rollups", 0) + 1
                    continue

                ev.company_id = ev.company_id or company_id
                store.append(ev)
                existing.add(ev.event_id)
                appended["green_flag"] += 1

    # 2. Claim validation -> contradictions the filter must exclude.
    #
    # `cutoff` is passed through: without it check_claims defaults to now() and
    # validates a historical replay against present-day evidence, which is a
    # lookahead leak in the one artifact whose credibility rests on there being none.
    #
    # The validator persists its own VALIDATION_RESULT events, stamped with when the
    # evidence existed rather than when we ran. We do not write them here too — that
    # duplicated every verdict at the wrong timestamp.
    if validate:
        before = len(store.events(as_of=cutoff, company_id=company_id, kind="validation_result"))
        _check_claims(company_id, cutoff)
        after = len(store.events(as_of=cutoff, company_id=company_id, kind="validation_result"))
        appended["validation_result"] = max(0, after - before)

    return {"company_id": str(company_id), "entities": len(entities), "appended": appended}


MAX_CHECKPOINTS = 24


def _checkpoints(entity_id: UUID, as_of: datetime) -> list[datetime]:
    """Dates at which to re-read the sensor: month-ends where evidence exists.

    Monthly rather than per-event so a founder with 200 commits in one week
    contributes one reading, not 200 — volume must not masquerade as certainty.
    The final checkpoint is `as_of` itself so the band reflects any recent
    silence rather than stopping at the last thing that happened.
    """
    from memory import store

    raw = [
        e
        for e in store.events(as_of=as_of, entity_id=entity_id)
        if e.kind not in (EventKind.GREEN_FLAG, EventKind.VALIDATION_RESULT)
    ]
    if not raw:
        return []

    months = sorted({(e.observed_at.year, e.observed_at.month) for e in raw})
    if len(months) > MAX_CHECKPOINTS:
        step = len(months) / MAX_CHECKPOINTS
        months = [months[int(i * step)] for i in range(MAX_CHECKPOINTS)]

    out: list[datetime] = []
    for y, mo in months:
        end = datetime(y + (mo == 12), (mo % 12) + 1, 1, tzinfo=as_of.tzinfo)
        out.append(min(end, as_of))
    if out and out[-1] < as_of:
        out.append(as_of)
    return sorted(set(out))


def _evaluate_flags(entity_id: UUID, as_of: datetime) -> list[Event]:
    """A stage that raises must not take the whole pipeline with it."""
    try:
        from intelligence import flags

        return flags.evaluate(entity_id, as_of)
    except NotImplementedError:
        return []
    except Exception:  # noqa: BLE001 - one bad entity shouldn't stop the run
        log.warning("flag evaluation failed for %s", entity_id, exc_info=True)
        return []


def _check_claims(company_id: UUID, as_of: datetime) -> list:
    try:
        from intelligence import validator

        return validator.check_claims(company_id, as_of)
    except NotImplementedError:
        return []
    except Exception:  # noqa: BLE001 - validation needs network; never fatal
        log.warning("claim validation failed for %s", company_id, exc_info=True)
        return []


def derive_all(as_of: datetime | None = None, *, validate: bool = False) -> dict:
    """Derive across every company in the store. Defaults to validate=False —
    validating 13 companies makes a lot of live calls, so opt in deliberately."""
    from memory import store

    totals: dict[str, int] = {
        "companies": 0,
        "green_flag": 0,
        "validation_result": 0,
        "stale_rollups": 0,
    }
    for row in store.all_companies():
        cid = row.get("company_id")
        cid = UUID(cid) if isinstance(cid, str) else cid
        if not cid:
            continue
        out = derive(cid, as_of, validate=validate)
        totals["companies"] += 1
        for k, v in out["appended"].items():
            # setdefault, not indexing: derive() may report a counter this summary
            # does not know about, and a new diagnostic must never crash the run
            # that produced it.
            totals[k] = totals.setdefault(k, 0) + v
    return totals
