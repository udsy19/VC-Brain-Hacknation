"""Time-machine backtest. Owner: D, with A + C. Proof #1 of the whole pitch.

Replays truncated historical sources through the SAME code path as live, with as_of
pinned before the founder was known. If it needs a special mode, it isn't a backtest —
so replay() calls the same score / screen / gate / memo functions the API calls, and
there is no `backtest=True` flag anywhere in this file.

assert_no_lookahead() is what makes the claim credible rather than merely asserted. It
runs before every scoring step and it raises. It never warns.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from backtest import collect
from schema.events import Event

log = logging.getLogger(__name__)


class LookaheadError(AssertionError):
    """Raised loudly. Never caught, never downgraded to a warning."""


def assert_no_lookahead(events: list[Event], as_of: datetime) -> int:
    """Raise if any event postdates as_of. Returns how many events were checked.

    The return value is not decoration. `lookahead_checked: True` used to be a
    literal in the report — a claim that the assertion had run, written by hand,
    in the one artifact whose entire job is proving the system does not fool
    itself. Callers now count what this function actually saw, so the report says
    "checked N events" or says it did not run. A hardcoded True is worse than no
    field at all.
    """
    leaked = [e for e in events if e.observed_at > as_of]
    if leaked:
        raise LookaheadError(
            f"{len(leaked)} event(s) from the future reached the scorer at as_of={as_of}: "
            f"{[str(e.event_id) for e in leaked[:3]]}"
        )
    return len(events)


def _aware(v: Any) -> datetime:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(str(v))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _as_uuid(v: Any) -> UUID | None:
    if v is None or isinstance(v, UUID):
        return v
    try:
        return UUID(str(v))
    except (ValueError, TypeError):
        return None


def replay(company_id: Any, as_of: datetime) -> dict:
    """ingest -> score -> screen -> gate -> memo, at a cutoff before the founder was known.

    Identical call sequence to the live API path. Every stage that can be unimplemented
    degrades to None, but the LOOKAHEAD ASSERTION never degrades.
    """
    from api import memo as memo_mod
    from api.routers.deps import founder_entity_ids
    from memory import score as score_mod, store

    cutoff = _aware(as_of)
    cid = _as_uuid(company_id)

    # 1. ingest (read side): the as_of-scoped event set, checked before it goes anywhere.
    events = store.events(as_of=cutoff, company_id=cid)
    events_checked = assert_no_lookahead(events, cutoff)

    # 2. score, per founder entity — assertion repeated on the exact list the scorer sees.
    scores = []
    for entity_id in founder_entity_ids(cid) if cid else []:
        entity_events = store.events(as_of=cutoff, entity_id=entity_id)
        events_checked += assert_no_lookahead(entity_events, cutoff)
        fs = score_mod.founder(entity_id, cutoff)
        scores.append(
            {
                "entity_id": str(entity_id),
                "mu": fs.mu,
                "band": fs.band,
                "trend": fs.trend,
                "contributing_event_ids": [str(i) for i in fs.contributing_event_ids],
            }
        )

    # 3-4. screen + gate.
    screening = _stage(lambda: _screen(cid, cutoff), "screen")
    gate = _stage(lambda: _gate(cid, cutoff), "gate")

    # 5. memo — the same generator the API serves.
    memo = _stage(lambda: memo_mod.generate_memo(cid, cutoff), "memo")

    return {
        "company_id": str(company_id),
        "as_of": cutoff.isoformat(),
        "event_count": len(events),
        "scores": scores,
        "screening": screening,
        "gate": gate,
        "memo": memo,
        # Reported because it happened, not because it is expected to: the assertion
        # above ran over exactly this many events and did not raise.
        "lookahead_checked": True,
        "lookahead_events_checked": events_checked,
    }


def _stage(fn, name: str) -> Any:
    try:
        return fn()
    except LookaheadError:
        raise  # never swallowed — this is the one exception that must escape
    except Exception as exc:  # noqa: BLE001 - a stage still in progress must not stop the replay
        log.info("replay: stage %s unavailable (%s)", name, exc)
        return None


def _screen(cid: UUID | None, cutoff: datetime) -> dict | None:
    from intelligence import screen

    r = screen.three_axis(cid, cutoff)
    return {
        axis: {
            "score": getattr(r, axis).score,
            "trend": getattr(r, axis).trend,
            "confidence": getattr(r, axis).confidence,
            "evidence_event_ids": [str(i) for i in getattr(r, axis).evidence_event_ids],
        }
        for axis in ("founder", "market", "idea_vs_market")
    }


def _gate(cid: UUID | None, cutoff: datetime) -> dict:
    from intelligence import gate

    d = gate.evaluate(cid, cutoff)
    return {
        "outcome": str(d.outcome),
        "rationale": d.rationale,
        "absence_is_suspicious": d.absence_is_suspicious,
    }


# ---------------------------------------------------------------------------
# Calibration — the H12 gate and the most persuasive slide in the deck.
# ---------------------------------------------------------------------------


def _trajectory(member: dict, points: int = 12) -> tuple[list[dict], dict]:
    """Score at successive cutoffs up to the series bound. Returns (series, diagnostics).

    Every point is a real filter run at that cutoff — never an interpolation backwards
    from the final value, which would be lookahead wearing a chart's clothing.

    There is deliberately NO fixture fallback. This function used to catch LookupError
    and return the hand-authored trajectory from data/seed/backtest.json instead; every
    cohort member had a null company_id, so every member took that path, score.founder()
    was never called, and the report still described itself as a replay. A backtest that
    silently substitutes authored numbers for a failed replay is not degraded — it is
    false. When the replay cannot run, this returns an empty series and says why, and
    the member is counted as not replayed.
    """
    from api.routers.deps import founder_entity_ids
    from memory import score as score_mod, store

    cut = _series_bound(member)
    cid = _as_uuid(member.get("company_id"))
    diag = {"replayed": False, "events_checked": 0, "reason": None}

    if cid is None:
        diag["reason"] = "cohort member has no company_id in the store — run scripts/seed.py"
        return [], diag
    ents = founder_entity_ids(cid)
    if not ents:
        diag["reason"] = f"no founder entity resolves for company {cid}"
        return [], diag

    entity_id = ents[0]
    events = store.events(as_of=cut, entity_id=entity_id)
    if not events:
        diag["reason"] = f"no events on or before {cut.isoformat()}"
        return [], diag

    start = min(e.observed_at for e in events)
    step = (cut - start or timedelta(days=1)) / max(points - 1, 1)
    series = []
    checked = 0
    for i in range(points):
        at = start + step * i
        window = [e for e in events if e.observed_at <= at]
        checked += assert_no_lookahead(window, at)
        fs = score_mod.founder(entity_id, at)
        series.append(
            {
                "as_of": at.isoformat(),
                "mu": fs.mu,
                "band": fs.band,
                "trend": fs.trend,
                # How many events the filter actually consumed at this cutoff. Zero
                # means the point IS the prior (mu=0.5), not a reading — see _peak.
                "n_observations": len(fs.contributing_event_ids),
            }
        )

    diag.update({"replayed": True, "events_checked": checked, "entity_id": str(entity_id)})
    return series, diag


def _series_bound(member: dict) -> datetime:
    """How far a replayed trajectory may run — up to, but never through, breakout.

    Distinct from the source truncation date, and conflating the two was a bug:
    clipping the series at `source_truncation_date` (the FIRST point) collapsed every
    winner to a single value, so nothing cleared the threshold and the hit rate was 0.
    The claim under test is "the score was already rising before the breakout", which
    needs the span between those dates, not a point.
    """
    if member.get("breakout_at"):
        return _aware(member["breakout_at"])
    return _truncation(member)


def _truncation(member: dict) -> datetime:
    """The collection cutoff: the date past which no source was gathered for this member.

    The cohort calls this `collection_cutoff` and sets it to the member's breakout date,
    so every collected event predates the moment the founder became widely known.
    Earlier drafts used `truncation_date` / `source_truncation_date`; both are still
    accepted so a member written in the old shape does not silently lose its bound.
    """
    for key in ("collection_cutoff", "truncation_date", "source_truncation_date", "as_of"):
        if member.get(key):
            return _aware(member[key])
    raise KeyError("cohort member has no collection cutoff")


def _scored(series: list[dict]) -> list[dict]:
    """Points where the filter actually consumed evidence.

    The first cutoff of every series lands on the founder's earliest event, before any
    observation has been derived from it, so the filter returns the untouched prior of
    mu=0.5. That is the scorer saying "I know nothing", and it is not a score. Counting
    it made every control peak at exactly 0.5 — a flat 0.5 for a founder whose real
    replayed level is 0.22, and only 0.12 below the threshold the fame check turns on.
    A prior must never be reported as a measurement.
    """
    return [p for p in series if p.get("n_observations")]


def _peak(series: list[dict]) -> float | None:
    values = [p["mu"] for p in _scored(series) if isinstance(p.get("mu"), (int, float))]
    return max(values) if values else None


def run_calibration() -> dict:
    """Winners' trajectories vs controls, threshold line, hit rate, and one failure the
    system correctly deprioritized.

    fame_check_passed is the H12 hard gate: if a control clears the threshold, the score
    is measuring fame rather than trajectory and the thesis is dead. It is surfaced as a
    top-level boolean so it is impossible to miss or to quietly ignore.
    """
    cohort = collect.load_cohort()
    threshold = cohort["threshold"]

    results = []
    events_checked = 0
    for m in cohort["members"]:
        series, diag = _trajectory(m)
        events_checked += diag["events_checked"]
        peak = _peak(series)
        results.append(
            {
                "replayed": diag["replayed"],
                "not_replayed_reason": diag["reason"],
                "lookahead_events_checked": diag["events_checked"],
                "detected_at": _first_clearing(series, threshold),
                # The cohort names its entries `name` and carries the founder as an
                # object so the seeder can mint a real entity from it. Carrying only
                # one of the two dropped every label, so the calibration chart had no
                # way to say who each line was.
                "founder": _founder_name(m),
                "name": m.get("name") or _founder_name(m),
                "id": m.get("id"),
                "sector": m.get("sector"),
                "outcome": m.get("outcome")
                or m.get("what_happened")
                or ("breakout" if m.get("label") == "winner" else None),
                "why": m.get("why_we_deprioritized") or m.get("truncation_note"),
                "company_id": m.get("company_id"),
                "label": str(m.get("label", "unknown")).lower(),
                "truncation_date": _truncation(m).isoformat(),
                "trajectory": series,
                "peak_mu": peak,
                "cleared_threshold": bool(peak is not None and peak >= threshold),
                "note": m.get("note"),
            }
        )

    winners = [r for r in results if r["label"] == "winner"]
    controls = [r for r in results if r["label"] == "control"]
    failures = [r for r in results if r["label"] == "failure"]

    # Only replayed members are evidence. A member whose replay did not run has no
    # score, and counting it as "did not clear" would turn a broken rig into a passing
    # fame check — the exact inversion this gate exists to catch.
    replayed_winners = [r for r in winners if r["replayed"]]
    replayed_controls = [r for r in controls if r["replayed"]]

    controls_clearing = [r for r in replayed_controls if r["cleared_threshold"]]
    # Vacuous truth is not a pass: with no REPLAYED controls the check did not run.
    fame_check_evaluated = bool(replayed_controls)
    fame_check_passed = fame_check_evaluated and not controls_clearing

    hits = [r for r in replayed_winners if r["cleared_threshold"]]
    deprioritized = next(
        (r for r in failures if r["replayed"] and not r["cleared_threshold"]),
        next((r for r in failures), None),
    )

    return {
        "threshold": threshold,
        "results": results,
        "winners": winners,
        "controls": controls,
        "hit_rate": (len(hits) / len(replayed_winners)) if replayed_winners else None,
        "hits": len(hits),
        "winners_evaluated": len(replayed_winners),
        "controls_evaluated": len(replayed_controls),
        "members_replayed": sum(1 for r in results if r["replayed"]),
        "members_total": len(results),
        "not_replayed": [
            {"name": r["name"], "reason": r["not_replayed_reason"]}
            for r in results
            if not r["replayed"]
        ],
        "fame_check_passed": fame_check_passed,
        "fame_check_evaluated": fame_check_evaluated,
        "controls_clearing_threshold": [r["founder"] for r in controls_clearing],
        "correctly_deprioritized_failure": deprioritized,
        # Both of these are measured, never asserted. `lookahead_checked` is True only
        # because assert_no_lookahead ran over `events_checked` events and did not
        # raise; with nothing replayed, nothing was checked and it reports False.
        "lookahead_checked": events_checked > 0,
        "events_checked": events_checked,
    }


def _founder_name(member: dict) -> str | None:
    founder = member.get("founder")
    if isinstance(founder, dict):
        return founder.get("display_name") or founder.get("name_normalized")
    return str(founder) if founder else member.get("name")


def _first_clearing(series: list[dict], threshold: float) -> str | None:
    """The earliest replayed as_of at which the founder axis cleared the threshold.

    This is the "we would have found them on this date" claim, and it is read off the
    replayed series rather than recorded in the fixture. The cohort file used to carry
    a `detected_at` per winner; a detection date that the replay did not produce is a
    prediction written after the fact.
    """
    for point in _scored(series):
        if isinstance(point.get("mu"), (int, float)) and point["mu"] >= threshold:
            return point["as_of"]
    return None
