"""The backtest is proof #1, so its guarantees get tested harder than anything else.

Three claims must hold: no future event ever reaches the scorer, control founders do
not clear the threshold, and every number in the report is one the replay produced.
If the second fails, the score measures fame. If the third fails, the artifact whose
job is proving the system does not fool itself is fooling the reader.

These tests run the REAL path: cohort members are written into a temporary event store
as entities with events, and run_calibration() scores them through memory.score.founder
at successive as_of dates. A test that asserted on a hand-authored trajectory would be
testing the fixture, which is how the fabricated report passed CI for as long as it did.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from backtest import collect
from backtest.runner import LookaheadError, assert_no_lookahead, replay, run_calibration
from memory import db, store
from schema.events import Event, EventKind, Source

CUT = datetime(2020, 1, 1, tzinfo=timezone.utc)
PAST = CUT - timedelta(days=30)
FUTURE = CUT + timedelta(days=30)
START = CUT - timedelta(days=360)


def _event(observed_at: datetime, **kw) -> Event:
    return Event(
        kind=kw.pop("kind", EventKind.GREEN_FLAG),
        source=kw.pop("source", Source.GITHUB),
        observed_at=observed_at,
        **kw,
    )


def _readings(values: list[float]) -> list[dict]:
    """Green-flag rollups the scorer consumes, spread evenly over the year before CUT.

    These are sensor readings, not scores: the filter still has to turn them into a
    level and a trend, and the level it produces is not any of these numbers.
    """
    step = (CUT - START) / max(len(values) - 1, 1)
    return [
        {
            "kind": "green_flag",
            "source": "github",
            "observed_at": (START + step * i).isoformat(),
            "payload": {"value": v, "n_flags": 20},
        }
        for i, v in enumerate(values)
    ]


RISING = _readings([0.20, 0.30, 0.42, 0.55, 0.68, 0.80, 0.88, 0.92])
FLAT_LOW = _readings([0.16, 0.15, 0.17, 0.16, 0.15, 0.16, 0.17, 0.16])
FLAT_MID = _readings([0.30, 0.29, 0.31, 0.30, 0.29, 0.30, 0.31, 0.30])


def _member(name: str, label: str, events: list[dict], **extra) -> dict:
    return {
        "id": f"{label}-{name}",
        "name": name,
        "company_name": name,
        "label": label,
        "founder": {"display_name": f"{name} founder", "name_normalized": f"{name} founder"},
        "breakout_at": CUT.isoformat(),
        "collection_cutoff": CUT.isoformat(),
        "events": events,
        **extra,
    }


COHORT = {
    "threshold": 0.6,
    "cohort": [
        _member("winner-a", "winner", RISING),
        _member("winner-b", "winner", RISING),
        _member("control-a", "control", FLAT_LOW),
        _member("control-b", "control", FLAT_MID),
        _member("failure-a", "failure", FLAT_LOW, note="high visibility, no shipping trajectory"),
    ],
}


def _seed(cohort: dict) -> None:
    """Write the cohort into the store the same way scripts/seed.py does."""
    for m in cohort["cohort"]:
        company_id = store.upsert_company(m["company_name"])
        entity_id = store.upsert_entity(
            m["founder"]["display_name"], m["founder"]["name_normalized"]
        )
        for raw in m["events"]:
            store.append(
                Event(
                    entity_id=entity_id,
                    company_id=company_id,
                    kind=EventKind(raw["kind"]),
                    source=Source(raw["source"]),
                    observed_at=datetime.fromisoformat(raw["observed_at"]),
                    payload=raw["payload"],
                )
            )


def _write_cohort(tmp_path, monkeypatch, cohort: dict, filename: str = "backtest.json") -> None:
    path = tmp_path / filename
    path.write_text(json.dumps(cohort))
    monkeypatch.setattr(collect, "SEED_PATH", path)


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("VCBRAIN_DB_PATH", str(tmp_path / "test.db"))
    db.reset_connections()
    _write_cohort(tmp_path, monkeypatch, COHORT)
    # No live network in tests: B's scanners are real now and will hit rate limits.
    monkeypatch.setattr(collect, "_scan", lambda founder: [])
    monkeypatch.setattr(
        "core.llm.complete", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no network"))
    )
    yield
    db.reset_connections()


# --- the lookahead assertion ------------------------------------------------


def test_assert_no_lookahead_passes_on_truncated_events() -> None:
    assert assert_no_lookahead([_event(PAST), _event(CUT)], CUT) == 2


def test_assert_no_lookahead_returns_the_number_it_checked() -> None:
    """The count is what lets the report state a real number instead of a literal."""
    assert assert_no_lookahead([], CUT) == 0
    assert assert_no_lookahead([_event(PAST) for _ in range(7)], CUT) == 7


def test_assert_no_lookahead_raises_on_a_future_event() -> None:
    with pytest.raises(LookaheadError) as exc:
        assert_no_lookahead([_event(PAST), _event(FUTURE)], CUT)
    assert "future" in str(exc.value)


def test_lookahead_raises_rather_than_warns(recwarn) -> None:
    with pytest.raises(LookaheadError):
        assert_no_lookahead([_event(FUTURE)], CUT)
    assert not recwarn.list, "a lookahead leak must raise, never warn"


def test_replay_raises_when_the_store_leaks_the_future(monkeypatch) -> None:
    """If as_of scoping regresses anywhere upstream, the rig must fail loudly."""
    company_id = store.upsert_company("Leaky")
    monkeypatch.setattr(store, "events", lambda **kw: [_event(FUTURE, company_id=company_id)])
    with pytest.raises(LookaheadError):
        replay(company_id, CUT)


def test_replay_uses_the_live_code_path() -> None:
    company_id = store.upsert_company("Ferrite")
    store.append(_event(PAST, company_id=company_id, payload={"value": 0.7}))
    store.append(_event(FUTURE, company_id=company_id, payload={"value": 0.95}))

    out = replay(company_id, CUT)
    assert out["lookahead_checked"] is True
    assert out["lookahead_events_checked"] >= 1, "the assertion reports what it saw"
    assert out["event_count"] == 1, "the post-cutoff event was not replayed"
    assert out["memo"] is not None and "gaps" in out["memo"]


# --- the replay is a replay -------------------------------------------------


def test_every_cohort_member_takes_the_live_path() -> None:
    """The bug this guards: all 9 members had a null company_id, so every one of them
    fell through to a hand-authored trajectory while the report called itself a replay."""
    _seed(COHORT)
    report = run_calibration()

    assert report["members_replayed"] == report["members_total"] == 5
    assert report["not_replayed"] == []
    assert all(r["replayed"] for r in report["results"])


def test_scores_come_from_the_scorer_not_the_fixture(monkeypatch) -> None:
    """score.founder must actually be called. If it is not, there is no backtest."""
    from memory import score as score_mod

    calls: list[tuple[UUID, datetime]] = []
    real = score_mod.founder
    monkeypatch.setattr(
        score_mod, "founder", lambda e, at: (calls.append((e, at)), real(e, at))[1]
    )

    _seed(COHORT)
    run_calibration()

    assert len(calls) >= 5 * 12, "each member is scored at every cutoff in its series"
    assert len({e for e, _ in calls}) == 5, "every member's own entity was scored"


def test_a_member_absent_from_the_store_is_reported_not_fabricated() -> None:
    """The honest degradation: say the replay did not run. Never substitute numbers."""
    report = run_calibration()  # nothing seeded

    assert report["members_replayed"] == 0
    assert [n["name"] for n in report["not_replayed"]] == [
        m["name"] for m in COHORT["cohort"]
    ]
    assert all(r["trajectory"] == [] and r["peak_mu"] is None for r in report["results"])
    assert report["lookahead_checked"] is False, "nothing ran, so nothing was checked"
    assert report["events_checked"] == 0
    assert report["hit_rate"] is None


def test_lookahead_checked_is_measured_not_asserted() -> None:
    """A literal True here is a false claim in the artifact that exists to prevent them."""
    _seed(COHORT)
    report = run_calibration()

    assert report["events_checked"] > 0
    assert report["lookahead_checked"] is True
    assert sum(r["lookahead_events_checked"] for r in report["results"]) == report[
        "events_checked"
    ]


def test_the_prior_is_not_reported_as_a_score() -> None:
    """mu=0.5 with no observations is the filter saying "I know nothing"."""
    _seed(COHORT)
    report = run_calibration()

    for r in report["results"]:
        for point in r["trajectory"]:
            if not point["n_observations"]:
                assert point["mu"] == pytest.approx(0.5), "unobserved points are the prior"
        assert r["peak_mu"] != pytest.approx(0.5), "the prior never becomes the peak"


# --- calibration ------------------------------------------------------------


def test_run_calibration_reports_the_fame_check() -> None:
    _seed(COHORT)
    report = run_calibration()

    assert report["fame_check_evaluated"] is True
    assert report["fame_check_passed"] is True
    assert report["controls_clearing_threshold"] == []


def test_fame_check_fails_when_a_control_clears_the_threshold(tmp_path, monkeypatch) -> None:
    """The gate must actually be able to fail, or it isn't a gate.

    The control here is given a genuinely rising evidence stream — the same readings as
    a winner. The gate fails because the SCORER rates it highly, not because a number
    was edited into a fixture.
    """
    famous = json.loads(json.dumps(COHORT))
    famous["cohort"][2]["events"] = RISING
    _write_cohort(tmp_path, monkeypatch, famous, "famous.json")
    _seed(famous)

    report = run_calibration()
    assert report["fame_check_passed"] is False
    assert report["controls_clearing_threshold"] == ["control-a founder"]


def test_calibration_reports_hit_rate_and_a_deprioritized_failure() -> None:
    _seed(COHORT)
    report = run_calibration()

    assert report["hit_rate"] == 1.0
    assert report["winners_evaluated"] == 2
    failure = report["correctly_deprioritized_failure"]
    assert failure["name"] == "failure-a"
    assert failure["cleared_threshold"] is False
    assert failure["note"]


def test_winners_rise_and_controls_stay_flat() -> None:
    """The separation the whole pitch rests on, measured rather than drawn."""
    _seed(COHORT)
    report = run_calibration()

    assert all(w["peak_mu"] >= report["threshold"] for w in report["winners"])
    assert all(c["peak_mu"] < report["threshold"] for c in report["controls"])
    for w in report["winners"]:
        scored = [p["mu"] for p in w["trajectory"] if p["n_observations"]]
        assert scored[-1] > scored[0], "the winner's replayed level rises before breakout"


def test_detection_date_is_read_off_the_replay() -> None:
    """`detected_at` used to be recorded in the fixture, which makes it a prediction
    written after the fact rather than a result."""
    _seed(COHORT)
    report = run_calibration()

    for w in report["winners"]:
        assert w["detected_at"], "a cleared winner has a replayed detection date"
        at = datetime.fromisoformat(w["detected_at"])
        assert at <= datetime.fromisoformat(w["truncation_date"])
        point = next(p for p in w["trajectory"] if p["as_of"] == w["detected_at"])
        assert point["mu"] >= report["threshold"]
    for c in report["controls"]:
        assert c["detected_at"] is None


def test_no_controls_is_not_a_pass(tmp_path, monkeypatch) -> None:
    """Vacuous truth would let the whole thesis through unchecked."""
    winners_only = {"threshold": 0.6, "cohort": [COHORT["cohort"][0]]}
    _write_cohort(tmp_path, monkeypatch, winners_only, "winners_only.json")
    _seed(winners_only)

    report = run_calibration()
    assert report["fame_check_evaluated"] is False
    assert report["fame_check_passed"] is False


def test_unreplayed_controls_do_not_count_as_a_pass(tmp_path, monkeypatch) -> None:
    """A control that never ran is not a control that stayed below the line."""
    partial = json.loads(json.dumps(COHORT))
    seeded = {"threshold": 0.6, "cohort": [m for m in partial["cohort"] if m["label"] != "control"]}
    _write_cohort(tmp_path, monkeypatch, partial)
    _seed(seeded)  # winners and the failure only — controls are absent from the store

    report = run_calibration()
    assert report["controls_evaluated"] == 0
    assert report["fame_check_evaluated"] is False
    assert report["fame_check_passed"] is False


def test_trajectories_are_truncated_at_the_cutoff() -> None:
    _seed(COHORT)
    report = run_calibration()

    for r in report["results"]:
        cut = datetime.fromisoformat(r["truncation_date"])
        for point in r["trajectory"]:
            assert datetime.fromisoformat(point["as_of"]) <= cut


# --- collection -------------------------------------------------------------


def test_collect_records_the_truncation_date_explicitly() -> None:
    fp = collect.collect("winner-a", CUT, label="winner")
    assert fp.truncation_date == CUT
    assert fp.as_dict()["truncation_date"] == CUT.isoformat()


def test_collect_drops_post_cutoff_signals() -> None:
    member = {
        "founder": "sig",
        "label": "winner",
        "truncation_date": CUT.isoformat(),
        "signals": [
            {"observed_at": PAST.isoformat(), "url": "keep"},
            {"observed_at": FUTURE.isoformat(), "url": "drop"},
        ],
    }
    collect.SEED_PATH.write_text(json.dumps({"threshold": 0.6, "cohort": [member]}))
    fp = collect.collect("sig", CUT)
    assert [s["url"] for s in fp.raw_signals] == ["keep"]


def test_collect_truncates_scanner_events(monkeypatch) -> None:
    """The scanner path is truncated at collection too, not only at read time."""
    monkeypatch.setattr(collect, "_scan", lambda founder: ["raw"])
    monkeypatch.setattr(collect, "_ingest", lambda signals: [_event(PAST), _event(FUTURE)])

    fp = collect.collect("sig", CUT)
    assert fp.origin == "scanners"
    assert [e.observed_at for e in fp.events] == [PAST]


def test_load_cohort_resolves_company_ids_from_the_store() -> None:
    """Null company_ids are what routed every member to the fabricated fixture path."""
    assert all(m["company_id"] is None for m in collect.load_cohort()["members"])

    _seed(COHORT)
    members = collect.load_cohort()["members"]
    assert all(m["company_id"] for m in members)
    assert {m["company_id"] for m in members} == {
        str(c["company_id"]) for c in store.all_companies()
    }


def test_load_cohort_raises_without_collected_data(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(collect, "SEED_PATH", tmp_path / "missing.json")
    with pytest.raises(LookupError):
        collect.load_cohort()
