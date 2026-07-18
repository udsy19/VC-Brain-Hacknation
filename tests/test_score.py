"""Founder Score (Kalman): determinism, momentum, uncertainty, receipts, and the
contradiction boundary."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from memory import score, store
from schema.events import Event, EventKind, FounderScore, Source

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _entity():
    return store.get_store().create_entity("F", "f").entity_id


def _green(eid, month, value, *, sc=0.9, source=Source.MANUAL, kind=EventKind.GREEN_FLAG):
    return store.append(
        Event(
            entity_id=eid,
            kind=kind,
            source=source,
            observed_at=T0 + timedelta(days=30 * month),
            payload={"value": value, "self_consistency": sc},
        )
    )


def test_no_evidence_returns_wide_neutral_prior() -> None:
    fs = score.founder(_entity(), as_of=T0)
    assert isinstance(fs, FounderScore)
    assert abs(fs.mu - 0.5) < 1e-9
    assert abs(fs.band - 0.5) < 1e-9
    assert abs(fs.trend) < 1e-9
    assert fs.contributing_event_ids == []
    assert fs.model == "kalman"


def test_deterministic_same_input_same_output() -> None:
    eid = _entity()
    _green(eid, 0, 0.6)
    _green(eid, 3, 0.8)
    a = score.founder(eid, as_of=T0 + timedelta(days=200))
    b = score.founder(eid, as_of=T0 + timedelta(days=200))
    assert (a.mu, a.band, a.trend) == (b.mu, b.band, b.trend)


def test_rising_evidence_lifts_mu_and_gives_positive_trend() -> None:
    eid = _entity()
    _green(eid, 0, 0.6)
    _green(eid, 3, 0.72)
    _green(eid, 6, 0.82)
    fs = score.founder(eid, as_of=T0 + timedelta(days=190))
    assert fs.mu > 0.6
    assert fs.trend > 0
    assert len(fs.contributing_event_ids) == 3


def test_an_observation_tightens_the_band() -> None:
    eid = _entity()
    as_of = T0 + timedelta(days=1)
    prior = score.founder(eid, as_of=as_of).band
    _green(eid, 0, 0.7)
    posterior = score.founder(eid, as_of=as_of).band
    assert posterior < prior


def test_staleness_widens_the_band() -> None:
    eid = _entity()
    last = T0 + timedelta(days=30)
    _green(eid, 0, 0.7)
    _green(eid, 1, 0.75)
    fresh = score.founder(eid, as_of=last).band
    stale = score.founder(eid, as_of=last + timedelta(days=365)).band
    assert stale > fresh


def test_low_noise_proof_event_moves_score_more_than_noisy_web_flag() -> None:
    eid_proof = _entity()
    _green(eid_proof, 0, 0.9, sc=0.95, source=Source.PROOF_PROTOCOL, kind=EventKind.PROOF_ARTIFACT)
    eid_web = _entity()
    _green(eid_web, 0, 0.9, sc=0.4, source=Source.WEB)
    at = T0 + timedelta(days=1)
    assert score.founder(eid_proof, as_of=at).mu > score.founder(eid_web, as_of=at).mu


def test_contributing_event_ids_are_the_receipts() -> None:
    eid = _entity()
    e1 = _green(eid, 0, 0.6)
    e2 = _green(eid, 2, 0.7)
    fs = score.founder(eid, as_of=T0 + timedelta(days=120))
    assert set(fs.contributing_event_ids) == {e1, e2}


def test_contradicted_observation_is_excluded() -> None:
    eid = _entity()
    good = _green(eid, 0, 0.85)
    bad = _green(eid, 1, 0.2, sc=0.4, source=Source.WEB)
    with_bad = score.founder(eid, as_of=T0 + timedelta(days=90))
    # Now contradict the weak flag.
    store.append(
        Event(
            entity_id=eid,
            kind=EventKind.CONTRADICTION,
            source=Source.VALIDATOR,
            observed_at=T0 + timedelta(days=45),
            payload={"target_event_id": str(bad)},
        )
    )
    without_bad = score.founder(eid, as_of=T0 + timedelta(days=90))
    assert bad not in without_bad.contributing_event_ids
    assert good in without_bad.contributing_event_ids
    assert without_bad.mu != with_bad.mu  # dropping the contradicted flag changed the score


def test_forecast_widens_interval_further_out() -> None:
    eid = _entity()
    _green(eid, 0, 0.7)
    _green(eid, 2, 0.75)
    as_of = T0 + timedelta(days=70)
    _mu30, band30 = score.forecast(eid, as_of, k_days=30)
    _mu365, band365 = score.forecast(eid, as_of, k_days=365)
    assert band365 > band30
