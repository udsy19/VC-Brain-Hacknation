"""Integrated Founder Score contract: calibration, Kalman, and fallback."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest

from memory import score, store
from schema.events import Event, EventKind, FounderScore, Source

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _flag(entity_id: UUID, day: int, value: float, **kwargs) -> Event:
    payload = kwargs.pop("payload", {})
    if not payload or set(payload) == {"claim_id"}:
        payload = {"value": value, **payload}
    event = Event(
        kind=kwargs.pop("kind", EventKind.GREEN_FLAG),
        source=kwargs.pop("source", Source.WEB),
        entity_id=entity_id,
        observed_at=T0 + timedelta(days=day),
        payload=payload,
        **kwargs,
    )
    store.append(event)
    return event


def _series(entity_id: UUID, values: list[float], step: int = 14) -> None:
    for index, value in enumerate(values):
        _flag(entity_id, index * step, value)


def _at(n_obs: int, step: int = 14) -> datetime:
    return T0 + timedelta(days=(n_obs - 1) * step, hours=1)


def test_no_evidence_returns_wide_neutral_prior() -> None:
    result = score.founder(uuid4(), as_of=T0)
    assert isinstance(result, FounderScore)
    assert result.mu == pytest.approx(0.5)
    assert result.band == pytest.approx(0.5)
    assert result.trend == pytest.approx(0.0)
    assert result.contributing_event_ids == []


def test_deterministic_same_input_same_output() -> None:
    entity_id = uuid4()
    _series(entity_id, [0.6, 0.8], step=90)
    first = score.founder(entity_id, T0 + timedelta(days=200))
    second = score.founder(entity_id, T0 + timedelta(days=200))
    assert (first.mu, first.band, first.trend) == (second.mu, second.band, second.trend)


def test_band_tightens_monotonically_as_observations_accumulate() -> None:
    entity_id = uuid4()
    _series(entity_id, [0.7] * 8)
    bands = [score.founder(entity_id, _at(n)).band for n in range(1, 9)]
    assert all(a > b for a, b in zip(bands, bands[1:])), bands
    assert bands[0] < 0.5
    assert bands[-1] < bands[0] * 0.7
    assert bands[-1] < score.P0[0] ** 0.5


def test_band_tightens_on_an_irregular_real_world_cadence() -> None:
    """The uniform-step case above passed even while the band was widening on real data:
    with an undamped transition the propagated level variance picks up dt**2 * P11, so
    the band tracked observation SPACING, not observation COUNT, and rose the moment a
    monthly cadence slipped to bi-monthly. Tensorpage's real cadence, which is what
    regressed (0.175 -> 0.183 -> 0.176 -> 0.216 -> 0.241)."""
    entity_id = uuid4()
    # Month offsets of Tensorpage's actual 11 observations: monthly, then irregular.
    days = [0, 30, 61, 122, 181, 242, 273, 334, 395, 487, 518]
    values = [0.21, 0.21, 0.38, 0.67, 0.67, 0.66, 0.67, 0.67, 0.67, 0.67, 0.67]
    for day, value in zip(days, values):
        _flag(entity_id, day, value)

    bands = [score.founder(entity_id, T0 + timedelta(days=day, hours=1)).band for day in days]
    assert all(a > b for a, b in zip(bands, bands[1:])), bands
    assert bands[-1] < bands[0] * 0.6, bands


def test_staleness_widens_the_band() -> None:
    entity_id = uuid4()
    _series(entity_id, [0.7] * 6)
    fresh = score.founder(entity_id, _at(6))
    stale = score.founder(entity_id, _at(6) + timedelta(days=365))
    assert stale.band > fresh.band
    assert stale.contributing_event_ids == fresh.contributing_event_ids


@pytest.mark.parametrize(
    "values,expect",
    [
        ([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], "positive"),
        ([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3], "negative"),
        ([0.6] * 7, "flat"),
    ],
)
def test_trend_is_structural_momentum(values: list[float], expect: str) -> None:
    entity_id = uuid4()
    _series(entity_id, values)
    got = score.founder(entity_id, _at(len(values)))
    if expect == "positive":
        assert got.trend > 1e-3
    elif expect == "negative":
        assert got.trend < -1e-3
    else:
        assert abs(got.trend) < 0.05


def test_no_lookahead_and_truncated_history_are_equivalent() -> None:
    entity_id = uuid4()
    values = [0.5, 0.55, 0.6, 0.95, 0.97, 0.99]
    _series(entity_id, values)
    midpoint, final = _at(3), _at(len(values))
    mid = score.founder(entity_id, midpoint)
    end = score.founder(entity_id, final)
    assert mid.mu < end.mu
    assert len(mid.contributing_event_ids) == 3
    assert len(end.contributing_event_ids) == 6

    truncated_entity = uuid4()
    _series(truncated_entity, values[:3])
    truncated = score.founder(truncated_entity, midpoint)
    assert truncated.mu == pytest.approx(mid.mu)
    assert truncated.band == pytest.approx(mid.band)
    assert truncated.trend == pytest.approx(mid.trend)


def test_contradicted_claims_are_excluded_from_observations() -> None:
    entity_id, claim_id = uuid4(), uuid4()
    _series(entity_id, [0.2, 0.2, 0.2])
    tainted = _flag(entity_id, 42, 0.99, payload={"claim_id": str(claim_id)})
    as_of = T0 + timedelta(days=60)
    with_claim = score.founder(entity_id, as_of)
    assert tainted.event_id in with_claim.contributing_event_ids
    store.append(
        Event(
            kind=EventKind.VALIDATION_RESULT,
            source=Source.VALIDATOR,
            company_id=uuid4(),
            observed_at=T0 + timedelta(days=50),
            payload={"claim_id": str(claim_id), "status": "contradicted"},
        )
    )
    after = score.founder(entity_id, as_of)
    assert tainted.event_id not in after.contributing_event_ids
    assert len(after.contributing_event_ids) == 3
    assert after.mu < with_claim.mu
    assert score.observations(entity_id, as_of).dropped_contradicted == [tainted.event_id]


def test_verified_claims_are_kept() -> None:
    entity_id, claim_id = uuid4(), uuid4()
    supported = _flag(entity_id, 0, 0.8, payload={"claim_id": str(claim_id)})
    store.append(
        Event(
            kind=EventKind.VALIDATION_RESULT,
            source=Source.VALIDATOR,
            observed_at=T0,
            payload={"claim_id": str(claim_id), "status": "verified"},
        )
    )
    assert score.founder(entity_id, T0 + timedelta(days=1)).contributing_event_ids == [
        supported.event_id
    ]


def test_score_always_carries_receipts() -> None:
    entity_id = uuid4()
    _series(entity_id, [0.6, 0.7])
    got = score.founder(entity_id, _at(2))
    assert len(got.contributing_event_ids) == 2
    empty = score.founder(uuid4(), T0)
    assert empty.contributing_event_ids == []
    assert empty.mu == pytest.approx(score.MU0)
    assert empty.band == pytest.approx(score.P0[0] ** 0.5)


def test_payload_shapes_are_parsed_and_calibrated_defensively() -> None:
    entity_id = uuid4()
    _flag(entity_id, 0, 0.6)
    _flag(
        entity_id,
        1,
        0.0,
        payload={"flags": [{"fired": True, "weight": 3.0}, {"fired": False, "weight": 1.0}]},
    )
    _flag(entity_id, 2, 0.0, payload={"notes": "unrecognised"})
    observed = score.observations(entity_id, T0 + timedelta(days=3))
    assert len(observed.kept) == 2
    assert [item.y for item in observed.kept] == pytest.approx(
        [score.calibrate(0.6), score.calibrate(0.75, 2)]
    )
    assert all(0.0 <= item.y <= 1.0 for item in observed.kept)


def _proof_pair(entity_id: UUID, day: int, artifact: float = 0.9, behavior: float = 0.85) -> None:
    """One grading of a challenge: the (artifact, behaviour) pair the API appends."""
    _flag(entity_id, day, artifact, kind=EventKind.PROOF_ARTIFACT, source=Source.PROOF_PROTOCOL)
    _flag(entity_id, day + 1, behavior, kind=EventKind.PROOF_BEHAVIOR, source=Source.PROOF_PROTOCOL)


def test_proof_events_move_the_score_hard() -> None:
    entity_id = uuid4()
    for index, value in enumerate([0.45, 0.5, 0.48]):
        _flag(entity_id, index * 21, value, source=Source.DECK)
    before = score.founder(entity_id, T0 + timedelta(days=60))
    _proof_pair(entity_id, 70)
    after = score.founder(entity_id, T0 + timedelta(days=80))
    assert after.mu > before.mu + 0.15
    assert after.band < before.band
    assert after.trend > before.trend


def _effective_noise(entity_id: UUID) -> float:
    """Noise the filter actually uses for an entity's single observation."""
    (observation,) = score.build_observations(entity_id, T0 + timedelta(days=5))
    return score._noise_for_schema(observation)


def test_proof_is_the_strongest_single_observation_type() -> None:
    """Proof still beats every other source, flatly: its noise is a floor, immune to the
    source and consistency penalties that inflate everything else."""
    proof_entity = uuid4()
    _flag(proof_entity, 0, 0.9, kind=EventKind.PROOF_ARTIFACT, source=Source.PROOF_PROTOCOL)
    proof_noise = _effective_noise(proof_entity)

    # Compared against every other source at PERFECT self-consistency, its best case.
    for source in (Source.DECK, Source.MANUAL, Source.WEB, Source.HN, Source.GITHUB,
                   Source.ARXIV, Source.VALIDATOR):
        other = uuid4()
        _flag(other, 0, 0.9, source=source, confidence=1.0)
        assert _effective_noise(other) > proof_noise, source

    # A low-consistency proof result is still floored, not diluted away.
    noisy = uuid4()
    _flag(noisy, 0, 0.9, kind=EventKind.PROOF_ARTIFACT, source=Source.PROOF_PROTOCOL,
          confidence=0.1)
    assert _effective_noise(noisy) == pytest.approx(proof_noise)
    # ... and it is floored well above the raw 0.15 * 0.2 = 0.03 compounded penalty
    # that let two proof events out-certain an 18-month track record.
    assert proof_noise > score.R0 * 0.15 * 0.2 * 5


def test_proof_cannot_out_certain_a_real_track_record() -> None:
    """Spec 2c: proof intervals stay wide and displayed. A 60-90 minute exercise must
    never yield more certainty than an accumulated record of shipped work, however many
    times the exercise is re-run."""
    shipped = uuid4()
    _series(shipped, [0.6, 0.62, 0.65, 0.66, 0.68, 0.67, 0.69, 0.7], step=30)
    track_record = score.founder(shipped, _at(8, step=30))

    cold_start = uuid4()
    for round_no in range(6):
        _proof_pair(cold_start, 10 + round_no * 3)
    proof_only = score.founder(cold_start, T0 + timedelta(days=40))

    assert proof_only.band > track_record.band, (proof_only.band, track_record.band)


def test_repeated_proof_gradings_have_diminishing_returns() -> None:
    """Each grading appends a fresh (artifact, behaviour) pair under a new uuid5.
    Re-running the demo beat must not drive the score up without bound."""
    entity_id = uuid4()
    mus, bands = [], []
    for round_no in range(6):
        _proof_pair(entity_id, 10 + round_no * 3)
        got = score.founder(entity_id, T0 + timedelta(days=40))
        mus.append(got.mu)
        bands.append(got.band)

    # The first grading does the work; five more move the level almost not at all.
    first_step = mus[1] - mus[0]
    assert mus[-1] - mus[1] < first_step, (mus, first_step)
    assert mus[-1] - mus[0] < 0.05, mus
    # The band converges rather than collapsing toward zero.
    assert bands[-1] > bands[0] * 0.8, bands
    assert bands[-1] > 0.15, bands


def test_momentum_decays_across_silence_and_drift_is_bounded() -> None:
    """Silence is not evidence. A rising trend must not extrapolate forever: it decays
    with MOMENTUM_HALFLIFE_DAYS, so mu converges instead of running away from the
    readings."""
    entity_id = uuid4()
    _series(entity_id, [0.20, 0.28, 0.33, 0.37], step=30)
    last = T0 + timedelta(days=90)
    fresh = score.founder(entity_id, last + timedelta(hours=1))

    half_life = score.MOMENTUM_HALFLIFE_DAYS
    trends, mus = [], []
    for gap in (0, half_life, 2 * half_life, 4 * half_life, 8 * half_life):
        got = score.founder(entity_id, last + timedelta(days=gap, hours=1))
        trends.append(got.trend)
        mus.append(got.mu)

    assert all(a > b > 0 for a, b in zip(trends, trends[1:])), trends
    # One half-life of silence halves the momentum; two quarter it.
    assert trends[1] == pytest.approx(trends[0] / 2, rel=0.05)
    assert trends[2] == pytest.approx(trends[0] / 4, rel=0.05)
    assert mus == sorted(mus)
    # Total drift is bounded by v0 / decay_rate, never an unbounded extrapolation.
    assert mus[-1] - fresh.mu < fresh.trend / score._DECAY_RATE + 1e-9
    assert mus[-1] < 0.5


def test_band_never_exceeds_the_no_evidence_prior() -> None:
    """Staleness widens the band, but uncertainty about a founder can never exceed
    knowing nothing at all."""
    entity_id = uuid4()
    _series(entity_id, [0.7] * 4, step=30)
    prior_band = score.P0[0] ** 0.5
    for gap in (0, 365, 730, 3650):
        got = score.founder(entity_id, T0 + timedelta(days=90 + gap, hours=1))
        assert got.band <= prior_band + 1e-9, (gap, got.band)
    _, far_band = score.forecast(entity_id, T0 + timedelta(days=91), 3650)
    assert far_band <= prior_band + 1e-9


def test_source_noise_and_consistency_are_safe() -> None:
    deck = Event(kind=EventKind.GREEN_FLAG, source=Source.DECK, observed_at=T0)
    proof = Event(kind=EventKind.PROOF_ARTIFACT, source=Source.PROOF_PROTOCOL, observed_at=T0)
    assert score._noise(deck, {}) > score._noise(proof, {})
    web = Event(kind=EventKind.GREEN_FLAG, source=Source.WEB, observed_at=T0)
    assert score._noise(web, {"self_consistency": 0.2}) > score._noise(
        web, {"self_consistency": 1.0}
    )
    assert score._noise(deck, {"self_consistency": 0.0}) < float("inf")


def test_trend_is_expressed_per_year_not_per_day() -> None:
    """Every transition is built from ``_dt_years``, so the velocity component of the
    state is capability-per-YEAR. A renderer that reads it as per-day overstates the
    trend by 365.25x. Asserted here rather than assumed, because it was not."""
    # The velocity-to-level coupling over a short horizon is the horizon itself, and it
    # is expressed in years: one day of propagation couples 1/365.25, not 1.0.
    one_day = 1.0 / 365.25
    assert score._F(one_day)[0, 1] == pytest.approx(one_day, rel=1e-2)
    assert score._F(one_day)[0, 1] < 0.01

    # Behaviourally: a series climbing ~0.30 over one year reports a trend of that
    # order -- not 0.30/365 (per-day) and not 0.30*365 (per-year read as per-day).
    entity_id = uuid4()
    _series(entity_id, [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60], step=61)
    got = score.founder(entity_id, _at(7, step=61))
    assert 0.03 < got.trend < 0.60, got.trend

    # And the documented conversion is the one the API owner must apply.
    assert score.trend_per_days(got.trend, 365.25) == pytest.approx(got.trend)
    assert score.trend_per_days(got.trend, 30) == pytest.approx(got.trend * 30 / 365.25)


def test_forecast_propagates_uncertainty_forward() -> None:
    entity_id = uuid4()
    _series(entity_id, [0.3, 0.4, 0.5, 0.6, 0.7])
    as_of = _at(5)
    now = score.founder(entity_id, as_of)
    mu_30, band_30 = score.forecast(entity_id, as_of, 30)
    mu_90, band_90 = score.forecast(entity_id, as_of, 90)
    assert mu_30 > now.mu
    assert mu_90 >= mu_30
    assert band_90 > band_30 > now.band
    assert 0.0 <= mu_90 <= 1.0


def test_score_model_flag_dispatches_to_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    entity_id = uuid4()
    _series(entity_id, [0.4, 0.6, 0.8, 0.9])
    as_of = _at(4)
    assert score.founder(entity_id, as_of).model == "kalman"
    monkeypatch.setenv("SCORE_MODEL", "beta_binomial")
    result = score.founder(entity_id, as_of)
    assert isinstance(result, FounderScore)
    assert result.model == "beta_binomial"
    assert result.entity_id == entity_id and result.as_of == as_of
    assert 0.0 < result.mu < 1.0
    assert result.band > 0.0 and result.trend > 0.0
    assert len(result.contributing_event_ids) == 4


def test_fallback_honours_contradictions_and_empty_case(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCORE_MODEL", "beta_binomial")
    entity_id, claim_id = uuid4(), uuid4()
    _series(entity_id, [0.2, 0.2])
    tainted = _flag(entity_id, 30, 0.99, payload={"claim_id": str(claim_id)})
    store.append(
        Event(
            kind=EventKind.VALIDATION_RESULT,
            source=Source.VALIDATOR,
            observed_at=T0,
            payload={"claims": [{"claim_id": str(claim_id), "status": "contradicted"}]},
        )
    )
    result = score.founder(entity_id, T0 + timedelta(days=40))
    assert tainted.event_id not in result.contributing_event_ids
    assert result.mu < 0.5
    empty = score.founder(uuid4(), T0)
    assert empty.model == "beta_binomial"
    assert empty.mu == pytest.approx(0.5)
    assert empty.contributing_event_ids == []
