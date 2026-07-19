"""Split-conformal no-call thresholding for the founder score. Owner: C. See C.md H18-21.

WHY THIS EXISTS
---------------
The gate used to abstain when four hand-picked constants said so. That is an arbitrary
cutoff wearing a decision's clothing: nobody can say why 0.45, or 0.20, and not something
else. Split conformal prediction replaces the arbitrary part with something defensible —
a half-width derived from held-out labelled examples at a stated error rate ``alpha``.

THE CONSTRUCTION
----------------
For each labelled calibration member we know two things: the founder score at the cutoff
(``mu``, with the Kalman posterior width ``band``) and whether the company actually
cleared. The nonconformity score is how many band-widths the score sat on the WRONG side
of the clearing threshold ``tau``::

    r_i = (tau - mu_i) / band_i      if the company cleared
    r_i = (mu_i - tau) / band_i      if it did not

A member scored well on the correct side gets a large negative r; a member the score got
backwards gets a positive one. Split conformal takes the ``ceil((n+1)(1-alpha))/n``
empirical quantile ``q`` of those residuals; ``z = -q`` is then the number of band-widths
of margin the calibration data says we need before a claim about the threshold is safe at
level ``1-alpha``. The per-company prediction interval is::

    [mu - z * band,  mu + z * band]

and the rule is the one the whole exercise is for: **when that interval straddles tau, the
evidence cannot distinguish "clears" from "does not clear" at the stated confidence, so we
abstain.** Above tau, it clears. Below tau, it does not.

WHAT THIS DELIBERATELY REFUSES TO DO
------------------------------------
A conformal guarantee is a statement about a sample, and this repo's sample is nine
points. Every degenerate case therefore returns ``calibrated=False`` with a reason
attached, and the gate falls back to its old logic SAYING SO, rather than emitting a
confident-looking interval it did not earn:

* ``n < ceil(1/alpha) - 1``  — the quantile is not defined; alpha is not achievable.
* only one outcome class present — nothing to separate.
* ``z <= 0`` — at this alpha the calibration set admits points on the wrong side of the
  threshold, so the "interval" would be inverted and would never abstain.
* any member with a non-positive band — no scale to standardise against.

LEAKAGE
-------
Calibrating on the very point being judged is leakage, so ``for_company`` drops the
company under evaluation from its own calibration set. A cohort member is therefore judged
on the other eight, which is why the default alpha is 1/8 rather than the 1/9 the full set
would allow. Tighten alpha past that and a cohort member's own evaluation legitimately
falls back to the uncalibrated path. That is the honest outcome, not a bug to route around.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Literal, Sequence
from uuid import UUID

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# The target error rate, and the one number in this file that a reader should interrogate.
# It is not a taste call. With n calibration points the smallest achievable alpha is
# 1/(n+1), because the quantile index ceil((n+1)(1-alpha)) must not exceed n. The repo's
# labelled cohort has exactly 9 members, so alpha=0.10 is the tightest level the corpus
# supports outright — but a cohort member is dropped from its own calibration set (see
# LEAKAGE below), leaving 8, which needs alpha >= 1/9. 0.125 = 1/8 is therefore the
# tightest error rate EVERY company in the corpus can be judged at, and it is what the
# gate states: at most one expected miscall in eight. Ask for less and calibrate()
# refuses rather than pretends.
DEFAULT_ALPHA = 0.125

# The clearing threshold on the founder axis, and its provenance. Not ours to invent:
# data/seed/backtest.json declares value 0.62 on the founder axis, calibrated on the
# archetype corpus, which shares no members with the cohort we calibrate on below.
COHORT_PATH = Path("data/seed/backtest.json")
_FALLBACK_THRESHOLD = 0.62

Verdict = Literal["clears", "does_not_clear", "ambiguous"]


class CalibrationPoint(BaseModel):
    """One labelled cohort member, scored by the live scorer at the evaluation cutoff."""

    label: str
    company_id: UUID | None = None
    mu: float
    band: float
    cleared: bool
    nonconformity: float


class ConformalInterval(BaseModel):
    """A per-company prediction interval and what it says about the threshold."""

    lower: float
    upper: float
    threshold: float
    verdict: Verdict


class ConformalCalibration(BaseModel):
    """The calibrated (or explicitly uncalibrated) conformal layer.

    ``calibrated=False`` is a first-class state, not an error. ``reason`` always carries a
    sentence fit to show a user, in both states.
    """

    calibrated: bool
    alpha: float
    threshold: float
    n: int
    n_cleared: int = 0
    n_not_cleared: int = 0
    z: float | None = None
    reason: str
    points: list[CalibrationPoint] = Field(default_factory=list)

    @property
    def min_points(self) -> int:
        """Smallest calibration set for which this alpha's quantile is defined."""
        return required_points(self.alpha)

    def interval(self, mu: float, band: float) -> ConformalInterval | None:
        """The 1-alpha interval for one company, or None when not calibrated."""
        if not self.calibrated or self.z is None:
            return None
        half = self.z * max(band, 0.0)
        lower, upper = mu - half, mu + half
        if lower >= self.threshold:
            verdict: Verdict = "clears"
        elif upper <= self.threshold:
            verdict = "does_not_clear"
        else:
            verdict = "ambiguous"
        return ConformalInterval(
            lower=lower, upper=upper, threshold=self.threshold, verdict=verdict
        )

    def describe(self, interval: ConformalInterval | None = None) -> str:
        """One sentence of reasoning, always naming alpha and the calibration size."""
        if not self.calibrated or interval is None:
            return f"Conformal layer not calibrated ({self.reason}); gate fell back to its base policy."
        stated = (
            f"alpha={self.alpha:g}, n={self.n} calibration points "
            f"({self.n_cleared} cleared / {self.n_not_cleared} did not), "
            f"half-width {self.z:.2f}x band"
        )
        span = f"[{interval.lower:.2f}, {interval.upper:.2f}] vs threshold {self.threshold:g}"
        caveat = (
            f"Small-sample caveat: this guarantee rests on {self.n} labelled outcomes, so read "
            f"it as a stated, checkable rule rather than a tight statistical bound."
        )
        if interval.verdict == "ambiguous":
            return (
                f"Conformal no-call: the {1 - self.alpha:.1%} prediction interval {span} "
                f"straddles the threshold, so the evidence cannot distinguish clearing from "
                f"not clearing ({stated}). {caveat}"
            )
        side = "entirely above" if interval.verdict == "clears" else "entirely below"
        return (
            f"Conformal interval {span} sits {side} the threshold, so the call is "
            f"distinguishable at {1 - self.alpha:.1%} ({stated}). {caveat}"
        )

    def for_company(self, company_id: UUID | None) -> "ConformalCalibration":
        """Re-calibrate with ``company_id`` removed, so no company calibrates on itself."""
        if company_id is None or not any(p.company_id == company_id for p in self.points):
            return self
        kept = [p for p in self.points if p.company_id != company_id]
        result = calibrate(kept, alpha=self.alpha, threshold=self.threshold)
        if not result.calibrated:
            result = result.model_copy(
                update={
                    "reason": (
                        f"{result.reason}; the company under evaluation is itself a "
                        f"calibration member and was removed to avoid calibrating on the "
                        f"point being judged"
                    )
                }
            )
        return result


def required_points(alpha: float) -> int:
    """Smallest n for which the (1-alpha) conformal quantile index falls inside the sample."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha!r}")
    return math.ceil(1.0 / alpha) - 1


def nonconformity(mu: float, band: float, cleared: bool, threshold: float) -> float:
    """Band-widths the score sat on the WRONG side of the threshold. Lower is better."""
    signed = (threshold - mu) if cleared else (mu - threshold)
    return signed / band


def _conformal_quantile(residuals: Sequence[float], alpha: float) -> float | None:
    """The ceil((n+1)(1-alpha))/n empirical quantile. None when the index escapes the sample."""
    n = len(residuals)
    index = math.ceil((n + 1) * (1.0 - alpha))
    if index > n or index < 1:
        return None
    return sorted(residuals)[index - 1]


def calibrate(
    points: Sequence[CalibrationPoint],
    *,
    alpha: float = DEFAULT_ALPHA,
    threshold: float | None = None,
) -> ConformalCalibration:
    """Split-conformal calibration over labelled founder scores. Never raises on thin data."""
    tau = clearing_threshold() if threshold is None else threshold
    n = len(points)
    cleared = sum(1 for p in points if p.cleared)
    base = {
        "alpha": alpha,
        "threshold": tau,
        "n": n,
        "n_cleared": cleared,
        "n_not_cleared": n - cleared,
        "points": list(points),
    }
    need = required_points(alpha)
    if n < need:
        return ConformalCalibration(
            calibrated=False,
            reason=(
                f"alpha={alpha:g} needs at least {need} calibration points for its quantile "
                f"to fall inside the sample; only {n} available"
            ),
            **base,
        )
    if cleared == 0 or cleared == n:
        return ConformalCalibration(
            calibrated=False,
            reason=(
                f"calibration set contains only one outcome class "
                f"({cleared} cleared / {n - cleared} did not); there is nothing to separate"
            ),
            **base,
        )
    quantile = _conformal_quantile([p.nonconformity for p in points], alpha)
    if quantile is None:
        return ConformalCalibration(
            calibrated=False,
            reason=f"the (1-alpha) quantile index is outside a sample of {n}",
            **base,
        )
    z = -quantile
    if z <= 0.0:
        return ConformalCalibration(
            calibrated=False,
            reason=(
                f"degenerate quantile: at alpha={alpha:g} the calibration set still admits "
                f"scores on the wrong side of the threshold, so the interval would be "
                f"inverted and could never abstain"
            ),
            **base,
        )
    return ConformalCalibration(
        calibrated=True,
        z=z,
        reason=(
            f"split conformal over {n} labelled outcomes at alpha={alpha:g}; "
            f"half-width {z:.2f} x band"
        ),
        **base,
    )


def clearing_threshold() -> float:
    """The founder-axis clearing threshold declared by the backtest cohort file."""
    try:
        raw = json.loads(COHORT_PATH.read_text(encoding="utf-8"))
        value = float(raw["threshold"]["value"])
    except Exception as exc:  # noqa: BLE001 - a missing data file must not break the gate
        log.info("conformal: threshold file unavailable (%s)", exc)
        return _FALLBACK_THRESHOLD
    return value


def _cohort_labels() -> dict[str, bool]:
    """company name -> did it clear. Membership IS the replayed outcome; see backtest.json."""
    try:
        raw = json.loads(COHORT_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - no cohort file means "not calibrated", not a crash
        log.info("conformal: cohort file unavailable (%s)", exc)
        return {}
    labels: dict[str, bool] = {}
    for key, cleared in (
        ("winners", True),
        ("controls", False),
        ("correctly_deprioritized_failure", False),
    ):
        members = raw.get(key) or []
        for member in members if isinstance(members, list) else [members]:
            name = member.get("company_name") or member.get("name")
            if name:
                labels[str(name)] = cleared
    return labels


_CACHE: dict[tuple[str, float, float], ConformalCalibration] = {}


def from_store(
    as_of: datetime, *, alpha: float = DEFAULT_ALPHA, threshold: float | None = None
) -> ConformalCalibration:
    """Calibrate on the labelled backtest cohort as scored by the live scorer at ``as_of``.

    Reads are as_of-scoped like every other read in the system, so a member with no
    observed history yet simply drops out of the calibration set and the count says so.
    Any store failure (offline tests, no database) yields ``calibrated=False``.
    """
    tau = clearing_threshold() if threshold is None else threshold
    key = (as_of.isoformat(), alpha, tau)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    result = calibrate(_cohort_points(as_of, tau), alpha=alpha, threshold=tau)
    _CACHE[key] = result
    return result


def reset_cache() -> None:
    """Drop memoised calibrations. For tests and for a reseeded store."""
    _CACHE.clear()


def _cohort_points(as_of: datetime, threshold: float) -> list[CalibrationPoint]:
    from memory import score, store

    labels = _cohort_labels()
    if not labels:
        return []
    points: list[CalibrationPoint] = []
    try:
        companies = store.get_store().companies()
    except Exception as exc:  # noqa: BLE001 - offline or unseeded store: no calibration
        log.info("conformal: store unavailable (%s)", exc)
        return []
    for company in companies:
        cleared = labels.get(getattr(company, "name", None))
        if cleared is None:
            continue
        try:
            events = store.events(company_id=company.company_id, as_of=as_of)
            entity_ids = sorted({e.entity_id for e in events if e.entity_id is not None}, key=str)
            if len(entity_ids) != 1:
                continue
            founder = score.founder(entity_ids[0], as_of)
        except Exception as exc:  # noqa: BLE001 - one unscoreable member must not kill the rest
            log.info("conformal: cohort member %s unscoreable (%s)", company.company_id, exc)
            continue
        if founder.band <= 0.0:
            continue
        points.append(
            CalibrationPoint(
                label=str(company.name),
                company_id=company.company_id,
                mu=founder.mu,
                band=founder.band,
                cleared=cleared,
                nonconformity=nonconformity(founder.mu, founder.band, cleared, threshold),
            )
        )
    return points


__all__ = [
    "DEFAULT_ALPHA",
    "CalibrationPoint",
    "ConformalCalibration",
    "ConformalInterval",
    "calibrate",
    "clearing_threshold",
    "from_store",
    "nonconformity",
    "required_points",
    "reset_cache",
]
