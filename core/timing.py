"""Signal-to-decision time.

The rubric asks for this under Investment Utility: "signal-to-decision time
instrumented". It is the answer to "how long does your system take to go from a
founder appearing in the world to a decision about them" — and unlike most of what
this system reports, it is a fact about US, not about the founder.

Two clocks, deliberately separate, because they answer different questions:

  SIGNAL AGE      earliest observed_at -> the decision's as_of.
                  How long the evidence sat in the world before we ruled on it.
                  Measured in days, and bounded by when the founder started
                  building, not by how fast we compute.

  COMPUTE TIME    wall clock spent producing the decision, per stage.
                  Measured in milliseconds. This is the one an engineer can
                  actually shorten.

Reporting one without the other is how a system claims to be fast: a 900ms compute
time next to evidence that sat unexamined for 400 days is not speed. Both travel
together or neither is meaningful.

Nothing here is persisted to the event log. These are measurements of the system's
own behaviour, not observations about the world, and the log is reserved for the
latter — an event with an `observed_at` of "when we happened to run" would corrupt
every as_of query that reads it.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator


@dataclass
class Stages:
    """Wall-clock milliseconds per pipeline stage, in the order they ran."""

    marks: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            # += so a stage entered more than once accumulates rather than
            # reporting only its last visit, which would understate the total.
            self.marks[name] = self.marks.get(name, 0.0) + (time.perf_counter() - start) * 1000

    @property
    def total_ms(self) -> float:
        return round(sum(self.marks.values()), 1)

    def as_dict(self) -> dict:
        return {k: round(v, 1) for k, v in self.marks.items()}


def signal_age_days(earliest_observed_at: datetime | None, as_of: datetime) -> float | None:
    """Days between the first evidence existing and the decision being taken.

    None when there is no evidence — which is a real state (the cold-start
    archetype) and must not be reported as an age of zero. Zero would read as
    "instant", the opposite of "we have nothing to go on".
    """
    if earliest_observed_at is None:
        return None
    return round((as_of - earliest_observed_at).total_seconds() / 86400.0, 1)


def measure(company_id, as_of: datetime, stages: Stages, events: list | None = None) -> dict:
    """Assemble the report. Both clocks, or neither."""
    observed = [e.observed_at for e in (events or []) if getattr(e, "observed_at", None)]
    earliest = min(observed) if observed else None
    age = signal_age_days(earliest, as_of)

    return {
        "company_id": str(company_id),
        "as_of": as_of.isoformat(),
        # What we can shorten.
        "compute_ms": stages.total_ms,
        "stages_ms": stages.as_dict(),
        # What we cannot — this is the founder's timeline, not ours.
        "signal_age_days": age,
        "earliest_signal_at": earliest.isoformat() if earliest else None,
        "evidence_count": len(observed),
        "note": (
            "compute_ms is wall clock spent deciding; signal_age_days is how long the "
            "earliest evidence existed before we decided. A fast compute over stale "
            "evidence is not a fast decision."
        ),
    }
