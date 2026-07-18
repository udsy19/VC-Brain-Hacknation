"""Time-machine backtest. Owner: D, with A + C. Proof #1 of the whole pitch.

Replays truncated historical sources through the SAME code path as live, with as_of
pinned before the founder was known. If it needs a special mode, it isn't a backtest.

assert_no_lookahead() is what makes the claim credible rather than merely asserted.
"""

from __future__ import annotations

from datetime import datetime

from schema.events import Event


class LookaheadError(AssertionError):
    """Raised loudly. Never caught, never downgraded to a warning."""


def assert_no_lookahead(events: list[Event], as_of: datetime) -> None:
    leaked = [e for e in events if e.observed_at > as_of]
    if leaked:
        raise LookaheadError(
            f"{len(leaked)} event(s) from the future reached the scorer at as_of={as_of}: "
            f"{[str(e.event_id) for e in leaked[:3]]}"
        )


def replay(company_id, as_of: datetime) -> dict:
    raise NotImplementedError("D: H12-16")
