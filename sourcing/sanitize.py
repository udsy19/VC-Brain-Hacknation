"""Injection guard. Owner: B. The Type 5 demo beat.

STRIP, don't reject — and log an INTEGRITY event quoting the offending span.
The trace showing the caught injection IS the demo.
"""

from __future__ import annotations

from schema.events import Event


def sanitize(text: str, *, source_url: str | None = None) -> tuple[str, list[Event]]:
    """Returns (clean_text, integrity_events)."""
    raise NotImplementedError("B: H3-8")
