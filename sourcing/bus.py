"""The one funnel. Owner: B. Inbound decks and outbound scanners take the same path.

normalize -> sanitize -> stamp observed_at -> emit Events. No special cases.
"""

from __future__ import annotations

from schema.events import Event, RawSignal


def ingest(raw: RawSignal) -> list[Event]:
    raise NotImplementedError("B: H3-8")
