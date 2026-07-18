"""hn scanner. Owner: B. Emits RawSignal -> bus.ingest().

observed_at must come from the source's own timestamp. If a source cannot give a
real one, it does not get ingested. Cache raw responses to data/raw/.
"""

from __future__ import annotations

from schema.events import RawSignal


def scan(query: str, limit: int = 50) -> list[RawSignal]:
    raise NotImplementedError("B: H1-3")
