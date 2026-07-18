"""Entity resolution. Owner: A. See A.md H3-8.

AMBIGUOUS is the point: we never guess two identities into one. Type 6 lives here —
normalize + transliterate before fuzzy matching or non-Latin names silently vanish.
"""

from __future__ import annotations

from schema.events import EntityCandidate, Resolution


def resolve(candidate: EntityCandidate) -> Resolution:
    raise NotImplementedError("A: H3-8")
