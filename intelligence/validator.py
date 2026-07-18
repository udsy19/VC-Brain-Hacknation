"""Per-claim validation, four states. Owner: C. See C.md H3-8.

Independent source = core.search (Tavily). Rules that keep this honest:
  - a VERIFIED with no stored snippet+URL is NOT_ATTEMPTED
  - search results are UNTRUSTED (a founder can plant a page) -> llm.complete(untrusted=)
  - empty results -> UNVERIFIABLE, NEVER CONTRADICTED
  - compare observed_at: "$40K ARR" in March vs "pre-revenue" in January is GROWTH
Contradiction reprices the CLAIM, not the deal.
"""

from __future__ import annotations

from uuid import UUID

from schema.events import ClaimVerdict


def check_claims(company_id: UUID) -> list[ClaimVerdict]:
    raise NotImplementedError("C: H3-8")
