"""Inbound applications: company name + deck upload, and the status funnel.

Owner: B. SHARED.md S1 — the inbound half of "inbound (deck+name) + outbound (scanners +
PPR graph diffusion) -> activate -> ONE FUNNEL".

Thin on purpose. The upload is validated, converged and funnelled in sourcing/intake.py
(which routes the PDF through sourcing/deck.py -> sourcing/bus.py, the only path that
sanitizes founder-supplied text and keeps a slide id on every span), and the status is
derived there too. This file turns exceptions into HTTP codes and does nothing else.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.routers.deps import resolve_as_of
from sourcing import intake

router = APIRouter(prefix="/applications", tags=["applications"])


@router.post("")
async def post_application(
    company_name: str = Form(...),
    deck: UploadFile = File(...),
    submitted_by: Optional[str] = Form(default=None),
    founder_name: Optional[str] = Form(default=None),
    founder_email: Optional[str] = Form(default=None),
) -> dict:
    """Submit a company name plus a deck.

    A rejection is a 400 with a sentence — non-PDF, empty, oversized or unreadable. The
    founder on the other end of this gets told what was wrong with their file, never a
    stack trace, and nothing is written before the file passes.
    """
    content = await deck.read()
    try:
        return intake.submit(
            company_name,
            content,
            filename=deck.filename,
            submitted_by=submitted_by,
            founder_name=founder_name,
            founder_email=founder_email,
        )
    except intake.Rejected as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("")
def get_applications(company_id: Optional[str] = None) -> dict:
    """Every application received, newest first."""
    items = intake.applications(company_id)
    return {"count": len(items), "items": items}


@router.get("/{application_id}")
def get_application(application_id: str) -> dict:
    row = intake.get(application_id)
    if row is None:
        raise HTTPException(404, f"no such application: {application_id}")
    row["arrival"] = intake.arrival(row["company_id"])
    return row


@router.get("/{application_id}/status")
def get_status(application_id: str, as_of: Optional[datetime] = None) -> dict:
    """Where this application actually is: received -> ingested -> screened -> gated
    -> decided.

    Recomputed on every call from the event log, the gate and the outbound tables.
    Nothing here reads a stored state, because there isn't one.
    """
    try:
        return intake.status(application_id, as_of=resolve_as_of(as_of))
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
