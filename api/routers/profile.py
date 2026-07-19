"""The VC profile: survey, decision upload, derived profile, and the gap.

Owner: personalisation layer (docs/DIFFERENTIATOR.md §2).

Every route here requires a session, because a profile without an owner is meaningless.
That is the ONLY part of the API that requires one — the objective ranking in
api/main.py and api/routers/companies.py stays open, so a login outage costs
personalisation and nothing else (§1).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from api.auth import required_user
from memory import profiles
from schema.vc import SURVEY_QUESTIONS, ProfileUpdate, SurveySubmission, User

router = APIRouter(prefix="/profile", tags=["profile"])

MAX_UPLOAD_BYTES = 2 * 1024 * 1024


@router.get("")
def get_profile(user: User = Depends(required_user)) -> dict:
    """The profile and everything derived from it.

    `derived.not_inferred` is part of the payload, not an afterthought: a client that
    renders only the populated fields would present a thin profile as a complete one.
    """
    return profiles.get_profile(user.user_id).model_dump(mode="json")


@router.put("")
def put_profile(body: ProfileUpdate, user: User = Depends(required_user)) -> dict:
    """Partial update — an omitted field is left alone rather than cleared."""
    profiles.update_profile(
        user.user_id,
        fund_name=body.fund_name,
        focus_sectors=body.focus_sectors,
        stated_red_lines=body.stated_red_lines,
    )
    return profiles.get_profile(user.user_id).model_dump(mode="json")


@router.get("/survey")
def get_survey(user: User = Depends(required_user)) -> dict:
    """The question set plus whatever this user has already answered.

    The catalog is served rather than hard-coded in the client so the signals and the
    questions can never drift apart — a question the frontend shows that the derivation
    has no signals for would contribute silently nothing.
    """
    answered = {a.question_id: str(a.choice) for a in profiles.get_survey(user.user_id)}
    return {
        "questions": [q.model_dump(mode="json") for q in SURVEY_QUESTIONS],
        "answers": answered,
        "answered": len(answered),
        "total": len(SURVEY_QUESTIONS),
    }


@router.post("/survey")
def post_survey(body: SurveySubmission, user: User = Depends(required_user)) -> dict:
    """Submit answers. Partial submissions are legal and are NOT padded — an unanswered
    question contributes nothing and lowers the confidence, rather than being imputed."""
    known = {q.id for q in SURVEY_QUESTIONS}
    unknown = sorted({a.question_id for a in body.answers} - known)
    stored = profiles.save_survey(user.user_id, body.answers)
    derived = profiles.derive(user.user_id)
    return {
        "stored": stored,
        # Named rather than silently dropped, so a stale client is visible instead of
        # quietly contributing nothing.
        "ignored_unknown_question_ids": unknown,
        "answered": derived.survey_answered,
        "total": derived.survey_total,
        "derived": derived.model_dump(mode="json"),
    }


@router.post("/decisions")
async def post_decisions(
    request: Request,
    replace: bool = True,
    user: User = Depends(required_user),
) -> dict:
    """Upload a past-decision history as CSV or JSON.

    Accepts either a multipart file field or a raw request body, because the demo posts
    with curl and the dashboard posts a form, and requiring one shape would break the
    other. Format is detected from the CONTENT, not the filename — an extension is a
    claim, not a fact.

    `replace=true` (the default) swaps the whole history: a re-upload is a correction of
    the file, not an append, and appending would double every row on a second upload.
    """
    content = await _read_upload(request)
    if not content.strip():
        raise HTTPException(400, "empty upload — expected CSV or JSON decision rows")

    decisions, result = profiles.parse_decisions(content)
    profiles.save_decisions(user.user_id, decisions, replace=replace)
    derived = profiles.derive(user.user_id)
    return {
        "upload": result.model_dump(mode="json"),
        "derived": derived.model_dump(mode="json"),
    }


async def _read_upload(request: Request) -> str:
    """Multipart file field if there is one, raw body otherwise."""
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        for value in form.values():
            data = await value.read() if hasattr(value, "read") else value
            if isinstance(data, bytes):
                data = data.decode("utf-8", errors="replace")
            if isinstance(data, str) and data.strip():
                return data
        raise HTTPException(400, "multipart upload contained no file")

    body = await request.body()
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"upload exceeds {MAX_UPLOAD_BYTES} bytes")
    return body.decode("utf-8", errors="replace")


@router.get("/gap")
def get_gap(user: User = Depends(required_user)) -> dict:
    """The stated-vs-revealed divergence (§2.3).

    `uncomputable` travels with the findings on purpose. A gap report that returned only
    what it could compare would read as agreement on every dimension it lacked data for,
    which would be a fabrication about the user.
    """
    report = profiles.gap(user.user_id)
    derived = profiles.derive(user.user_id)
    return {
        **report.model_dump(mode="json"),
        "personalisation_enabled": derived.personalisation_enabled,
        "personalisation_reason": derived.personalisation_reason,
    }
