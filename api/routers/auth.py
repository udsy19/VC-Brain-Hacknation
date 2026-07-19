"""Register / login / logout / me. Owner: personalisation layer (DIFFERENTIATOR §1).

Thin: every decision of consequence lives in api/auth.py or memory/profiles.py. The one
thing this file is careful about is what it tells the client on failure — see the
`BAD_CREDENTIALS` note below.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from api import auth
from memory import profiles
from schema.vc import LoginRequest, RegisterRequest, User

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", status_code=201)
def register(body: RegisterRequest, response: Response) -> dict:
    """Create an account, an empty profile, and a session.

    Registering DOES disclose that an email is taken, unlike login. That is a deliberate
    asymmetry and the standard trade-off: without it the user who already has an account
    is told "success" and then cannot log in with the password they just typed. The
    throttle on login is what stops registration being used to enumerate the user list
    at speed.
    """
    hashed = auth.hash_password(body.password)
    try:
        user = profiles.create_user(body.email, hashed)
    except ValueError:
        raise HTTPException(409, "an account with that email already exists") from None

    profiles.ensure_profile(user.user_id)
    if body.fund_name:
        profiles.update_profile(user.user_id, fund_name=body.fund_name)
    expires = auth.start_session(response, user.user_id)
    return {
        "user": user.model_dump(mode="json"),
        "session_expires_at": expires.isoformat(),
        # Said out loud on the way in, so nobody reads a fresh account as a configured one.
        "personalisation": "off until a survey and a decision history are submitted",
    }


@router.post("/login")
def login(body: LoginRequest, response: Response) -> dict:
    """One failure message for every kind of credential failure.

    Unknown email and wrong password return the identical 401, and the verify runs even
    when the email is unknown so the two take the same time. Anything else turns login
    into an oracle for which addresses have accounts.
    """
    auth.check_rate_limit(body.email)

    stored = profiles.password_hash_for(body.email)
    if not auth.verify_password(body.password, stored):
        profiles.record_login_failure(body.email)
        raise HTTPException(401, auth.BAD_CREDENTIALS)

    user = profiles.get_user_by_email(body.email)
    if user is None:  # a hash with no user is not a state that should exist
        raise HTTPException(401, auth.BAD_CREDENTIALS)

    profiles.clear_login_failures(body.email)
    expires = auth.start_session(response, user.user_id)
    return {"user": user.model_dump(mode="json"), "session_expires_at": expires.isoformat()}


@router.post("/logout")
def logout(request: Request, response: Response) -> dict:
    """Idempotent, and never an error. Logging out of a session that is already gone is
    a request for a state that already holds."""
    auth.end_session(request, response)
    return {"ok": True}


@router.get("/me")
def me(user: User | None = Depends(auth.optional_user)) -> dict:
    """200 with `authenticated: false` rather than a 401.

    This route is what the frontend asks on boot to decide whether to show a personal
    rank. A 401 here would be an error the client has to special-case in order to render
    the perfectly valid anonymous state, and clients that get that wrong render a blank
    page for logged-out users — the exact failure §1 forbids.
    """
    if user is None:
        return {
            "authenticated": False,
            "user": None,
            "personalisation_enabled": False,
            "reason": "no session — the core objective ranking is unaffected",
        }
    derived = profiles.derive(user.user_id)
    return {
        "authenticated": True,
        "user": user.model_dump(mode="json"),
        "personalisation_enabled": derived.personalisation_enabled,
        "reason": derived.personalisation_reason,
    }
