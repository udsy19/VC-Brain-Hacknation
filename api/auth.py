"""Password hashing, sessions, and the request dependencies that read them.

Owner: personalisation layer (docs/DIFFERENTIATOR.md §1).

Email + password with argon2id, sessions in the database, httpOnly cookie. No
third-party IdP on purpose: a demo that depends on an OAuth callback surviving a
conference network is a demo that fails on stage.

THE RULE THAT OUTRANKS EVERYTHING ELSE HERE (§1): the CORE product keeps working
unauthenticated. Personalisation requires a session; the objective ranking does not. So
`optional_user` returns None for anything it cannot resolve — no session, a bad session,
an expired session, even a database that will not answer — and never raises. A broken
login degrades to the objective product, never to a blank page. `required_user` is used
ONLY on the routes that are meaningless without an owner.

Never logged, never returned, never serialised: the password and the hash. The hash has
exactly one read path (memory.profiles.password_hash_for) and it terminates here.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
from datetime import datetime
from uuid import UUID

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, VerifyMismatchError
from fastapi import Depends, HTTPException, Request, Response

from memory import profiles
from schema.vc import User

log = logging.getLogger(__name__)

COOKIE_NAME = "vcbrain_session"

# argon2id at the library defaults, which are the RFC 9106 low-memory profile. Left
# alone deliberately: hand-tuned KDF parameters are a way to make hashing weaker while
# feeling thorough, and the defaults are revised by people who track the hardware.
_hasher = PasswordHasher()

# A hash of a throwaway password, verified against when the email does not exist. Without
# it, a missing account returns in microseconds and a real one takes ~50ms, which turns
# every login into an account-existence oracle no matter what the response body says.
_DUMMY_HASH = _hasher.hash(secrets.token_urlsafe(32))

#: One message for every credential failure. Which half was wrong is not the client's
#: business — telling them narrows an attack from "guess an email and a password" to
#: "guess a password", and hands out the user list on the way.
BAD_CREDENTIALS = "invalid email or password"


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, stored_hash: str | None) -> bool:
    """Constant-time-ish by construction: argon2's verify does the comparison, and a
    missing hash still costs a full verify against the dummy so the timing matches."""
    try:
        _hasher.verify(stored_hash or _DUMMY_HASH, password)
        return stored_hash is not None
    except (VerifyMismatchError, VerificationError):
        return False
    except Exception:  # noqa: BLE001 - a malformed stored hash is a failed login, not a 500
        log.warning("password verification failed on a malformed stored hash")
        return False


def new_token() -> str:
    """The raw session token. Handed to the client once, in a cookie, and never stored."""
    return secrets.token_urlsafe(32)


def token_hash(token: str) -> str:
    """What actually goes in the database. A read of `sessions` yields hashes, not live
    credentials."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def start_session(response: Response, user_id: UUID) -> datetime:
    token = new_token()
    expires = profiles.create_session(user_id, token_hash(token))
    response.set_cookie(
        COOKIE_NAME,
        token,
        httponly=True,  # JavaScript must never be able to read this
        samesite="lax",
        secure=_secure_cookies(),
        expires=expires,
        path="/",
    )
    return expires


def end_session(request: Request, response: Response) -> None:
    token = request.cookies.get(COOKIE_NAME)
    if token:
        profiles.delete_session(token_hash(token))
    response.delete_cookie(COOKIE_NAME, path="/")


def _secure_cookies() -> bool:
    """Secure everywhere except local development, where there is no https to set it on
    and a Secure cookie would simply never be stored."""
    if os.getenv("VCBRAIN_INSECURE_COOKIES") == "1":
        return False
    return os.getenv("VERCEL_ENV", "") not in ("", "development")


# ---------------------------------------------------------------------------
# Request dependencies
# ---------------------------------------------------------------------------


def optional_user(request: Request) -> User | None:
    """The session's user, or None — for ANY reason, including failure.

    This function must not raise. It is what keeps the core product alive when auth is
    broken: an unreachable database or a corrupt cookie makes a request anonymous, and
    an anonymous request still gets the full objective ranking.
    """
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    try:
        return profiles.user_for_session(token_hash(token))
    except Exception:  # noqa: BLE001 - auth degrades to anonymous, never to an error page
        log.warning("session lookup failed; continuing unauthenticated", exc_info=True)
        return None


def required_user(user: User | None = Depends(optional_user)) -> User:
    """For routes that are meaningless without an owner. Used ONLY on the personalisation
    surface — never on anything that serves the objective product."""
    if user is None:
        raise HTTPException(401, "authentication required for personalisation")
    return user


def check_rate_limit(email: str) -> None:
    """Raises 429 when this email is locked out. Failing OPEN on a storage error is
    deliberate: a throttle that cannot read its own state must not become an outage."""
    try:
        remaining = profiles.lockout_remaining(email)
    except Exception:  # noqa: BLE001
        log.warning("login throttle unavailable; allowing the attempt", exc_info=True)
        return
    if remaining is not None:
        raise HTTPException(
            429,
            f"too many failed login attempts; try again in {int(remaining.total_seconds()) + 1}s",
        )
