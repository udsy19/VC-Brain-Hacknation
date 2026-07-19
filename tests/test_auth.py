"""Auth: hashing, sessions, throttling, and the invariant that outranks all of them —
the core product keeps working with no session at all (DIFFERENTIATOR §1).

Offline by construction: conftest points VCBRAIN_DB_PATH at a temp SQLite file, which
db.backend() honours over any DATABASE_URL on disk.
"""

from __future__ import annotations

import shutil
import uuid
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import auth
from api.main import app
from memory import profiles


@pytest.fixture(autouse=True)
def _isolated_seed_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the seed fixtures at a throwaway copy for the duration of the test.

    `GET /thesis` makes core.thesis.load() backfill a derived `clearing_score` into
    data/seed/thesis.json. Against the real directory that is a tracked repo file being
    rewritten mid-suite, which moved the evidence bar under tests/test_intelligence_gate
    — that module reads the bar once at import, so a later write made its constants
    disagree with the running gate. Isolating the directory keeps this file's reads from
    having consequences anywhere else.
    """
    seed = tmp_path / "seed"
    shutil.copytree(Path("data/seed"), seed)
    monkeypatch.setenv("VCBRAIN_SEED_DIR", str(seed))
    yield


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clean_tables():
    """Each test starts with no users, sessions or throttle state."""
    c = profiles.conn()
    for table in (
        "vc_decisions",
        "vc_survey_answers",
        "vc_profiles",
        "sessions",
        "login_attempts",
        "users",
    ):
        c.execute(f"delete from {table}")
    yield


def _email() -> str:
    return f"vc-{uuid.uuid4().hex[:12]}@fund.example"


PASSWORD = "correct-horse-battery-staple"


def register(client: TestClient, email: str | None = None, password: str = PASSWORD):
    return client.post(
        "/auth/register", json={"email": email or _email(), "password": password}
    )


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def test_hash_is_argon2id_and_salted():
    a = auth.hash_password(PASSWORD)
    b = auth.hash_password(PASSWORD)
    assert a.startswith("$argon2id$")
    assert a != b, "identical passwords must not produce identical hashes (no salt)"
    assert PASSWORD not in a


def test_verify_round_trip_and_rejection():
    stored = auth.hash_password(PASSWORD)
    assert auth.verify_password(PASSWORD, stored) is True
    assert auth.verify_password("wrong-password-entirely", stored) is False


def test_verify_against_missing_hash_is_false_not_an_error():
    """An unknown email still runs a full verify, so the timing matches a real account."""
    assert auth.verify_password(PASSWORD, None) is False


def test_verify_against_malformed_hash_is_a_failed_login_not_a_crash():
    assert auth.verify_password(PASSWORD, "not-a-hash") is False


# ---------------------------------------------------------------------------
# Register / login / logout / me
# ---------------------------------------------------------------------------


def test_register_creates_user_session_and_empty_profile(client: TestClient):
    email = _email()
    r = register(client, email)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["user"]["email"] == email
    assert auth.COOKIE_NAME in r.cookies

    me = client.get("/auth/me").json()
    assert me["authenticated"] is True
    # A brand-new account has no survey and no decisions, so personalisation is off.
    assert me["personalisation_enabled"] is False


def test_password_and_hash_never_appear_in_any_response(client: TestClient):
    email = _email()
    r = register(client, email)
    login = client.post("/auth/login", json={"email": email, "password": PASSWORD})
    me = client.get("/auth/me")
    for response in (r, login, me):
        text = response.text
        assert PASSWORD not in text
        assert "argon2" not in text
        assert "password_hash" not in text


def test_email_is_normalised_so_case_does_not_create_a_second_account(client: TestClient):
    email = _email()
    assert register(client, email).status_code == 201
    assert register(client, email.upper()).status_code == 409

    client.cookies.clear()
    r = client.post("/auth/login", json={"email": email.upper(), "password": PASSWORD})
    assert r.status_code == 200


def test_duplicate_registration_is_rejected(client: TestClient):
    email = _email()
    assert register(client, email).status_code == 201
    assert register(client, email).status_code == 409


def test_short_password_is_rejected(client: TestClient):
    r = client.post("/auth/register", json={"email": _email(), "password": "short"})
    assert r.status_code == 422


def test_malformed_email_is_rejected(client: TestClient):
    r = client.post("/auth/register", json={"email": "not-an-email", "password": PASSWORD})
    assert r.status_code == 422


def test_login_succeeds_and_sets_a_session(client: TestClient):
    email = _email()
    register(client, email)
    client.cookies.clear()
    r = client.post("/auth/login", json={"email": email, "password": PASSWORD})
    assert r.status_code == 200
    assert client.get("/auth/me").json()["authenticated"] is True


def test_bad_password_and_unknown_email_are_indistinguishable(client: TestClient):
    """The generic-failure rule: the response must not disclose whether the email exists."""
    email = _email()
    register(client, email)
    client.cookies.clear()

    wrong_password = client.post("/auth/login", json={"email": email, "password": "wrong-" * 4})
    unknown_email = client.post("/auth/login", json={"email": _email(), "password": PASSWORD})

    assert wrong_password.status_code == unknown_email.status_code == 401
    assert wrong_password.json() == unknown_email.json()
    assert wrong_password.json()["detail"] == auth.BAD_CREDENTIALS


def test_logout_clears_the_session_and_is_idempotent(client: TestClient):
    register(client)
    assert client.get("/auth/me").json()["authenticated"] is True

    assert client.post("/auth/logout").status_code == 200
    assert client.get("/auth/me").json()["authenticated"] is False
    # Logging out twice asks for a state that already holds.
    assert client.post("/auth/logout").status_code == 200


def test_me_is_200_and_anonymous_without_a_session(client: TestClient):
    """Not a 401: a client forced to special-case an error to render the valid
    logged-out state is a client that renders a blank page for logged-out users."""
    r = client.get("/auth/me")
    assert r.status_code == 200
    assert r.json() == {
        "authenticated": False,
        "user": None,
        "personalisation_enabled": False,
        "reason": "no session — the core objective ranking is unaffected",
    }


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def test_session_cookie_is_httponly(client: TestClient):
    r = register(client)
    cookie_header = r.headers["set-cookie"].lower()
    assert "httponly" in cookie_header
    assert "samesite=lax" in cookie_header


def test_raw_token_is_never_stored_only_its_hash(client: TestClient):
    r = register(client)
    token = r.cookies[auth.COOKIE_NAME]
    rows = profiles.conn().execute("select token_hash from sessions").fetchall()
    stored = {row["token_hash"] for row in rows}
    assert token not in stored
    assert auth.token_hash(token) in stored


def test_writes_are_committed_and_survive_the_connection_that_made_them(client: TestClient):
    """REGRESSION. Python's sqlite3 does not autocommit: a bare `execute` opens a
    transaction visible only to its own connection and discarded when the process exits.

    Every in-process assertion still passed, because the reads shared that connection —
    so the whole suite was green while a server restart would have dropped every account,
    session and uploaded decision on the floor. Reading through a SEPARATE connection is
    what makes the difference observable.
    """
    import sqlite3

    from memory import db

    email = _email()
    register(client, email)

    side_channel = sqlite3.connect(db.db_path())
    try:
        rows = side_channel.execute("select email from users where email = ?", (email,)).fetchall()
    finally:
        side_channel.close()
    assert rows == [(email,)], "the account was never committed — it would not survive a restart"


def test_tokens_are_unguessable_and_unique():
    tokens = {auth.new_token() for _ in range(200)}
    assert len(tokens) == 200
    assert all(len(t) >= 32 for t in tokens)


def test_garbage_cookie_is_anonymous_not_an_error(client: TestClient):
    client.cookies.set(auth.COOKIE_NAME, "not-a-real-token")
    r = client.get("/auth/me")
    assert r.status_code == 200
    assert r.json()["authenticated"] is False


def test_expired_session_is_rejected_and_cleaned_up(client: TestClient):
    email = _email()
    register(client, email)
    user = profiles.get_user_by_email(email)

    token = auth.new_token()
    profiles.create_session(user.user_id, auth.token_hash(token), ttl=timedelta(seconds=-1))
    assert profiles.user_for_session(auth.token_hash(token)) is None
    remaining = profiles.conn().execute(
        "select * from sessions where token_hash = ?", (auth.token_hash(token),)
    ).fetchall()
    assert remaining == []


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_login_is_throttled_per_email(client: TestClient):
    email = _email()
    register(client, email)
    client.cookies.clear()

    for _ in range(profiles.MAX_LOGIN_FAILURES):
        assert client.post("/auth/login", json={"email": email, "password": "nope-" * 4}).status_code == 401

    locked = client.post("/auth/login", json={"email": email, "password": PASSWORD})
    assert locked.status_code == 429, "the correct password must not bypass an active lockout"


def test_throttle_is_scoped_to_one_email_not_the_whole_service(client: TestClient):
    """Keyed on email, so one locked-out account cannot lock out the conference room."""
    victim, bystander = _email(), _email()
    register(client, victim)
    register(client, bystander)
    client.cookies.clear()

    for _ in range(profiles.MAX_LOGIN_FAILURES):
        client.post("/auth/login", json={"email": victim, "password": "nope-" * 4})

    assert client.post("/auth/login", json={"email": victim, "password": PASSWORD}).status_code == 429
    assert (
        client.post("/auth/login", json={"email": bystander, "password": PASSWORD}).status_code
        == 200
    )


def test_successful_login_clears_the_failure_count(client: TestClient):
    email = _email()
    register(client, email)
    client.cookies.clear()

    for _ in range(profiles.MAX_LOGIN_FAILURES - 1):
        client.post("/auth/login", json={"email": email, "password": "nope-" * 4})
    assert client.post("/auth/login", json={"email": email, "password": PASSWORD}).status_code == 200

    rows = profiles.conn().execute(
        "select * from login_attempts where email = ?", (email,)
    ).fetchall()
    assert rows == []


# ---------------------------------------------------------------------------
# THE INVARIANT: the core product does not need a session (§1)
# ---------------------------------------------------------------------------


# The objective product surface. `/health` is deliberately absent: it dials GitHub to
# report the live rate limit, and the suite stays offline.
CORE_ROUTES = ["/thesis", "/companies", "/hidden"]


def test_core_routes_serve_without_any_session(client: TestClient):
    for route in CORE_ROUTES:
        r = client.get(route)
        assert r.status_code != 401, f"{route} demands a session; the core product must not"
        assert r.status_code < 500, f"{route} failed unauthenticated: {r.status_code}"


def test_core_routes_still_serve_with_a_broken_session(client: TestClient):
    """A corrupt cookie must make a request anonymous, not fail it."""
    client.cookies.set(auth.COOKIE_NAME, "garbage-token-value")
    for route in CORE_ROUTES:
        assert client.get(route).status_code < 500


def test_core_routes_survive_a_totally_broken_auth_backend(client: TestClient, monkeypatch):
    """The degradation contract: if session lookup raises, the core product still serves
    and personalisation simply switches off. This is the difference between a broken
    login costing the personal layer and a broken login costing the whole demo."""

    def explode(*_args, **_kwargs):
        raise RuntimeError("session store is down")

    monkeypatch.setattr(profiles, "user_for_session", explode)
    client.cookies.set(auth.COOKIE_NAME, "any-token")

    for route in CORE_ROUTES:
        assert client.get(route).status_code < 500, f"{route} died with auth broken"

    me = client.get("/auth/me")
    assert me.status_code == 200
    assert me.json()["authenticated"] is False

    # And the personalisation surface fails closed — 401, not 500.
    assert client.get("/profile").status_code == 401


def test_profile_routes_require_a_session(client: TestClient):
    for method, route in (
        ("get", "/profile"),
        ("put", "/profile"),
        ("post", "/profile/survey"),
        ("post", "/profile/decisions"),
        ("get", "/profile/gap"),
    ):
        r = getattr(client, method)(route, **({"json": {}} if method in ("put", "post") else {}))
        assert r.status_code == 401, f"{method.upper()} {route} should require a session"
