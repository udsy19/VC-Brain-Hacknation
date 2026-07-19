"""Vercel serverless entrypoint.

Vercel's Python runtime serves an ASGI app exported as `app`, so this re-exports
the same FastAPI instance the local server runs. There is deliberately no second
application object: a deployment that diverges from what was tested locally is a
demo that fails in a way nobody rehearsed.

Routing (see vercel.json): the frontend is served from `/`, this from `/api/*`.
Because both live on one deployment the browser calls a SAME-ORIGIN relative path,
which removes CORS from the picture entirely rather than widening it.

The repo root must be importable for `from api.main import app` to resolve — on
Vercel the function's working directory is the project root, but sys.path is not
guaranteed to include it, so it is added explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.main import app as _fastapi_app  # noqa: E402


async def _strip_api_prefix(scope, receive, send):
    """Serve the FastAPI app under the /api prefix the rewrite forwards.

    vercel.json rewrites /api/(.*) to this function, and the platform passes the
    ORIGINAL path through — so the app receives "/api/health" while its routes are
    declared as "/health". Locally nothing changes: the dev frontend calls
    http://localhost:8000/health with no prefix, and uvicorn serves api.main:app
    directly, never this wrapper.
    """
    if scope["type"] == "http" and scope.get("path", "").startswith("/api"):
        path = scope["path"][4:] or "/"
        scope = dict(scope, path=path, raw_path=path.encode())
    await _fastapi_app(scope, receive, send)


# Module-level `app` at column 0 — Vercel's builder detects the function by
# scanning for exactly this; hide it and the deploy fails with "api/index.py
# doesn't match any Serverless Functions".
app = _strip_api_prefix

__all__ = ["app"]
