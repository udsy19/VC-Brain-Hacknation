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

from api.main import app  # noqa: E402

__all__ = ["app"]
