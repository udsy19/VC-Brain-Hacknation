"""Warm the computed caches before a demo. Run it after starting the API.

Two things are computed on demand rather than on the list path, both deliberately:

  THREE-AXIS SCREENING   ~7s per company. On the list that is ~95s, so the list
                         serves the seeded market/idea axes with `live: false` and
                         computes them when a detail page is opened.
  STANDOUT SUMMARIES     a corpus comparison plus an LLM call. Cold, the list
                         honestly reports `not_generated` rather than blocking.

Both are correct behaviour — but on a cold process the ranked list shows seeded
axes and no summaries, which is not what you want to open a demo on. This walks
every company once and leaves the caches warm.

    uv run python scripts/warm.py                  # against localhost:8000
    uv run python scripts/warm.py --api URL

KNOWN LIMITATION, stated rather than hidden: the standout summary cache is keyed by
a corpus digest that does not survive a process restart, so RESTARTING THE API GOES
COLD AGAIN even though the summaries are still on disk. The frame reloads; the
per-summary key does not match. Warm after the last restart before you present, not
before it.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request


def get(api: str, path: str, timeout: float = 180.0):
    with urllib.request.urlopen(f"{api}{path}", timeout=timeout) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    args = ap.parse_args()
    api = args.api.rstrip("/")

    try:
        rows = get(api, "/companies", 180)
    except urllib.error.URLError as exc:
        print(f"api unreachable at {api}: {exc}")
        return 1

    ids = [r["id"] for r in rows if r.get("id")]
    print(f"warming {len(ids)} companies via {api}\n")

    start = time.time()
    failures = 0
    for i, cid in enumerate(ids, 1):
        try:
            get(api, f"/companies/{cid}", 180)  # computes the three axes
            get(api, f"/companies/{cid}/standout", 180)  # computes + stores the summary
            print(f"  {i:2}/{len(ids)}  {cid}")
        except Exception as exc:  # noqa: BLE001 - one failure must not stop the warm
            failures += 1
            print(f"  {i:2}/{len(ids)}  {cid}  FAILED {type(exc).__name__}")

    rows = get(api, "/companies", 180)
    live = sum(
        1
        for r in rows
        for a in (r.get("axes") or {}).values()
        if isinstance(a, dict) and a.get("live")
    )
    summarised = sum(
        1
        for r in rows
        if isinstance(r.get("standout"), dict) and r["standout"].get("status") != "not_generated"
    )
    print(
        f"\n  {time.time() - start:.0f}s · {failures} failed · "
        f"{live} live axes · {summarised}/{len(rows)} rows carry a summary"
    )
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
