"""FastAPI app. Owner: D. Thin — it calls into memory/sourcing/intelligence, nothing more.

Every route returns fixtures until the real module lands. D never blocks on anyone:
mock against the SHARED.md §4 signature, swap when their PR merges.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="VC Brain", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SEED_DIR = Path("data/seed")


def _seed(name: str) -> dict:
    path = SEED_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, f"no seed fixture: {name}")
    return json.loads(path.read_text())


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/thesis")
def get_thesis() -> dict:
    """Config, not code: sectors, stage, geo, check size, risk appetite."""
    return _seed("thesis")


@app.get("/companies")
def list_companies(as_of: datetime | None = None) -> list[dict]:
    """Ranked list + momentum. Ranked by an explicit policy — never by a mean of the axes."""
    return _seed("companies")["companies"]


@app.get("/companies/{company_id}")
def get_company(company_id: str, as_of: datetime | None = None) -> dict:
    return _seed(f"company_{company_id}")


@app.get("/companies/{company_id}/trace/{event_id}")
def get_trace(company_id: str, event_id: str) -> dict:
    """Score -> contributing events -> source span -> original URL/slide ID.

    Judges will click this. It must bottom out in a quoted span, not a source name.
    """
    raise HTTPException(501, "D: H3-8")


@app.get("/companies/{company_id}/memo")
def get_memo(company_id: str, dissent_viewed: bool = False) -> dict:
    """Recommendation stays null until dissent is opened. Enforced HERE, not in the UI."""
    memo = _seed(f"memo_{company_id}")
    if not dissent_viewed:
        memo["recommendation"] = None
        memo["recommendation_locked_reason"] = "open the dissent view first"
    return memo


@app.get("/companies/{company_id}/dissent")
def get_dissent(company_id: str) -> dict:
    return _seed(f"dissent_{company_id}")


@app.get("/backtest")
def get_backtest() -> dict:
    """Winners rising vs controls flat, threshold line, and one correctly-deprioritized failure."""
    return _seed("backtest")
