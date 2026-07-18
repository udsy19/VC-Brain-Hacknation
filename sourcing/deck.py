"""Deck OCR. Owner: B. Keep slide IDs on every span — the memo cites 'slide 7'.

Low-confidence extraction sets integrity_flags=['ocr_low_conf'] so D can SURFACE it.
A bad scan that just quietly scores low is the Type 6 failure mode.
"""

from __future__ import annotations

from pathlib import Path

from schema.events import Event


def extract(pdf_path: Path, company_id) -> list[Event]:
    raise NotImplementedError("B: H3-8")
