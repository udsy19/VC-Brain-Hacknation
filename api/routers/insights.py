"""Cross-company routes: hidden ranking, NL compound query, backtest. Owner: D."""

from __future__ import annotations

import logging
import re
from datetime import datetime

from fastapi import APIRouter

from api.routers.deps import degrade, pick, resolve_as_of, seed, seed_or

log = logging.getLogger(__name__)
router = APIRouter(tags=["insights"])

FILTER_SYSTEM = (
    "Translate an investor's natural-language query into a JSON filter. "
    "Return ONLY JSON with these keys, using null for anything the query does not "
    "constrain: sectors (list of lowercase strings — ANY industry the investor names, "
    "there is no fixed vocabulary, so 'climate hardware', 'defense', 'consumer social' "
    "and 'logistics' are all valid), stages (list from 'pre-seed','seed','series-a',"
    "'series-b','growth'), geos (list of lowercase places or regions, any granularity), "
    "check_size_min_usd (number or null), check_size_max_usd (number or null), "
    "trend ('rising'|'falling'|null), min_score (float 0-1 or null), "
    "verification ('unverified'|'verified'|null — use 'unverified' for any query about "
    "contradicted, unverifiable or unattempted claims), flagged (true|false|null), "
    "gate ('proceed'|'proof_protocol'|'no_call'|null), limit (int or null), and "
    "unparsed (list of the individual clauses you could NOT express in the keys above "
    "— never leave a constraint out silently, put it here instead; put ONLY the "
    "specific words you could not use, never the whole query, and leave it null when "
    "you expressed everything). Anything that describes WHAT A COMPANY DOES or what "
    "kind of company it is — an industry, a product area, a descriptor like 'cold "
    "start' or 'pre-product' — belongs in sectors, not in unparsed. "
    "Judge on substance only — never emit a filter on schooling, employer brand or "
    "investor name."
)

# --- deterministic fallback -------------------------------------------------
# There is deliberately NO sector vocabulary here. The old `_SECTOR_WORDS` list
# capped what an investor could express to eight words, and — worse — a query in an
# unlisted industry parsed to `sectors: None`, which matches EVERY company while the
# readback claimed "no filters recognised". Silently returning the whole list is the
# one answer a filter must never give. Instead: recognise the structural clauses
# (stage, geo, money, trend, gate, verification, limit), and whatever survives that
# pass is read as an open-vocabulary SECTOR PHRASE and reported as such. An industry
# this system has never seen therefore returns zero rows and says why.

_STAGE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("pre-seed", r"pre[\s-]?seed"),
    ("series-a", r"series[\s-]?a\b"),
    ("series-b", r"series[\s-]?b\b"),
    ("growth", r"\bgrowth[\s-]?stage\b"),
    ("seed", r"\bseed\b"),
)

# Fallback-only gazetteer. The LLM path is open-vocabulary for geography; without a
# model we only claim a geo when we are sure, because a WRONG parse ("companies in
# AI" read as geography = ai) is worse than an unparsed one. Anything not listed
# falls through to the sector phrase, where it is named in the readback.
_GEO_WORDS: tuple[str, ...] = (
    "north america", "south america", "latin america", "latam", "europe", "emea",
    "apac", "asia", "southeast asia", "sea", "middle east", "africa", "oceania",
    "united states", "usa", "us", "canada", "mexico", "brazil", "uk",
    "united kingdom", "britain", "ireland", "france", "germany", "spain", "portugal",
    "italy", "netherlands", "sweden", "norway", "denmark", "finland", "poland",
    "estonia", "switzerland", "israel", "india", "china", "japan", "korea",
    "singapore", "indonesia", "vietnam", "australia", "new zealand", "nigeria",
    "kenya", "egypt", "south africa", "global", "remote",
)

# Filler an investor types around the substance. Stripped before the leftover is read
# as a sector phrase, so "seed-stage fintech companies in Europe" leaves "fintech".
_STOPWORDS = frozenset(
    """a an and or the of for in into on at to with without that which who whose
    show me find list all any some companies company startups startup founders founder
    teams team businesses raising raise round rounds stage staged based headquartered
    hq located doing building build built working work works focused focus focusing
    space sector sectors industry industries market markets
    please still only just more most less than under over above below between up
    ideally preferably new early late top first thats theyre is are was were be been
    has have had do does did not no yes but so if then when where what how""".split()
)

_MONEY_SCALES = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}


@router.get("/hidden")
def get_hidden(as_of: datetime | None = None, k: int = 50) -> dict:
    """High proximity to greatness, low individual visibility — plus the access lift."""
    cutoff = resolve_as_of(as_of)

    def live() -> dict:
        from sourcing import graph

        ranked = graph.hidden_ranking(cutoff, k)
        if not ranked:
            # An empty live ranking must not silently beat a seeded one on stage.
            raise LookupError("hidden ranking returned no candidates")
        picks = [c.entity_id for c in ranked]
        candidates = [
            {
                "entity_id": str(c.entity_id),
                "ppr": c.ppr,
                "visibility": c.visibility,
                "hidden_score": c.hidden_score,
            }
            for c in ranked
        ]
        try:
            lift = graph.access_lift(picks)
        except NotImplementedError:
            lift = None
        return {
            "as_of": cutoff.isoformat(),
            "candidates": candidates,
            "access_lift": lift,
            "degraded": False,
        }

    return degrade(live, lambda: {**seed("hidden"), "degraded": True})


FILTER_KEYS = (
    "sectors", "stages", "geos", "check_size_min_usd", "check_size_max_usd",
    "trend", "min_score", "verification", "flagged", "gate", "limit", "unparsed",
)


def _norm_text(s: object) -> str:
    """Lowercase, punctuation-to-space. `ai-infra` and `AI infra` must compare equal."""
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def _blank_filter() -> dict:
    return dict.fromkeys(FILTER_KEYS)


def _coerce_filter(raw: dict) -> dict:
    """Normalise whatever came back into exactly FILTER_KEYS.

    A model that invents a key must not have it silently dropped: unknown keys are
    surfaced through `unparsed` so the readback can admit the query said something
    this filter cannot execute.
    """
    f = _blank_filter()
    extra: list[str] = []
    for k, v in (raw or {}).items():
        if k in FILTER_KEYS:
            f[k] = v
        elif v not in (None, [], "", False):
            extra.append(f"{k}={v!r}")

    for k in ("sectors", "stages", "geos", "unparsed"):
        v = f[k]
        if isinstance(v, str):
            v = [v]
        f[k] = [str(x).strip().lower() for x in v if str(x).strip()] if isinstance(v, list) else None
        if not f[k]:
            f[k] = None

    f["stages"] = [_norm_text(s).replace(" ", "-") for s in f["stages"]] if f["stages"] else None
    if extra:
        f["unparsed"] = (f["unparsed"] or []) + extra
    return f


def _parse_money(text: str) -> tuple[float | None, float | None]:
    """(min, max) in USD from phrases like `cheque under $2M` or `$250k-$1.5m`."""
    tokens = [
        (m.start(), float(m.group(1)) * _MONEY_SCALES.get((m.group(2) or "").lower(), 1))
        for m in re.finditer(r"\$\s*(\d+(?:\.\d+)?)\s*([kmb])?\b", text)
    ]
    if not tokens:
        return None, None
    if len(tokens) >= 2 and re.search(r"\bbetween\b|\bfrom\b", text[: tokens[0][0]]):
        lo, hi = sorted(v for _, v in tokens[:2])
        return lo, hi

    pos, value = tokens[0]
    lead = text[max(0, pos - 40) : pos]
    if re.search(r"\b(under|below|less than|up to|max|at most|no more than|sub)\b\s*$", lead):
        return None, value
    if re.search(r"\b(over|above|more than|at least|min|from|north of)\b\s*$", lead):
        return value, None
    # A bare amount is the cheque the investor writes — treat it as the ceiling, and
    # say so in the readback rather than guessing silently.
    return None, value


def _keyword_filter(q: str) -> dict:
    """Deterministic parse. Open sector vocabulary by construction: everything the
    structural passes do not claim is read as the sector phrase and reported."""
    text = _norm_text(q)
    f = _blank_filter()

    def take(pattern: str) -> bool:
        """Consume a span so it cannot also be read as part of the sector phrase."""
        nonlocal text
        new, n = re.subn(pattern, " ", text)
        text = new
        return bool(n)

    # Numeric clauses first, most specific to least — otherwise the money pass eats
    # the digits out of "top 5" and the limit is silently lost.
    m = re.search(r"\b(?:top|first|best)\s+(\d+)\b", text)
    if m:
        f["limit"] = int(m.group(1))
        take(r"\b(top|first|best)\s+\d+\b")

    # Read the score off the RAW query: normalisation turns "0.7" into "0 7", and
    # reading that back gave min_score 0.0 — a clause that silently did nothing.
    m = re.search(r"\b(?:score|scoring)\D{0,12}?(\d+(?:\.\d+)?)", q.lower())
    if m:
        v = float(m.group(1))
        f["min_score"] = v / 100 if v > 1 else v
        take(r"\b(score|scoring)\b\D{0,12}?\d+(\s+\d+)?\b")

    f["check_size_min_usd"], f["check_size_max_usd"] = _parse_money(q.lower())
    if f["check_size_min_usd"] or f["check_size_max_usd"]:
        take(r"\b(cheque|check|ticket)\s*(size)?\b")
        take(r"\b(under|below|less than|up to|at most|over|above|more than|at least|between)\b")
        take(r"\b\d+(\s+\d+)*\s*[kmb]?\b")

    stages = [name for name, pat in _STAGE_PATTERNS if take(pat)]
    f["stages"] = stages or None
    take(r"\bstage\b")

    geos = [g for g in sorted(_GEO_WORDS, key=len, reverse=True) if take(rf"\b{re.escape(g)}\b")]
    f["geos"] = geos or None

    if take(r"\brising\b|\bmomentum\b|\bimproving\b"):
        f["trend"] = "rising"
    elif take(r"\bfalling\b|\bdeclining\b|\bdeteriorating\b"):
        f["trend"] = "falling"
    take(r"\btrend\b")

    if take(r"\bunverified\b|\bunverifiable\b|\bcontradicted\b|\bunproven\b"):
        f["verification"] = "unverified"
    elif take(r"\bverified\b"):
        f["verification"] = "verified"
    take(r"\bclaims?\b")

    if f["verification"]:
        # "unverified REVENUE" names the SUBJECT of the claim. The ranked row carries a
        # claim count, not claim text, so this clause cannot be executed here. It is
        # recorded as unparsed rather than left in the leftover, where it would have
        # become a phantom sector filter ("sector ~ 'revenue'") and silently returned
        # nothing. Under-filtering and saying so beats a wrong filter said confidently.
        subjects = [
            w
            for w in ("revenue", "arr", "traction", "pilot", "customer", "partnership", "growth")
            if take(rf"\b{w}s?\b")
        ]
        if subjects:
            f["unparsed"] = [
                f"claim subject {', '.join(subjects)} — the ranked list stores a claim "
                f"count, not claim text, so verification was applied to ALL claims, "
                f"not only those about {subjects[0]}"
            ]

    if take(r"\bproof protocol\b|\bproof_protocol\b"):
        f["gate"] = "proof_protocol"
    elif take(r"\bno call\b|\bno_call\b|\brejected\b"):
        f["gate"] = "no_call"
    elif take(r"\bproceed\b"):
        f["gate"] = "proceed"
    take(r"\brouted to\b|\bgate\b")

    if take(r"\bintegrity\b|\bflags?\b|\bflagged\b|\binjection\b|\badversarial\b"):
        f["flagged"] = True

    leftover = [w for w in text.split() if w not in _STOPWORDS and len(w) > 1]
    f["sectors"] = [" ".join(leftover)] if leftover else None
    return f


def _parse_filter(q: str) -> dict:
    """LLM first; deterministic pass if it's unavailable. Both return FILTER_KEYS."""
    try:
        from core import llm

        out = llm.complete(
            f"Query: {q}", system=FILTER_SYSTEM, tier="fast", untrusted=q, json_mode=True
        )
        if isinstance(out, dict):
            return _coerce_filter(out)
    except Exception as exc:  # noqa: BLE001 - query must still work without a model
        log.info("query: model unavailable, using keyword filter (%s)", exc)

    f = _coerce_filter(_keyword_filter(q))
    f["degraded"] = True
    return f


def _trend_value(c: dict) -> float | None:
    # The founder axis is where trend actually lives now. Checked first: a row can
    # still carry a flat top-level `momentum`, and reading that instead made every
    # "rising trend" query return nothing while six companies were plainly rising.
    axis = (c.get("axes") or {}).get("founder") or {}
    if isinstance(axis.get("trend"), (int, float)) and not isinstance(axis.get("trend"), bool):
        return float(axis["trend"])

    t = pick(c, "trend", "trend_value", "momentum")
    if isinstance(t, (int, float)) and not isinstance(t, bool):
        return float(t)
    if isinstance(t, str):
        return {"rising": 1.0, "up": 1.0, "flat": 0.0, "falling": -1.0, "down": -1.0}.get(t.lower())
    return None


def _unverified(c: dict) -> bool:
    n = pick(c, "unverified_claims", "gap_count")
    if isinstance(n, (int, float)):
        return n > 0
    claims = pick(c, "claims", "claim_verdicts", default=[])
    statuses = {str(pick(x, "status", default="")).lower() for x in claims if isinstance(x, dict)}
    return bool(statuses & {"unverifiable", "not_attempted", "contradicted"})


def _haystack(c: dict) -> str:
    """Fixture key names are still settling — flatten every sector-ish field into one blob.

    Normalised, so a needle of "ai infra" matches a stored sector of "ai-infra".
    `one_liner` and `archetype_label` are in here because with an open sector
    vocabulary the phrase an investor types ("record and replay", "cold start") is
    frequently the product, not the taxonomy label.
    """
    parts: list[str] = []
    for key in (*_TAXONOMY_KEYS, "description", "name", "one_liner", "standout"):
        v = c.get(key)
        if isinstance(v, list):
            parts.extend(str(x) for x in v)
        elif v is not None:
            parts.append(str(v))
    return _norm_text(" ".join(parts))


# Where each structural field actually lives on a row. `None` from _field means the
# record does not carry the field at all — which is NOT the same as not matching.
_FIELD_KEYS: dict[str, tuple[str, ...]] = {
    "stages": ("stage", "round", "round_stage", "funding_stage"),
    "geos": ("geo", "region", "country", "location", "hq", "headquarters"),
}
_CHECK_KEYS = ("check_size", "round_size", "raising", "raise_usd", "ask_usd", "target_raise")


def _field(c: dict, keys: tuple[str, ...]) -> str | None:
    for k in keys:
        v = c.get(k)
        if isinstance(v, list) and v:
            return _norm_text(" ".join(str(x) for x in v))
        if v not in (None, ""):
            return _norm_text(v)
    return None


def _check_size(c: dict) -> float | None:
    for k in _CHECK_KEYS:
        v = c.get(k)
        if isinstance(v, dict):
            v = v.get("target", v.get("max", v.get("min")))
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


# Both the display label and the slug: the live route serves sector="Developer
# Infrastructure" alongside sector_key="dev-tools", and an investor typing "dev tools"
# matches only the second. Reading one and not the other returned zero rows for a
# sector the fund is actively invested in.
_TAXONOMY_KEYS = (
    "sector", "sector_key", "sectors", "category", "tags", "archetype", "archetype_label",
)


def _sector_hit(c: dict, sectors: list) -> bool:
    """Open-vocabulary sector match, with the two halves of the row held to different bars.

    A single token is enough against the TAXONOMY fields (sector, tags, archetype), so
    "data tooling" finds `data-infra` and "cold start" finds the Cold Start archetype.
    Against free PROSE (one-liner, description) the whole phrase must appear, because
    a single prose token is not a sector claim: "consumer social" would otherwise match
    a GPU company whose one-liner happens to contain the word "consumer".

    Either way there is no fixed vocabulary, so "climate hardware" simply finds nothing —
    which is the correct answer for a portfolio containing no climate hardware, and the
    whole reason the hardcoded sector list had to go.
    """
    taxonomy = f" {_norm_text(' '.join(str(c.get(k)) for k in _TAXONOMY_KEYS if c.get(k)))} "
    prose = f" {_haystack(c)} "
    for s in sectors:
        phrase = _norm_text(s)
        if not phrase:
            continue
        if f" {phrase} " in prose:
            return True
        for tok in phrase.split():
            if len(tok) > 1 and tok not in _STOPWORDS and f" {tok} " in taxonomy:
                return True
    return False


def _matches(c: dict, f: dict) -> bool:
    """True if the row survives every clause that is EVALUABLE against it.

    A clause the row carries no data for does not exclude it. That is the same rule
    `core/thesis.in_scope` applies — absent metadata is not disqualifying — and the
    number of rows a clause could not be tested on is reported in the readback rather
    than being quietly folded into the result.
    """
    sectors = f.get("sectors")
    if sectors and not _sector_hit(c, sectors):
        return False

    for key in ("stages", "geos"):
        wanted = f.get(key)
        have = _field(c, _FIELD_KEYS[key])
        if wanted and have is not None:
            if not any(_norm_text(w) and _norm_text(w) in have for w in wanted):
                return False

    lo, hi = f.get("check_size_min_usd"), f.get("check_size_max_usd")
    if isinstance(lo, (int, float)) or isinstance(hi, (int, float)):
        cs = _check_size(c)
        if cs is not None:
            if isinstance(lo, (int, float)) and cs < lo:
                return False
            if isinstance(hi, (int, float)) and cs > hi:
                return False

    if f.get("flagged") is not None:
        flags = c.get("flags")
        n = len(flags) if isinstance(flags, list) else pick(c, "flag_count", default=0)
        if bool(isinstance(n, (int, float)) and n > 0) is not bool(f["flagged"]):
            return False

    trend = f.get("trend")
    if trend in ("rising", "falling"):
        tv = _trend_value(c)
        if tv is None or (tv <= 0 if trend == "rising" else tv >= 0):
            return False

    min_score = f.get("min_score")
    if isinstance(min_score, (int, float)):
        # The founder axis is where the score lives on a ranked row. Reading only the
        # flat keys made every "founder score above X" query return nothing at all.
        mu = (c.get("axes") or {}).get("founder", {}).get("score")
        if mu is None:
            mu = pick(c, "mu", "score", "founder_score")
        mu = pick(mu, "mu", default=mu) if isinstance(mu, dict) else mu
        if not isinstance(mu, (int, float)) or mu < min_score:
            return False

    verification = f.get("verification")
    if verification == "unverified" and not _unverified(c):
        return False
    if verification == "verified" and _unverified(c):
        return False

    gate = f.get("gate")
    if (
        gate
        and str(pick(c, "gate", "gate_outcome", "outcome", default="")).lower() != str(gate).lower()
    ):
        return False
    return True


@router.get("/query")
def compound_query(q: str, as_of: datetime | None = None) -> dict:
    """NL compound query over the ranked list. The model only translates — the filter
    itself runs in plain Python, so the result set is inspectable and reproducible."""
    from api.main import list_companies

    # Filter the LIVE ranked list, not the seed file. Querying a different set than
    # the one on screen returns ids the page cannot resolve, and the client is left
    # waiting on a result it can never render.
    try:
        companies = list_companies(as_of)
    except Exception:  # noqa: BLE001
        companies = seed_or("companies", {}).get("companies", [])

    f = _parse_filter(q)
    results = [c for c in companies if _matches(c, f)]
    limit = f.get("limit")
    if isinstance(limit, int) and limit > 0:
        results = results[:limit]

    warnings = _filter_warnings(f, companies)

    # `company_ids` + `parsed` is the client's contract (lib/types.ts QueryResult).
    # Returning a different shape makes the client silently fall back to a local
    # interpreter that only knows fixture ids — which is what hung the button.
    return {
        "q": q,
        "parsed": _describe_filter(f),
        "warnings": warnings,
        "company_ids": [c.get("id") for c in results if c.get("id")],
        "count": len(results),
        "filter": f,
        "results": results,
        # This route NARROWS the view. It never edits the fund's standing thesis —
        # promoting a query to the thesis is a separate, explicit PUT /thesis.
        "scope": "view_filter",
    }


def _fmt_usd(v: float) -> str:
    for suffix, unit in (("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
        if abs(v) >= unit:
            return f"${v / unit:.2f}".rstrip("0").rstrip(".") + suffix
    return f"${v:,.0f}"


def _describe_filter(f: dict) -> str:
    """Plain-English readback of what the query was understood to mean.

    Every executed clause appears here, and anything the parse could not express
    appears as an explicit "not understood" clause. A natural-language box that drops
    half a sentence without saying so is worse than a form, because the user reads a
    result set as an answer to the question they asked.
    """
    parts = []
    if f.get("sectors"):
        parts.append(f"sector ~ {', '.join(repr(s) for s in f['sectors'])}")
    if f.get("stages"):
        parts.append(f"stage in {', '.join(f['stages'])}")
    if f.get("geos"):
        parts.append(f"geo in {', '.join(f['geos'])}")
    lo, hi = f.get("check_size_min_usd"), f.get("check_size_max_usd")
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
        parts.append(f"cheque {_fmt_usd(lo)}–{_fmt_usd(hi)}")
    elif isinstance(lo, (int, float)):
        parts.append(f"cheque >= {_fmt_usd(lo)}")
    elif isinstance(hi, (int, float)):
        parts.append(f"cheque <= {_fmt_usd(hi)}")
    if f.get("trend"):
        parts.append(f"{f['trend']} trend")
    if f.get("min_score") is not None:
        parts.append(f"founder score >= {f['min_score']}")
    if f.get("verification"):
        parts.append(f"{f['verification']} claims")
    if f.get("flagged") is not None:
        parts.append("has integrity flags" if f["flagged"] else "no integrity flags")
    if f.get("gate"):
        parts.append(f"gate = {f['gate']}")
    if f.get("limit"):
        parts.append(f"top {f['limit']}")
    if f.get("unparsed"):
        parts.append("NOT UNDERSTOOD: " + "; ".join(str(x) for x in f["unparsed"]))

    if not parts:
        # Never "showing everything" without saying that no constraint was applied.
        return "nothing in this query could be turned into a filter — every record is shown, unfiltered"
    return " · ".join(parts)


def _filter_warnings(f: dict, companies: list) -> list[str]:
    """Where the answer is weaker than it looks. Rendered next to the readback.

    Two honest failures live here: a clause naming something no screened record
    mentions (so the empty result is about our coverage, not the query), and a clause
    the records carry no data for at all (so it silently did not narrow anything).
    """
    out: list[str] = []
    total = len(companies)

    for s in f.get("sectors") or []:
        if not any(_sector_hit(c, [s]) for c in companies):
            out.append(
                f"no screened record mentions {s!r} — this system has never sourced that "
                f"industry, so the zero result is about our coverage, not about the query"
            )

    for key, label in (("stages", "stage"), ("geos", "geography")):
        if not f.get(key):
            continue
        missing = sum(1 for c in companies if _field(c, _FIELD_KEYS[key]) is None)
        if missing:
            out.append(
                f"{label} was not applied to {missing} of {total} records — they carry no "
                f"{label} field, and absent metadata is not treated as disqualifying"
            )

    if f.get("check_size_min_usd") or f.get("check_size_max_usd"):
        missing = sum(1 for c in companies if _check_size(c) is None)
        if missing:
            out.append(
                f"cheque size was not applied to {missing} of {total} records — no round "
                f"size is recorded for them. Cheque size is a property of your fund; set "
                f"it on the standing thesis where it governs the recommendation"
            )

    if f.get("degraded"):
        out.append(
            "parsed without a language model (deterministic fallback) — structural "
            "clauses are exact, and everything else was read as a sector phrase"
        )
    if f.get("unparsed"):
        out.append(
            "part of this query was not turned into a filter; the rows below were NOT "
            "narrowed by it: " + "; ".join(str(x) for x in f["unparsed"])
        )
    return out


@router.get("/backtest")
def get_backtest() -> dict:
    """Winners rising vs controls flat, threshold line, and one correctly-deprioritized
    failure. fame_check_passed is the H12 gate, surfaced so it cannot be missed."""
    from backtest import runner

    def live() -> dict:
        return _backtest_view(runner.run_calibration())

    return degrade(live, lambda: seed("backtest"))


# Scores are 0..1 internally and 0..100 in the client, same as the axes.
def _pct(v: float | None) -> float | None:
    return None if v is None else round(float(v) * 100, 1)


def _backtest_view(cal: dict) -> dict:
    """Map the calibration onto the client's Backtest contract (app/lib/types.ts).

    Without this the page validated on a `trajectories` array that nothing produced,
    failed silently, and rendered its hardcoded fixture — so the one artifact whose job
    is proving the system is not fooling itself was showing pre-computed numbers.
    """
    threshold = _pct(cal.get("threshold"))
    winners = cal.get("winners") or []
    controls = cal.get("controls") or []
    failure = cal.get("correctly_deprioritized_failure") or {}

    trajectories = [
        {
            "id": m.get("id") or f"{m.get('label')}-{i}",
            "name": m.get("name") or m.get("founder") or "unnamed",
            "label": m.get("label"),
            "outcome": m.get("outcome") or "",
            "points": [
                {
                    "t": p.get("as_of"),
                    "mu": _pct(p.get("mu", p.get("founder_score"))),
                    "band": _pct(p.get("band")),
                }
                for p in (m.get("trajectory") or [])
            ],
        }
        for i, m in enumerate(winners + controls)
    ]

    peak_control = max((m.get("peak_mu") or 0 for m in controls), default=None)
    # controls_clearing_threshold is a LIST of the controls that cleared. Formatting it
    # directly rendered "H12 gate: [] of 4 controls cleared" on screen — the gate's own
    # headline sentence, printed as a python repr.
    cleared = cal.get("controls_clearing_threshold")
    n_cleared = len(cleared) if isinstance(cleared, (list, tuple, set)) else (cleared or 0)

    # The controls are SYNTHETIC — invented composites, not real non-breakout
    # companies — and the same author wrote both sides of the comparison. The
    # separation is real in that the scorer produced it unaided, but a synthetic
    # control is a much weaker test than a real one, and the sentence has to say so
    # rather than implying these are real contemporaries.
    detail = (
        f"H12 gate: {n_cleared} of {len(controls)} controls "
        f"cleared the {threshold} threshold"
        + (f" (highest control: {_pct(peak_control)})." if peak_control is not None else ".")
        + " Controls are synthetic composites built to be visible but not shipping,"
        " replayed through the live scorer. If they had cleared, the score would be"
        " measuring visibility rather than capability."
    )

    return {
        "as_of": max((p["t"] for t in trajectories for p in t["points"] if p.get("t")), default=""),
        "truncation_note": cal.get("truncation_note")
        or "Sources truncated by hand to a pre-breakout date, recorded per cohort member.",
        "threshold": threshold,
        "fame_check_passed": bool(cal.get("fame_check_passed")),
        "fame_check_evaluated": bool(cal.get("fame_check_evaluated")),
        "fame_check_detail": detail,
        "hit_rate": cal.get("hit_rate"),
        "n_winners": len(winners),
        "n_controls": len(controls),
        "trajectories": trajectories,
        "correctly_deprioritized": {
            "name": failure.get("name") or failure.get("founder") or "",
            "final_score": _pct(failure.get("peak_mu")),
            "why": failure.get("why") or "",
            "outcome": failure.get("outcome") or "",
        },
        "lookahead_assertion": {
            "events_checked": cal.get("events_checked", 0),
            "violations": 0 if cal.get("lookahead_checked") else None,
            "detail": "Every scoring step asserts no event with observed_at > as_of "
            "reached the scorer. The assertion raises; it is never downgraded to a warning.",
        },
        "degraded": False,
    }
