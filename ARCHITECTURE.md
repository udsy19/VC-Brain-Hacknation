# VC Brain — how the system works

A description of the whole system: what it does, how the pieces fit, what runs in what
order, and where the seams currently are.

README.md is the quick start. SHARED.md is the contract between branches. This file is
the explanation — read it when you need to know *why* something is shaped the way it is,
or when you're picking up a part of the codebase you didn't write.

---

## 1. What this is

A venture-scouting system for one vertical — **AI-infra / dev-tools, pre-seed and seed** —
built around a single claim:

> The reason good founders get missed is that scouting measures fame. Fame is a property
> of the observer, not of the founder.

Every major component is an attempt to remove one specific way that bias gets in.

| Capability | Mechanism |
|---|---|
| Find founders others can't see | Graph diffusion — personalized PageRank minus visibility |
| Test founders with nothing to show | Proof Protocol — a generated challenge with two deliberate traps |
| Score without pedigree | Kalman filter over evidence; banned-term list enforced in CI |
| Argue before spending money | Dissent Engine + a three-role Council with a receipt-gated chair |
| Prove it against history | Time-machine backtest with a fame-vs-trajectory control check |

Ownership is by directory so PRs don't collide: **A** owns `schema/` + `memory/` + `core/`,
**B** owns `sourcing/`, **C** owns `intelligence/`, **D** owns `api/` + `app/` + `backtest/`
+ `data/seed/`.

---

## 2. The four invariants

These are the spine. Each is enforced structurally in several independent places, because
a convention that only lives in a docstring does not survive hour 19 of a hackathon.

### Invariant 1 — No lookahead

Scoring at `as_of` sees only events with `observed_at <= as_of`.

- `store.events()` makes `as_of` a **required keyword-only parameter with no default**.
  The docstring states the intent plainly: it "makes the lookahead bug hard to write
  rather than merely discouraged."
- The schema rejects naive datetimes at the boundary — a naive datetime silently breaks
  `as_of` comparisons.
- Timestamps are stored with fixed-width microseconds, so lexical string ordering equals
  chronological ordering and the `observed_at <= ?` comparison is correct in SQL.
- `observed_at` (when the world produced the signal) is structurally separate from
  `ingested_at` (when we saw it). `ingested_at` is never read by scoring.
- `backtest.assert_no_lookahead()` raises on any leak, and is the one exception that
  stage-level error handling deliberately re-raises rather than swallowing.

### Invariant 2 — Trust is per-claim, never one company number

`ClaimVerdict.trust` exists. No company-level trust field exists anywhere in the schema.
A contradicted revenue number kills the revenue claim and widens uncertainty; it does not
zero the founder. *Contradiction reprices the claim, not the deal.*

### Invariant 3 — No pedigree anywhere in scoring

`intelligence/banned.py` holds 54 terms in four groups:

- **Schools and credentials** — university names, `ivy league`, `gpa`, `pedigree`
- **Employer brand** — `ex-google`, `ex-openai`, `faang`, `big tech`, `blue-chip`
- **Investor halo** — accelerator and fund names, `backed by`
- **Network proxies** — `well-connected`, `warm intro`, `serial entrepreneur`

`tests/test_no_pedigree.py` greps all source and all prompts against this list. It runs in
CI and is a hard fail. The ban is reinforced at the prompt level too — the market-axis
prompt says "no assumptions about the people involved," and the Council's bias auditor is
explicitly tasked with catching reputation-proxy signals.

### Invariant 4 — Founder text is data, never instructions

The `<untrusted_content>` wrapper is applied **inside** `llm.complete()`, not by callers.
`core/llm.py` is the only file in the repo permitted to import a vendor SDK, which makes
it the single choke point where wrapping happens — so it cannot be forgotten under time
pressure. Callers pass `untrusted=` rather than concatenating into the prompt.

---

## 3. The pipeline, end to end

```
SCANNERS            github / hn / arxiv / tavily-web / deck-pdf
   |                live APIs, disk-cached under data/raw/ so replays run offline
   v
INGESTION BUS       sanitize -> stamp observed_at -> normalize -> Event
   |                sourcing/bus.py is the one funnel; deck.py routes through it too
   v
EVENT LOG           append-only Postgres (Supabase) or SQLite, uuid5-idempotent
   |                DB triggers physically reject UPDATE and DELETE
   v
ENTITY RESOLUTION   four weighted signals -> MERGED / NEW / AMBIGUOUS
   |
   v
DERIVATION          core/pipeline.py — green flags at monthly checkpoints, claim validation
   |
   v
SCORING             Kalman filter -> mu / band / trend
   |
   v
SCREENING           three axes, never averaged
   |
   v
GATE                PROCEED / PROOF_PROTOCOL / NO_CALL
   |
   v
PROOF PROTOCOL      challenge -> artifact + behavioral trace -> graded events -> back into the log
   |
   v
COUNCIL             scout vs skeptic vs bias-auditor -> chair, with policy overrides
   |
   v
MEMO + DISSENT      every sentence carries event-id receipts
```

The important structural property: **there is no separate backtest mode.** `backtest/runner.py`
calls the same `store.events` / `score.founder` / `screen.three_axis` / `gate.evaluate` /
`generate_memo` that the API calls. There is no `backtest=True` flag anywhere. The replay
is the product, run with the clock moved back.

---

## 4. Sourcing

### Scanners

Four live scanners, each disk-cached so a replay runs offline. None of them write events —
each returns `RawSignal`s for the bus to convert.

- **github** — profile, repo activity, commits, releases. Uses commit *author* date, not
  committer date. Fan-out is capped because unauthenticated rate limits are 60/hour.
- **hn** — Algolia search over stories and comments.
- **arxiv** — restricted to `cs.LG`, `cs.DC`, `cs.PL`. Uses `<published>` (the v1
  submission date), never `<updated>`, so a v3 revision cannot backdate credit. Entries
  with no honest timestamp are **dropped entirely**.
- **web** — Tavily enrichment. Runs the bare query first, deliberately, so English
  qualifiers don't bury a non-English footprint.

### The `observed_at` ladder

Three rungs, in `bus._stamp()`:

1. A real timestamp in the signal → use it, **no flag**.
2. An inferred date floor (from a URL path or body text) → use it, flag `date_inferred`.
3. Fetch time → use it, flag `date_inferred`.

Fetch time never grants retroactive credit in the backtest. A Tavily result with no publish
date gets flagged rather than silently stamped `now()` — that would quietly poison the
backtest and stay invisible until it mattered.

### The injection guard

`sourcing/sanitize.py`. The doctrine is **strip, don't reject** — the document is never
thrown away — and log an INTEGRITY event quoting the exact offending span. The trace showing
the caught injection *is* the demo.

Ten rules, and **their order is load-bearing**: `invisible_unicode` runs first, because
`i<ZWSP>gnore previous instructions` defeats every text rule below it until the zero-width
space is gone. The rest cover instruction override, role reassignment, chat control tokens
(`<|im_start|>`, `[INST]`, `<<SYS>>`), output hijacking, score manipulation, base64 blobs
(decoded and previewed — the decoded payload is the evidence), and keyword stuffing in both
run-shape and frequency-share forms.

Each match emits one INTEGRITY event carrying the offending text as `evidence_span`. The bus
lifts a single `injection_stripped` flag onto the parent event.

### The graph

```
hidden_score(v) = z(PPR(v)) - z(visibility(v))
visibility(v)   = log1p(followers) + log1p(owned-repo stars) + log1p(karma)
```

High network centrality, low public visibility. Edges come from **co-membership** — two
people both have an event pointing at the same repo, paper, or thread — never from a
pair list, because an event carries exactly one `entity_id`.

Two details that matter:

- **An edge is stamped with the *later* of the two participants' timestamps.** The
  collaboration is not observable until both halves exist; stamping the earlier one would
  leak an edge into the past and invalidate the backtest.
- **Stars only count when ownership is explicit.** Unknown ownership means zero — never a
  guess in the founder's favour.

Groups larger than 40 members are dropped entirely rather than truncated: a 300-reply thread
is not 45,000 collaborations, and the clique would swamp the graph.

`access_lift(picks)` is the closing number — the share of picks at or below the 25th
percentile of visibility across the whole graph. A documented percentile rather than an
absolute follower count, so it stays meaningful as the corpus grows.

---

## 5. Memory and scoring

### The event log

Append-only, and enforced four ways: by convention, by `insert or ignore`, by SQLite
triggers, and by a Postgres `reject_mutation()` trigger. Corrections are new events; nothing
is ever updated or deleted.

Backend is chosen by DSN. All call sites are written in the SQLite dialect and a translation
layer rewrites for Postgres, so nothing above `memory/db.py` knows which backend it's talking
to. The Postgres connection is autocommit — a rejected append-only write must never leave the
session in an aborted transaction — and reconnects once on a dropped connection, because the
Supabase session pooler hangs up on idle sessions.

Foreign keys are deliberately **off**: sourcing stamps `entity_id`/`company_id` from
resolution before those rows necessarily exist, and an unresolvable id must not reject an
observation.

### Entity resolution

Four weighted signals, summed then clamped:

```
identity    exact email 0.95  |  shared url/handle 0.60
name        up to 0.55, Jaro-Winkler over transliterated names
co-occur    up to 0.25, shared repo / thread / paper
temporal   -0.20 when the two activity eras do not overlap at all
```

Thresholds: merge above 0.85, new below 0.40, **AMBIGUOUS in between**.

A name alone tops out at 0.55 — deliberately inside the ambiguous band. Two real people
called "Wei Zhang" must never merge just because fuzzy matching says 1.0. Names are
transliterated and normalized before matching, or non-Latin names silently vanish.

Ambiguity is surfaced, never guessed: the memo says "we could not confirm this is the same
person," and every decision is written twice — to a `merges` ledger and as an `ENTITY_MERGE`
event.

### The Founder Score

A local-linear-trend Kalman filter over state `x = [mu, nu]` — capability level and momentum.

```
F = [[1, dt], [0, 1]]
Q = Q_ACCEL * [[dt^3/3, dt^2/2], [dt^2/2, dt]]     # continuous white-noise acceleration

Score = mu        Band = sqrt(P[0,0])        Trend = nu
```

`Trend` is the state vector's momentum term — **structural, never a difference of scores**.
That is the whole reason for choosing this model.

Two properties carry the design:

- **Process noise scales with elapsed time, not event count.** A founder silent for a year
  should widen; one shipping weekly should not.
- **A terminal predict runs forward to `as_of`.** Silence costs something — a founder whose
  last signal was a year ago cannot keep the tight band they earned back then.

Observation noise: `r = R0 / self_consistency * source_penalty * kind_noise`. Self-reported
sources are noisier (deck 2.0, github 0.6, proof 0.15). Proof events are fresh, verified and
behavioral, so they move the score hard — that is the intended demo moment.

Contradicted claims are filtered out **before** becoming observations, at the boundary, with
the dropped ids kept as receipts.

`mu` is clamped to [0,1]. **`band` deliberately is not** — a capability level of 1.3 is a
display bug, but the band stays honest.

A Beta-Binomial fallback with exponential forgetting sits behind `SCORE_MODEL=beta_binomial`.
It consumes the identical `observations()` output, so contradiction filtering, source
penalties and `as_of` scoping are the same on both paths. It exists as a demo-time escape
hatch and was verified working early, not late.

### Derivation

`core/pipeline.py` is the conductor: raw observations become derived observations. It exists
because every stage module was built to read the event log and write back to it, and
initially nothing ran the derivation — so scores sat at the prior no matter how rich a
founder's history was.

Green flags are evaluated at **successive monthly checkpoints**, not once at the cutoff.
Two reasons, both measured:

- A single call yields a single observation, and a filter given one point has no trend to
  estimate and accumulates process noise across the whole gap — a band of 8.96 on a 0..1
  scale. A trajectory needs a series.
- Monthly rather than per-event, so a founder with 200 commits in one week contributes one
  reading, not 200. Volume must not masquerade as certainty.

Derived events get deterministic `uuid5` ids, so re-running the pipeline appends nothing.
That matters because an append-only log has no undo.

---

## 6. Intelligence

### Green flags — the sensor, and not an LLM

34 deterministic, interpretable rules over structured events, in clusters: shipping cadence,
iteration, learning from failure, ambiguity-to-scoping, technical depth, users touching the
artifact, and proof-protocol behavior.

Two choices matter most:

- **Inapplicable rules are skipped, not scored zero.** A designer with no GitHub is never
  penalized for it. Absence is the gate's job, not the sensor's.
- **The proof-protocol rules carry the highest weights**, topped by `proof_pushed_back` at
  5.0 — pushing back on a bad constraint is the sharpest signal in the system.

Anti-gaming is built in: the burst rule refuses to fire on raw commit count alone, requiring
tests, diff entropy, or file diversity.

Output is `y = fired_weight / total_weight`, with noise inversely proportional to
`sqrt(n_events * mean_confidence)`.

### Three-axis screening

**Founder** (from the Kalman filter, not an LLM), **Market** (LLM), **Idea-vs-Market** (LLM).

The axes are **never averaged** — not here, not in the ranked list, not in the UI. A great
founder in a dead market is a different decision from the reverse, and averaging destroys
exactly that distinction. Ranking uses a lexicographic tuple: a stated preference ordering,
not a blend.

Every LLM axis must cite `evidence_event_ids` drawn from the supplied list. Cited ids are
intersected with what the model was actually shown; invented ids are dropped; and **if zero
receipts survive, the axis falls back to uninformative with confidence 0.0**. Nothing
invented ever reaches a score. Thin evidence means low confidence, never a fabricated number.

### The gate

```
mu + band < 0.45                          -> NO_CALL          upper bound still below threshold
mu >= 0.70 and band <= 0.20               -> PROCEED          strong signal, narrow uncertainty
technical claim, no code, and mu < 0.60   -> NO_CALL          suspicious absence
otherwise                                 -> PROOF_PROTOCOL
```

Note the asymmetry: rejection tests the *upper* confidence bound, so wide uncertainty never
kills a founder on a low mean alone.

The absence classifier is the part to get right. Signal absent because irrelevant (a designer
with no GitHub) is not a red flag. Signal absent and suspicious (an infra founder claiming a
distributed system with no code anywhere) is. Getting this backwards punishes exactly the
founders the thesis exists to find.

Ambiguous founder identity routes to PROOF_PROTOCOL — resolve ownership before making a call.

### The Proof Protocol

For founders with nothing public. The alternative to an automatic pass.

A challenge is **bound to the company's own highest-confidence deck claim**, not generic, and
contains two deliberate traps:

- an **ambiguous requirement** — do they ask, or assume and state?
- a **planted bad constraint** — do they push back, or comply?

The submission is an artifact plus a behavioral trace. Behavior weighting puts 0.55 on
constraint pushback. Timing is scored non-monotonically: a first commit within 3 minutes
scores **0.4**, not 1.0 — suspiciously instant. Commit regularity uses `median_gap / max_gap`,
so an even cadence scores high and one giant paste-dump scores low.

Both emitted events carry a permanent caveat — *a short proof exercise is informative but is
not full diligence* — and the confidence ceiling exists as a bar this path never reaches.
Seeded demo completions are labelled `seeded: True` with an explicit disclosure, because
saying it on stage scores better than a discovered fake.

### Attestation

`api/attest.py` hashes and signs nothing. It is a **provenance split** between server-observed
and client-asserted facts.

The server records challenge issue time in-process and merges `{**client_trace, **observed}`
so server values overwrite client claims. Trust accumulates: base 0.35, +0.25 anchored in
time, +0.30 if commits were fetched from the public repo, +0.10 if nothing was self-reported.

The sharp part: `apply()` **multiplies the event's confidence by that trust**, which flows
directly into the filter's observation noise. Proof events are the lowest-noise, highest-weight
observations in the system, so an unattested trace must not buy that weight. Demo runs are
pinned at 0.5 trust and labelled `demo_seeded`.

### The validator — four states

`VERIFIED` / `CONTRADICTED` / `UNVERIFIABLE` / `NOT_ATTEMPTED`, with a real distinction between
"we looked and found nothing" and "we did not look."

- **Empty search results mean UNVERIFIABLE, never CONTRADICTED.** Conflating them is how a
  founder whose footprint is not in English gets wrongly torched.
- A `VERIFIED` with no stored snippet and URL is downgraded to `NOT_ATTEMPTED`. Receipts or
  it did not happen.
- **The growth rule**: counter-evidence *older* than the claim downgrades CONTRADICTED to
  UNVERIFIABLE. "$40K ARR in March" against "pre-revenue in January" is growth, not fraud.
  Timestamps are what separate fraud-shaped from time-shaped.
- Live search runs **only when `as_of` is None**, so a historical replay can never be
  validated against present-day evidence.

Trust is per claim: verified 0.9 (0.6 if self-published), contradicted 0.15, otherwise 0.5.

### Dissent and the Council

The Dissent Engine is prompted adversarially on purpose — *do not write a balanced summary* —
because a polite balanced take makes the whole feature read as theater. Its real output is
`axis_spreads`: the per-axis gap between the bull score and the pessimistic re-score, i.e.
how far the evidence could actually move.

The Council runs three roles over the same frozen evidence packet — scout, skeptic, bias
auditor — resolved by a chair instructed to decide by evidence policy, "not vote count and
not an average." Hard-coded overrides do the real work:

- `REACH_OUT` is downgraded to `PROOF_PROTOCOL` unless the scout voted for it *with receipts*
  and no receipt-backed argument is blocking.
- `NO_CALL` is downgraded unless the skeptic or bias auditor voted for it *with receipts*.
- Any failure path attracts to `PROOF_PROTOCOL`, never to a confident call.

**The dissent lock is enforced in the response shape, not the frontend.** `deliberate()`
returns no decision and does not even run the council; only `view_dissent()` returns the
decision and anti-memo atomically. A validator makes the illegal states unrepresentable —
you cannot obtain a recommendation without the case against it. This is deliberate: the
frontend should not be *able* to bypass it, and neither should we during a live demo.

### The memo

Gaps and citations are computed in Python; only prose goes to the model. Any citation the
model invents is dropped, because a fabricated event id breaks the trace drill-down — which
is the thing people actually click.

Gaps are surfaced rather than filled: unverifiable and unattempted claims become explicit
gaps, and synthetic gaps are added when no validation ran or no independent source exists.
If no model is available the memo still ships, assembled from evidence, asserting nothing
extra.

---

## 7. API and dashboard

FastAPI, thin — routers call into the modules above.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/thesis` | thesis configuration |
| GET | `/companies` | ranked list |
| GET | `/companies/{id}` | detail, with a live founder-score overlay |
| GET | `/companies/{id}/trace/{event_id}` | evidence drill-down chain |
| GET | `/companies/{id}/score-history` | real filter runs at N cutoffs, never interpolation |
| GET | `/companies/{id}/memo` | five sections, gaps, citations |
| GET | `/companies/{id}/dissent` | bear case — **unlocks the memo recommendation** |
| POST | `/companies/{id}/council` | three-role deliberation |
| POST | `/companies/{id}/proof` | issue a challenge |
| POST | `/companies/{id}/proof/{cid}/grade` | attest, grade, append events |
| GET | `/hidden` | hidden ranking + access lift |
| GET | `/query` | natural-language compound query |
| GET | `/backtest` | calibration report |

`deps.degrade(live, fallback)` runs the real module and falls back to a fixture on anything
short of a 4xx. It exists because of the standing rule that no one blocks on a teammate —
mock against the contract signature, swap when the PR lands.

The dashboard is Next.js with four routes: `/` (a six-plate poster sequence driven by live
data), `/pipeline` (the working dashboard), `/backtest`, and `/company/[id]` (seven sections
in demo order). No function in `app/lib/api.ts` may reject — everything resolves to a result
carrying its own provenance, and a chip renders whether the data is live or fixture, so
liveness is never faked. A fallback is never *another record*: an unmatched id degrades to a
sparse view of itself, never to a different company.

---

## 8. Seed data

Fixtures are **events, never scores**. A pre-computed score is one the pipeline never has to
earn.

Thirteen companies across six archetypes, built as a test matrix:

| Type | Companies | What it tests |
|---|---|---|
| 1 Visible Builder | Tensorpage, Quillstack | Rich public footprint; 18 months of genuine iteration on one artifact |
| 2 Cold Start | Veritanode, Halcyon Runtime | Deck only, zero public signal. Founder and idea axes are `null` with `reason: "not a zero, an absence"`. Must route to PROOF_PROTOCOL, never NO_CALL |
| 3 Serial Founder | Meshledger, Baseplate | Founder history survives the company boundary — same `entity_id`, different `company_id`. Prior companies feed the score but are excluded from the list |
| 4 Contradiction | Arcwell (newer counter-evidence, `no_call`), Tallwind (older counter-evidence, `proceed`) | Same claim, same counter-post, only the ordering differs. **No verdicts are seeded** — the validator must reach it from `observed_at` alone |
| 5 Adversarial | Synthgrid (injection, keyword stuffing, 3,117 commits for 412 net lines, no tests), **Ferrite Labs — the control** | Ferrite has a *larger* burst with real substance and zero flags. If the detector flags Ferrite too, it is measuring activity rather than manipulation |
| 6 Invisible International | Zaryad, Tantu, Xiliu | Native-script names must survive transliteration without fragmenting into two entities. Institutions stored as neutral biographical facts, excluded from every scoring input |

Seeding shifts the whole authored corpus forward so the newest event lands about nine days
before today, preserving every relative gap. Without it, the filter's silence-decay correctly
scores every seeded founder as dormant. `VCBRAIN_NO_SHIFT=1` disables it.

---

## 9. The backtest

The falsification rig, and the reason to believe anything above.

`replay()` calls the ordinary production path with the clock moved back.
`assert_no_lookahead()` runs three times — on the company event set, on each founder's events
before scoring, and on every trajectory window. Stage failures are swallowed to `None`; a
`LookaheadError` is explicitly re-raised.

Truncation happens **twice** — once at collection and again at read time via `as_of` — because
this is the claim the whole pitch rests on.

`run_calibration()` reports:

- Per-member trajectories, each point a real filter run at that cutoff, never an
  interpolation backwards from the final value (which would be lookahead wearing a chart's
  clothing).
- `hit_rate` — winners clearing the threshold before breakout.
- **`fame_check_passed`** — the hard gate. If a control founder clears the threshold, the
  score is measuring fame rather than trajectory and the thesis is dead. Controls are matched
  contemporaries who did not break out; a cohort of winners alone proves nothing.
  `fame_check_evaluated` is reported separately because **vacuous truth is not a pass** —
  with no controls, the check simply did not run.

Current live output: `hit_rate 0.75` (3 of 4), `fame_check_passed: True`, zero controls
clearing. Computed, not seeded.

---

## 10. Known seams

Stated plainly so nobody mistakes them for finished work.

**The `/backtest` page still renders fixtures.** The backend now replays correctly, but the
frontend validates the response with `Array.isArray(v.trajectories)` and no `trajectories`
key exists in the runner's output — it returns `results`, `winners`, `controls`. The check
fails and the page falls back to its fixture every time. The calibration numbers on screen
are hardcoded even though the live ones now agree with them.

**`correctly_deprioritized_failure` is always `None`.** The cohort fixture carries it as a
top-level key rather than as a labelled cohort member, and there is no `failures` list, so
`load_cohort()` never yields a member with `label == "failure"`. The "one failure the system
correctly deprioritized" slide — described in D.md as the most credible in the deck — has no
data behind it. A backtest that only shows the winners it caught is a marketing document.

**Three ranking policies disagree.** `thesis.json` declares `min_axis_with_momentum_tiebreak`;
`companies.json` says rank comes from the weakest axis with momentum breaking ties;
`_rank_key`'s docstring says "gate, then founder trend, then founder level"; and the code
returns `(-founder.trend, -founder.score)` — no gate, no minimum axis, and the market and
idea axes never consulted. The implemented policy ranks by momentum first, which is
defensible, but it is not the documented one and it contradicts the weakest-axis claim that
the no-blended-score design rests on.

**`SEED_FOUNDERS` is empty**, so every `hidden_ranking()` call takes the degree-based
fallback. The system logs this honestly — *NO SEED FOUNDER RESOLVED, proximity is to hubs,
not to breakouts* — but "proximity to greatness" is currently "proximity to hubs." This is
the highest-leverage unset value in the codebase.

**HN contributes nothing to visibility.** The scanner emits `points` and `num_comments`;
the visibility function reads `karma` and `hn_karma`. A silent no-op.

**A GitHub profile's `observed_at` is the account creation date**, while its follower count
is read now. The "latest reading at or before `as_of`" logic therefore dates a current
follower count to the account's creation year.

---

## 11. Running it

```bash
make setup     # uv sync, create .env from .env.example
make test      # 427 tests, should pass on a clean checkout
make api       # http://localhost:8000/health
```

`DATABASE_URL` selects the backend — a `postgresql://` DSN uses Supabase, anything else uses
SQLite at `data/vcbrain.db`. `VCBRAIN_DB_PATH` always forces SQLite, so tests never reach the
network. Note that the direct Supabase host is IPv6-only; use the session pooler URL.

`SCORE_MODEL=beta_binomial` swaps the estimator. `LLM_PROVIDER` switches vendor in one place.
LLM completions are cached to `data/llm_cache/` by prompt hash — the pipeline gets re-run
dozens of times and should not be paid for dozens of times.
