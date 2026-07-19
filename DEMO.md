# Demo script — 2.5 minutes

Every number below was read off the running system, not written from the plan. If a
number here disagrees with the screen, the screen is right and this file is stale —
re-run `make demo-check` (below) before trusting it.

---

## Pre-flight (do this 10 minutes before, not 1)

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000    # backend
cd app && pnpm build && pnpm start -p 3005                # frontend
curl -s localhost:8000/health | python3 -m json.tool      # read the warnings
```

`/health` reports the things that fail **quietly**. Check `warnings` is empty. If it
mentions the GitHub rate limit, live scanning will return nothing — which on stage
is indistinguishable from a founder having no footprint. Set `GITHUB_TOKEN` in
`.env` (60/hr → 5000/hr) and restart.

Open all four tabs before you start talking: `/`, `/pipeline`, `/company/intl-zaryad`,
`/backtest`.

---

## The beats

**0 · Thesis (15s)** — `/pipeline`, thesis panel.
> "Config, not code. Sectors, stage, geo, check size. Geography is deliberately
> unrestricted — a geographic filter is the cheapest way to systematically miss the
> founder we're about to show you."

**1 · Visible builder (25s)** — Tensorpage, **rank 1**, founder **78.8**.
Click into the score → trace drill-down → a quoted span with its slide ID and URL.
> "Every number is clickable down to the text it came from. No score exists here
> without receipts."

**2 · Cold start — the centrepiece (35s)** — Veritanode, gate reads **`proof_protocol`**.
> "Deck only. No public footprint. Most systems either guess or abstain. We generate a
> challenge from this founder's own central technical claim — and we plant two things
> in it: one ambiguous requirement, and one constraint that's subtly wrong."

Show the planted constraint on the Proof Protocol panel. Then say the honest part:
> "The generator, the grader and the attestation are real. This completion is
> pre-run — we're telling you that rather than letting you wonder."

**3 · Serial founder (10s)** — Meshledger, founder **64.2**.
> "Same founder, second company. The founder score persists across the boundary; the
> opportunity score resets. Score it from either company and you get the same number."

**4 · Trust — pick one, keep the other as backup (25s)**
- *Contradiction*: Arcwell vs Tallwind — **identical $40,000 ARR claims**, opposite
  outcomes. Only the ordering differs. "If we flagged the second one, we'd be
  pattern-matching on words instead of reasoning about time."
- *Adversarial*: Synthgrid **37.3** vs the control Ferrite **42.7** — and **Ferrite's
  burst is bigger**. "We don't false-positive fast builders, and that's measured, not
  asserted." Injection caught on slide 7 is visible in the trace.

**5 · Backtest (25s)** — `/backtest`. **fame_check passed**, **hit_rate 0.75**,
8 trajectories.
> "Winners rise, controls stay flat and below the line. Three of four caught — and
> here's the fourth, the one we missed. Plus Veridian Stack at 38 against a 62
> threshold: correctly deprioritised. A backtest that only shows its winners is a
> marketing document."

**6 · Close — the equity thesis (20s)** — Zaryad Compute, **rank 2**, founder **67.5**.
> "Transliterated name, non-prestige institution, non-English sources. Second on the
> list. Earlier today this founder scored at the prior with zero evidence, because a
> transliterated name was silently disqualifying every event attached to them. That
> bug is exactly what this system exists to prevent, and we found it in our own code."

Point at the integrity flags on the page: surfaced, not hidden.

---

## Cut order if you're over time
Beat 3 (10s) → beat 0 (15s) → one half of beat 4. Never cut 2, 5 or 6.

## If something breaks
Every route falls back to fixtures and says so with a chip. A `FIXTURE DATA` chip on
screen is survivable — claiming live data while showing fixtures is not. If the
backend dies, say so and keep going; the page still renders.

## Say these out loud
- The Proof Protocol completion is pre-run.
- Backtest winners/controls are a curated cohort; the **replay** through the live code
  path is real, the cohort is hand-collected.
- Which axes are computed, precisely: on the RANKED LIST the founder axis is computed
  live and market / idea-vs-market are seeded, because assessing them costs an LLM call
  per company (~7s) and the list would take ~95s. Opening a company's DETAIL page
  computes all three for real and warms them into the list. Every axis carries a `live`
  flag and the UI shows a SEEDED — NOT COMPUTED chip, so you never have to remember
  which is which — point at the chip rather than asserting it.
- `access_lift` now reports **0.556** over a real visibility spread. It returns
  nothing when visibility is uniform, so a number means it actually measured.

## make demo-check
```bash
curl -s localhost:8000/health | python3 -m json.tool
curl -s localhost:8000/companies | python3 -c "import sys,json;[print(r['rank'],r['name'],r['axes']['founder']['score']) for r in json.load(sys.stdin)[:6]]"
curl -s localhost:8000/backtest  | python3 -c "import sys,json;d=json.load(sys.stdin);print('fame',d['fame_check_passed'],'hit',d['hit_rate'],'traj',len(d['trajectories']))"
```

---

# Deploying to Vercel

## THE ONE THING THAT WILL BREAK THE DEMO IF YOU SKIP IT

**Ship `data/llm_cache/`.** It is currently gitignored (`data/llm_cache/*`), so it does
not reach the deployment, and every LLM-backed route then makes its calls for real.
Measured on a read-only filesystem with the exact production code:

| route | cache shipped | cache absent |
|---|---|---|
| `GET /companies/{id}/dissent` | **10.8s** | **90.8s — exceeds `maxDuration: 60`, Vercel kills it** |
| `GET /companies/{id}/memo` | 9.0s | 27.1s |

The dissent view is the signature beat, and without the cache it returns a 504 on stage.
The cache is 11MB against a 250MB ceiling. To ship it, drop the `data/llm_cache/*` line
from `.gitignore` and commit the directory. Cache keys are derived from the prompt, so
they only hit if the deployment reads the SAME database the cache was warmed against —
which it does, since both point at the same Supabase project.

If you would rather not ship it, raise `maxDuration` to 300 in `vercel.json`. That
requires a Pro plan with Fluid compute; on Hobby the ceiling is 60 and the deploy will
be rejected.

## Required environment variables (names only — never paste values into a file)

Set in the Vercel project, Production + Preview:

| name | why |
|---|---|
| `DATABASE_URL` | **Must be the SESSION POOLER host** — see below. Everything durable lives here. |
| `OPENAI_API_KEY` | Memo, dissent, screening, proof generation. |
| `LLM_PROVIDER` | Optional; `openai` (default) or `anthropic`. |
| `ANTHROPIC_API_KEY` | Only if `LLM_PROVIDER=anthropic`. |
| `TAVILY_API_KEY` | Web evidence. Absent = search degrades, no crash. |
| `GITHUB_TOKEN` | 60/hr → 5000/hr. Absent = live scanning silently returns nothing. |
| `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` | Read by config; the store itself talks psycopg. |

`VERCEL` and `VERCEL_ENV` are set by the platform — do not set them yourself. `VERCEL`
is what flips `core.config.cache_root()` to `/tmp`, and `VERCEL_ENV` is what makes
session cookies `Secure`.

### DATABASE_URL must be the session pooler

Use `aws-0-<region>.pooler.supabase.com:5432`. The direct
`db.<project>.supabase.co` host is **IPv6-only and unreachable from Vercel** — this has
bitten this project before. Verify without printing the secret:

```bash
python3 -c "import os,socket;from urllib.parse import urlparse;h=urlparse(os.environ['DATABASE_URL']).hostname;print(h,'pooler' if 'pooler' in h else 'DIRECT — WILL NOT CONNECT');print(socket.getaddrinfo(h,None,socket.AF_INET)[0][4][0])"
```

## Apply the migrations before the first deploy

```bash
uv run python scripts/migrate.py    # 008 adds dissent_unlocks, proof_challenges, config_documents
```

The app also creates these three tables on first use, so a missed migration degrades
rather than breaks — but the Postgres-only row-level-security posture only comes from
the migration.

## What is different about the deployment (and why)

The deployment filesystem is **read-only except `/tmp`**, and consecutive requests may
land on **different processes**. Both change behaviour:

- **Caches** (`llm_cache`, `standout_cache`, decks) resolve under `/tmp` via
  `core.config.cache_root()`, and every write to them is guarded. `/tmp` does not
  survive the invocation, so caching is best-effort — which is exactly why the point
  above about shipping the warmed cache matters.
- **The dissent lock, proof challenges, and the edited thesis** live in Postgres
  (migration 008), because a module global is invisible to the next request and, worse,
  visible to the *wrong user* on a warm one.
- **The dissent lock is per-viewer.** Signed in, it is keyed to the user; anonymous, to
  an httpOnly `vcbrain_viewer` cookie minted the first time a bear case is served. One
  attendee opening the dissent does **not** unlock the recommendation for anybody else.
- **`PUT /thesis` returns 503 if it cannot persist**, rather than 200 over a lost edit.
  Uploaded decks are **not** durable and do not need to be — their claims are already
  events in Postgres and the dedupe key is the SHA-256, not the file.

## Bundle budget (measured)

123MB of Python dependencies + 4.3MB of tracked source (+11MB if you ship the LLM
cache) ≈ **138MB against the 250MB ceiling**. `data/seed/` is tracked and ships; the
routes read their fixtures from it.
