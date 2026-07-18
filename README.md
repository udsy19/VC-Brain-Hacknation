# VC Brain

Finds founders others can't see (graph diffusion), tests the ones who have nothing to
show (Proof Protocol), scores them with a model that sharpens over time (state-space
filter), argues with itself before spending money (Dissent Engine), and proves it
against history (time-machine backtest) — with receipts for every claim.

One vertical: **AI-infra / dev-tools founders.** Depth over breadth, always.

## Start here

```bash
make setup          # uv sync + create .env from .env.example
# fill in .env, then apply schema/migrations/001_init.sql to Supabase
make test           # 30 invariant tests, should pass on a clean checkout
make api            # http://localhost:8000/health
```

Then read, in this order:
1. **[SHARED.md](SHARED.md)** — stack, event schema, directory ownership, hard gates. Everyone.
2. Your role file: **[A.md](A.md)** memory/score · **[B.md](B.md)** sourcing/graph · **[C.md](C.md)** reasoning · **[D.md](D.md)** experience/backtest

## Layout — ownership is by directory, so PRs don't collide

```
schema/         A   Pydantic models + SQL. THE contract.
core/           A   llm.py (provider wrapper), search.py (Tavily). Everyone imports.
memory/         A   event store, entity resolution, Founder Score
sourcing/       B   scanners, ingestion bus, injection guard, graph + PPR
intelligence/   C   green flags, screening, validator, Proof Protocol, Dissent
api/ app/       D   FastAPI + Next.js dashboard
backtest/       D   time-machine rig
tests/          all
```

**Never edit a directory you don't own.** Need a change there? Ask the owner in chat.

## The four invariants

These are enforced by tests and DB constraints, not by good intentions — because good
intentions do not survive hour 19.

| # | Invariant | Enforced by |
|---|---|---|
| 1 | No lookahead — scoring at `as_of` sees only `observed_at <= as_of` | `store.events()` requires `as_of`; `backtest.assert_no_lookahead`; tz-aware validator |
| 2 | Trust is per-claim, never one company number | `ClaimVerdict.trust`; no company-level trust field exists |
| 3 | No pedigree anywhere in scoring | `tests/test_no_pedigree.py` greps all source against `intelligence/banned.py` |
| 4 | Founder text is data, never instructions | `llm.complete(untrusted=...)` applies the wrapper; `sourcing/sanitize.py` logs strikes |

## Working agreement

- Branches `a/`, `b/`, `c/`, `d/` + short name. **Merge to main every 3 hours.**
- `main` must always run. Break it and fixing it is your only job.
- H3 schema gate · H12 fame-vs-trajectory gate · H16 integration · **H21 feature freeze**.
- Cut list is in SHARED.md §7. When you're behind, cut in that order without re-litigating.
