# The personalisation layer — design

Six asks, one architecture. Written before the code so the load-bearing decision is
argued rather than assumed.

---

## 0. The decision everything else follows from

**Separate what is TRUE about a founder from what this VC CARES about, and never let
the second contaminate the first.**

```
  OBJECTIVE (exists today, unchanged)          SUBJECTIVE (new, per-VC)
  ───────────────────────────────────          ────────────────────────
  evidence -> green flags -> Kalman            VC profile -> custom council
  3 axes, never averaged                       -> fit score + re-rank
  gate, validator, dissent                     -> founder-market fit
  core rank (min-axis)                         -> outbound eligibility
         │                                              │
         └──────── same evidence graph ─────────────────┘
```

Why this way round, when merging them would be less code:

1. **The receipts survive.** Every claim traces to a quoted span. A preference weight
   cannot be traced to evidence, because it isn't evidence — it's taste. Mixing them
   makes the whole audit trail unfalsifiable.
2. **Two VCs, same truth, different ranking.** That's the product. If preference moved
   the core score, the same founder would be *more capable* at a bolder fund, which is
   nonsense — the same error the thesis engine already avoids by moving the evidence
   bar rather than the score.
3. **The gap becomes a feature** (§2.3). You can only show a VC that their stated
   preferences diverge from their revealed ones if the two are stored separately.

The core rank stays VC-agnostic and auditable. The personal layer sits *on top*, is
always attributable to the profile that produced it, and is always removable — the UI
must be able to show "core rank" and "your rank" side by side, because a system that
can only show you your own taste back is a mirror, not an analyst.

---

## 1. Auth (the entry point)

Not the interesting part, but it gates everything: a VC profile needs an owner.

- Email + password, sessions in Postgres, `argon2` hashing. No third-party IdP —
  a hackathon demo that depends on an OAuth callback surviving a conference network
  is a demo that fails on stage.
- `vc_profile` is owned by a user; every personalised artifact carries `profile_id`.
- The API keeps working unauthenticated for the **core** rank. Personalisation is
  the thing that requires a session, so a broken login degrades to the objective
  product rather than to a blank page.

## 2. Capturing the VC

### 2.1 Two inputs, deliberately

**Past decisions** (upload: CSV/JSON) — *revealed* preference.
`company, sector, stage, decision (invested|passed|watched), date, rationale?, outcome?`

**Survey** (~12 questions) — *stated* preference. Forced trade-offs, not Likert
scales: "A technically exceptional founder in a crowded market, or a competent one
in an empty market — pick." Agreement scales measure agreeableness; trade-offs
measure priorities.

### 2.2 What we derive

```
axis_weights      how much this VC weights founder / market / idea-vs-market
red_lines         things that are disqualifying regardless of score
conviction_style  where on evidence-heavy <-> conviction-heavy they sit
sector_priors     revealed sector concentration
stage_priors      revealed stage concentration
lenses            3-5 council personas, derived (§3)
```

Every derived value carries **provenance**: which decisions or answers produced it,
and a confidence. A profile inferred from 4 decisions must say so — the same rule the
scorer applies to a founder with thin evidence applies to a VC with a thin history.

### 2.3 The stated-vs-revealed gap

Compute both independently and **show the divergence**. *"You rate technical depth
highest, and passed on 4 of the 5 deepest-technical founders you saw."*

This is the most defensible thing in the feature. It isn't flattery, it's a finding
about the user, produced by the same machinery that produces findings about founders —
and it's only possible because §0 keeps the two preference sources separate.

## 3. The custom council

`intelligence/council.py` already exists with fixed roles. The custom council
**instantiates personas from the profile** rather than inventing a new mechanism.

- Each lens is a persona + a weight + the profile fields that justified it.
- The council runs on the **same evidence graph** as the core analysis. It reweights
  and reinterprets; it never gets private evidence, or the two layers would be
  arguing about different facts — exactly the bug found in the dissent engine, where
  the bear case was blind to 20-25% of what the memo could see.
- Output: a `fit_score` with per-lens contributions, plus a **founder-market fit**
  assessment read through this VC's thesis.
- Dissent still applies. A custom council that only agrees with its VC is an echo,
  and the existing rule stands: the recommendation stays locked until the bear case
  has actually been served.

**The failure mode to design against:** a council tuned to a VC's history will
reproduce that VC's blind spots, with machine authority. Mitigation — always show
core rank beside personal rank, and surface *every* company where they disagree
sharply. The disagreements are the value; agreement is just confirmation.

## 4. Sources and citations

Detailed registry: `docs/SOURCES.md` (researched separately). Two rules here:

- **Allowlist, not open crawl.** A fixed set of domains, each with a stated reason to
  be trusted and a note on who it systematically misses. Tavily takes
  `include_domains`, so the restriction is enforceable at the call.
- **A URL that was never fetched cannot be cited.** Every citation resolves to a
  recorded fetch (url, fetched_at, status, content hash, quoted span + offset).
  This makes a fabricated link structurally impossible rather than merely unlikely,
  and it extends the rule the validator already applies: a VERIFIED claim with no
  stored span is downgraded to NOT_ATTEMPTED.

Per-source attribution — the brief asks for *why* a source adds or subtracts, not
just that it was consulted — is a first-class field, with direction, magnitude and the
span behind it.

## 5. Founder page readability

The current page shows everything the system knows. The ask is everything **useful**.

Ordering principle: **what would change the decision, first.** Rough order —
the recommendation and its blockers; the three axes with their weakest first (that's
what min-axis ranking keys on); contradicted and unverified claims; the load-bearing
claim from the dissent; then evidence, grouped by what it evidences rather than by
source; then provenance notes.

Cut: anything that is neither evidence nor a decision input. If a field cannot
complete the sentence *"this matters because…"*, it goes behind a disclosure.

## 6. Outbound

Only for companies that **genuinely pass** — and "pass" must be a computed gate
(`PROCEED`, no contradicted claims, no unresolved red lines), never a threshold
someone typed.

- Drafted from the evidence trace, so the mail cites what was actually observed
  ("your work on X") rather than generic flattery. `sourcing/activate.py` is the stub.
- **Never auto-sent.** Drafts land in a review queue; a human sends. The system is
  making a claim about a person to that person — that is not a decision to automate,
  and a hallucinated detail in a cold email is a reputational cost the user pays,
  not us.
- Suppression list, one-touch opt-out, and a record of every send.

---

## Build order

1. Auth + profile storage (everything else needs an owner)
2. Survey + decision upload -> profile derivation, with provenance
3. Custom council on the existing evidence graph -> fit score + FMF
4. Source registry + fetch-backed citations
5. Founder page rework
6. Outbound drafts + review queue

1-3 are the differentiator; 4 makes it trustworthy; 5-6 make it usable.

## What could make this worse, not better

- **Personalisation as a black box.** If a VC can't see why their rank differs from
  core, it's astrology. Every personal adjustment shows its lens and its weight.
- **Training the system to agree.** Highlight disagreement by default.
- **A profile from too little data.** State the confidence; below a threshold, run
  core rank only and say why.
- **Cold outreach at scale.** Volume is the failure mode. Gate hard, send by hand.
