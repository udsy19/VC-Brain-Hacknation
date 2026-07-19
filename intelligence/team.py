"""Team-level scoring: composition and complementary coverage, DERIVED not stored.

The gap this closes: `screen.three_axis` took `entity_ids[0]` for the Founder axis, so
a two-founder company scored as if the second founder did not exist. Which founder
"first" meant was itself an accident of event ordering, not a choice.

Three things this module deliberately does NOT do:

  1. **It does not touch `memory/score.py`.** The per-entity Kalman score is
     authoritative and unchanged. Everything here is a pure function of N `FounderScore`
     values and N `TraitProfile` values, recomputed from an `as_of`-scoped read on every
     call. Nothing is persisted; there is no mutable team field to go stale.

  2. **It does not infer roles from credentials.** SHARED.md invariant #3 bans school,
     employer brand and investor name, so "technical co-founder / business co-founder"
     cannot be read off a title. See `role_split` below: today it is always
     NOT_DETERMINABLE, and that is a finding rather than a placeholder.

  3. **It does not average.** See `_aggregate` for why, and for what it does instead.

Who counts as a member: every distinct `entity_id` resolved against the company at or
before `as_of`. That is the same set `api.routers.deps.founder_entity_ids` returns.
LIMITATION, stated rather than hidden: entity resolution does not carry a founder/
non-founder role marker, so an incidentally-resolved person (an HN commenter on the
company's thread, say) would be counted as a member. In the current corpus every
resolved entity per company IS the founder, so this is latent rather than active — but
it is the first thing to break when the graph starts resolving non-founders.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from intelligence import traits as traits_mod
from schema.events import Axis, Event, FounderScore

# --- constants, each with the reason it has the value it has -----------------------

# The most a team's composition may add to its strongest member's level. Set to 0.15
# because a typical band on a well-evidenced founder in this corpus is 0.06-0.12: a
# STRUCTURAL bonus must not outrun the MEASUREMENT precision of the thing it adjusts,
# or the composition story starts dominating the evidence. This is a design judgement,
# argued not fitted, and should be challenged on the argument.
COMPLEMENTARITY_CAP = 0.15

# Prior standard deviation of the level, sqrt(P0[0]) from memory/score.py. The team band
# is clipped here for the same reason the per-founder filter clips its covariance: no
# amount of disagreement between members can make us more uncertain about a team than
# we were before we had heard of any of them.
_PRIOR_SD = 0.5

NOT_DETERMINABLE = "not_determinable"

# Why a technical/business role split cannot be emitted from this system's evidence.
# The seven traits in docs/TRAITS.md are ALL builder traits — they were derived from
# flags.py's rule groupings, and flags.py observes commits, releases, papers, HN
# threads and proof-protocol behaviour. There is no rule in the taxonomy that fires on
# distribution, pricing, hiring or customer development, so there is no evidence from
# which a "business co-founder" could be observed. Reading the split off a job title
# would violate SHARED.md #3 (no school, no employer brand, no investor name); reading
# it off "who has no GitHub" would be the visibility proxy the whole product exists to
# avoid. So the honest output is this string.
ROLE_SPLIT_REASON = (
    "The trait taxonomy observes only builder behaviour (shipping, iteration, rigor, "
    "depth, scoping, learning, scrutiny). No signal in the system evidences a "
    "commercial or go-to-market role, and inferring one from a title would breach the "
    "substance-only invariant (SHARED.md #3). A role split is not observable from the "
    "evidence we hold."
)


@dataclass(frozen=True)
class Member:
    """One founder's contribution to the team view. A reshape, never a re-derivation."""

    entity_id: UUID
    mu: float
    band: float
    trend: float
    observed_traits: tuple[str, ...]  # traits past their independent-channel gate
    evidenced_traits: tuple[str, ...]  # traits with any rule fired, gate or not
    contributing_event_ids: tuple[UUID, ...]


@dataclass(frozen=True)
class TeamScore:
    """Derived team view. Every field is recomputed per call; none is stored."""

    company_id: UUID
    as_of: datetime
    members: tuple[Member, ...]
    mu: float
    band: float
    trend: float

    # --- the aggregation, made visible rather than buried ---------------------------
    # The brief on this module was explicit that the choice must be inspectable in the
    # output. These four fields are what let a reader reconstruct `mu` by hand:
    #   mu == clip(anchor_mu + complementarity_lift, 0, 1)
    anchor_mu: float  # strongest member — the level the team is anchored on
    weakest_mu: float
    mean_mu: float  # reported for comparison ONLY; never used to compute mu
    complementarity_lift: float  # 0.0 when solo, and when nobody adds coverage

    complementarity: float | None  # None => not measured, NOT zero
    complementarity_basis: str
    role_split: str
    role_split_reason: str

    dispersion: float  # spread between members; the band-widening term
    trait_coverage: dict[str, tuple[UUID, ...]]  # trait -> members observed on it
    unique_coverage: dict[str, tuple[str, ...]]  # member (str uuid) -> traits only they hold
    contributing_event_ids: tuple[UUID, ...]

    @property
    def n_founders(self) -> int:
        return len(self.members)

    @property
    def is_solo(self) -> bool:
        return len(self.members) == 1

    @property
    def basis(self) -> str:
        if not self.members:
            return "no_resolved_founder"
        return "solo" if self.is_solo else "team"


def _member(fs: FounderScore, profile: traits_mod.TraitProfile) -> Member:
    observed = tuple(
        trait_id for trait_id, trait in sorted(profile.traits.items()) if trait.observed
    )
    evidenced = tuple(
        trait_id for trait_id, trait in sorted(profile.traits.items()) if trait.evidenced
    )
    return Member(
        entity_id=fs.entity_id,
        mu=fs.mu,
        band=fs.band,
        trend=fs.trend,
        observed_traits=observed,
        evidenced_traits=evidenced,
        contributing_event_ids=tuple(fs.contributing_event_ids),
    )


def _complementarity(members: tuple[Member, ...], anchor: Member) -> tuple[float, float, dict]:
    """Bounded lift from co-founders who cover ground the anchor does not.

    Complementarity is measured on OBSERVED traits only — traits whose independent-
    channel gate (docs/TRAITS.md §3) has been met. That gate is load-bearing here: it
    is what stops a keyword-stuffed deck from manufacturing a fake complementary
    co-founder. A trait the anchor merely *evidenced* does not block a co-founder from
    claiming it, and a trait a co-founder merely evidenced does not earn any lift.

    Per co-founder j:  lift_j = (|unique_j| / |taxonomy|) * mu_j

    The `* mu_j` term is the point. Coverage alone is not capability: a co-founder who
    is the only person on the team observed to ship, but who ships badly, adds little.
    A weak co-founder cannot lift a team by standing in an empty part of the taxonomy.
    """
    total_traits = len(traits_mod.trait_ids())
    anchor_observed = set(anchor.observed_traits)
    unique: dict[str, tuple[str, ...]] = {}
    raw_lift = 0.0
    for member in members:
        if member.entity_id == anchor.entity_id:
            unique[str(member.entity_id)] = ()
            continue
        held_by_others = set(anchor_observed)
        for other in members:
            if other.entity_id not in (member.entity_id, anchor.entity_id):
                held_by_others |= set(other.observed_traits)
        only_mine = tuple(sorted(set(member.observed_traits) - held_by_others))
        unique[str(member.entity_id)] = only_mine
        if total_traits:
            raw_lift += (len(only_mine) / total_traits) * member.mu
    return raw_lift, min(raw_lift, COMPLEMENTARITY_CAP), unique


def _band(members: tuple[Member, ...], dispersion: float) -> float:
    """Conservative propagation. An aggregate of uncertain quantities is MORE uncertain.

    band = max_i(band_i) + dispersion, clipped at the prior sd.

    This is deliberately the conservative choice and is stated as such. The tempting
    alternative — treating members as independent measurements of one latent team
    quality and shrinking the band by sqrt(n) — is wrong twice over: the members are not
    measuring the same quantity (that is the entire premise of complementarity), and
    their evidence is not independent (co-founders commit to the same repos and appear
    in the same threads, so their observation noise is correlated). Under unknown
    positive correlation the worst case is full correlation, which gives no shrinkage at
    all — hence `max`, not a variance-weighted combination.

    The `+ dispersion` term then makes the band strictly WIDEN when members disagree:
    "one strong and one weak founder" is a less certain read than either alone, because
    which of them the company's outcome depends on is itself unknown.

    Guaranteed invariant, asserted in tests/test_team.py:
        team.band >= max(member.band for member in members)
    """
    if not members:
        return _PRIOR_SD
    widest = max(member.band for member in members)
    return min(widest + dispersion, _PRIOR_SD)


def _aggregate(members: tuple[Member, ...]) -> tuple[float, float, float, dict]:
    """Anchor on the strongest member, then add bounded lift for genuine complementarity.

    NOT a mean. A mean says a 0.8 founder who takes on a 0.3 co-founder just got worse,
    which is false — the 0.8 founder is still there, and the company is not a worse bet
    for containing an additional person. A mean also collapses precisely the distinction
    this module exists to preserve: one strong solo founder and two mediocre co-founders
    are different bets, and `(0.8) vs mean(0.45, 0.45)` reports them as nearly the same
    when the difference is the whole decision.

    NOT a max either. A pure max says the second founder is worth exactly nothing, which
    is the same claim as the `entity_ids[0]` bug this module was written to fix, only
    reached deliberately.

    So: anchor on the max, and let additional founders earn a bounded increment by
    covering traits the anchor is not observed on. `mean_mu` is still reported, because a
    reader is entitled to see the number this deliberately did not use.
    """
    if not members:
        return 0.5, 0.0, 0.0, {}
    anchor = max(members, key=lambda m: (m.mu, str(m.entity_id)))
    raw_lift, lift, unique = _complementarity(members, anchor)
    return min(anchor.mu + lift, 1.0), lift, raw_lift, unique


def team_score(
    company_id: UUID, as_of: datetime, events: Sequence[Event] | None = None
) -> TeamScore:
    """Aggregate every resolved founder of ``company_id`` as observed at ``as_of``.

    Store-backed and `as_of`-scoped throughout (invariant #1): the member set, each
    Kalman score and each trait profile are all read at ``as_of``.

    ``events`` lets a caller that has ALREADY made the as_of-scoped company read hand it
    in rather than pay for it twice — same escape hatch, and same signature position, as
    ``traits.profile``. `screen.three_axis` uses it, which is why adding team scoring
    costs that path zero extra store round-trips.
    """
    from memory import score as founder_filter
    from memory import store

    if events is None:
        events = store.events(company_id=company_id, as_of=as_of)

    seen: dict[UUID, None] = {}
    for event in events:
        if event.entity_id is not None:
            seen.setdefault(event.entity_id, None)
    entity_ids = list(seen)

    members = tuple(
        _member(
            founder_filter.founder(entity_id, as_of),
            traits_mod.profile(entity_id, as_of, events, attribute=False),
        )
        for entity_id in entity_ids
    )
    return assemble(company_id, as_of, members)


def assemble(company_id: UUID, as_of: datetime, members: tuple[Member, ...]) -> TeamScore:
    """Pure aggregation over already-computed members. Store-free, so it is testable."""
    mu, lift, _raw_lift, unique = _aggregate(members)

    mus = [member.mu for member in members]
    anchor_mu = max(mus) if mus else 0.5
    weakest_mu = min(mus) if mus else 0.5
    mean_mu = (sum(mus) / len(mus)) if mus else 0.5

    # Half the spread: the band widens by the distance from the aggregate to either
    # extreme, not by the full extreme-to-extreme range, which would double-count it.
    dispersion = (anchor_mu - weakest_mu) / 2.0 if len(members) > 1 else 0.0
    band = _band(members, dispersion)

    anchor = max(members, key=lambda m: (m.mu, str(m.entity_id))) if members else None
    trend = anchor.trend if anchor is not None else 0.0

    coverage: dict[str, tuple[UUID, ...]] = {}
    for trait_id in traits_mod.trait_ids():
        holders = tuple(m.entity_id for m in members if trait_id in m.observed_traits)
        if holders:
            coverage[trait_id] = holders

    # A solo founder has no complementarity to measure. Emitting 0.0 would say "we
    # measured it and there was none", which is a different and false claim. None means
    # not measured, and `basis` says why. A solo founder is therefore neither penalised
    # nor credited by construction: mu and band are byte-identical to their own
    # FounderScore, which tests/test_team.py asserts.
    if len(members) < 2:
        complementarity: float | None = None
        basis = (
            "solo founder — complementarity is undefined for a team of one, not zero"
            if len(members) == 1
            else "no resolved founder entity — nothing to aggregate"
        )
    else:
        complementarity = lift
        basis = (
            f"observed-trait coverage across {len(members)} founders, "
            f"capped at {COMPLEMENTARITY_CAP}"
        )

    contributing: list[UUID] = []
    for member in members:
        for event_id in member.contributing_event_ids:
            if event_id not in contributing:
                contributing.append(event_id)

    return TeamScore(
        company_id=company_id,
        as_of=as_of,
        members=members,
        mu=mu,
        band=band,
        trend=trend,
        anchor_mu=anchor_mu,
        weakest_mu=weakest_mu,
        mean_mu=mean_mu,
        complementarity_lift=lift,
        complementarity=complementarity,
        complementarity_basis=basis,
        role_split=NOT_DETERMINABLE,
        role_split_reason=ROLE_SPLIT_REASON,
        dispersion=dispersion,
        trait_coverage=coverage,
        unique_coverage=unique,
        contributing_event_ids=tuple(contributing),
    )


def team_axis(ts: TeamScore) -> Axis:
    """The team score reshaped onto the Founder axis, exactly as `founder_axis` does.

    Same convention as `screen.founder_axis`: score=mu, trend=trend, and confidence
    narrows with the band. Because the band can only widen with more members, a team
    never reports MORE confidence than its most certain member — which is the property
    that makes it safe to put this behind the existing axis.
    """
    return Axis(
        score=ts.mu,
        trend=ts.trend,
        confidence=max(0.0, min(1.0, 1.0 - ts.band)),
        evidence_event_ids=list(ts.contributing_event_ids),
    )


def as_dict(ts: TeamScore) -> dict:
    """JSON-safe view for the API. Derived on read; nothing here is persisted."""
    return {
        "company_id": str(ts.company_id),
        "as_of": ts.as_of.isoformat(),
        "n_founders": ts.n_founders,
        "basis": ts.basis,
        "mu": ts.mu,
        "band": ts.band,
        "trend": ts.trend,
        "aggregation": {
            "method": "anchor_plus_complementarity",
            "anchor_mu": ts.anchor_mu,
            "weakest_mu": ts.weakest_mu,
            "mean_mu": ts.mean_mu,
            "complementarity_lift": ts.complementarity_lift,
            "cap": COMPLEMENTARITY_CAP,
            "note": "mu = min(anchor_mu + complementarity_lift, 1.0). mean_mu is shown "
            "for comparison and is deliberately NOT used.",
        },
        "uncertainty": {
            "band": ts.band,
            "widest_member_band": max((m.band for m in ts.members), default=_PRIOR_SD),
            "dispersion": ts.dispersion,
            "propagation": "conservative: max(member bands) + dispersion, clipped at the "
            "prior sd. Never tighter than the widest member.",
        },
        "complementarity": ts.complementarity,
        "complementarity_basis": ts.complementarity_basis,
        "role_split": ts.role_split,
        "role_split_reason": ts.role_split_reason,
        "trait_coverage": {k: [str(u) for u in v] for k, v in ts.trait_coverage.items()},
        "unique_coverage": {k: list(v) for k, v in ts.unique_coverage.items()},
        "members": [
            {
                "entity_id": str(m.entity_id),
                "mu": m.mu,
                "band": m.band,
                "trend": m.trend,
                "observed_traits": list(m.observed_traits),
                "evidenced_traits": list(m.evidenced_traits),
            }
            for m in ts.members
        ],
    }
