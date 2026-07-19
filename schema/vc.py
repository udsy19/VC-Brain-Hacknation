"""The VC-side contract: accounts, the two preference sources, and what we derive.

Owner: personalisation layer (docs/DIFFERENTIATOR.md §1-§2).

THE LOAD-BEARING RULE (§0): stated preference (survey) and revealed preference (past
decisions) are two SEPARATE inputs and are never merged into one blob. They are stored
in separate tables, derived independently, and compared in `GapReport`. Merging them
would save code and destroy the only finding in here that is about the USER rather than
about a founder.

THE SECOND RULE: nothing in a derived profile may exist without a real submission behind
it. Every derived value carries `Provenance` naming the exact questions or decision rows
that produced it, plus a confidence. Where the submitted data cannot support an
inference, the field is ABSENT and the reason is recorded in `DerivedProfile.not_inferred`
— never filled with a plausible default. This is the same discipline the memo already
follows: gaps are flagged, never filled.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from ._backport import StrEnum


def normalize_email(raw: str) -> str:
    """Lowercased and trimmed, with a deliberately minimal shape check.

    Not `EmailStr`: that pulls in `email-validator`, and the Vercel bundle is already
    against a 250MB ceiling. The address is an account key here, never something we
    send mail to, so shape is all that has to hold.
    """
    email = (raw or "").strip().lower()
    local, _, domain = email.partition("@")
    if not local or not domain or "." not in domain or " " in email:
        raise ValueError("not a valid email address")
    return email

# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------


class User(BaseModel):
    """An account. `password_hash` is deliberately NOT on this model — it lives only in
    the storage layer, so there is no route that can accidentally serialise it."""

    user_id: UUID = Field(default_factory=uuid4)
    email: str
    created_at: datetime


class RegisterRequest(BaseModel):
    email: str
    # A floor, not a composition rule. Length is the only password requirement with
    # evidence behind it; the upper bound stops a multi-megabyte body from becoming
    # an argon2 denial-of-service.
    password: str = Field(min_length=10, max_length=1024)
    fund_name: Optional[str] = None

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        return normalize_email(v)


class LoginRequest(BaseModel):
    email: str
    password: str = Field(max_length=1024)

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        return normalize_email(v)


# ---------------------------------------------------------------------------
# Input 1: the survey — STATED preference
# ---------------------------------------------------------------------------

# Signal keys an option may emit. Constrained, because a free-form key would silently
# vanish from the derivation instead of failing loudly.
#   founder / market / idea_vs_market -> the three axes the core screen already uses
#   conviction   -1 = evidence-heavy, +1 = conviction-heavy
#   stage_lean   -1 = earliest stage,  +1 = later stage
AXIS_SIGNALS = ("founder", "market", "idea_vs_market")
SCALAR_SIGNALS = ("conviction", "stage_lean")
SIGNAL_KEYS = AXIS_SIGNALS + SCALAR_SIGNALS


class Choice(StrEnum):
    A = "a"
    B = "b"


class SurveyOption(BaseModel):
    text: str
    signals: dict[str, float] = Field(default_factory=dict)

    @field_validator("signals")
    @classmethod
    def _known_keys(cls, v: dict[str, float]) -> dict[str, float]:
        unknown = set(v) - set(SIGNAL_KEYS)
        if unknown:
            raise ValueError(f"unknown signal keys: {sorted(unknown)}")
        return v


class SurveyQuestion(BaseModel):
    """A forced trade-off. Not a Likert item, on purpose: agreement scales measure
    agreeableness, trade-offs measure priorities (§2.1)."""

    id: str
    prompt: str
    option_a: SurveyOption
    option_b: SurveyOption

    def option(self, choice: Choice) -> SurveyOption:
        return self.option_a if choice == Choice.A else self.option_b


class SurveyAnswer(BaseModel):
    question_id: str
    choice: Choice


class SurveySubmission(BaseModel):
    """Partial submissions are legal. An unanswered question contributes NOTHING and
    lowers the confidence — it is never imputed from the answers around it."""

    answers: list[SurveyAnswer] = Field(default_factory=list)


# THE SURVEY. Twelve forced trade-offs, each one a choice a working investor actually
# has to make. Every option carries the signals it implies; a question that implies
# nothing about a dimension emits nothing for it rather than a token weight.
SURVEY_QUESTIONS: list[SurveyQuestion] = [
    SurveyQuestion(
        id="q01_founder_vs_market",
        prompt=(
            "A technically exceptional founder in a crowded market, or a competent "
            "operator in an empty one."
        ),
        option_a=SurveyOption(text="Exceptional founder, crowded market", signals={"founder": 1.0}),
        option_b=SurveyOption(text="Competent operator, empty market", signals={"market": 1.0}),
    ),
    SurveyQuestion(
        id="q02_traction_vs_demo",
        prompt=(
            "Eighteen months of real but flat usage, or a six-week-old prototype that "
            "makes you feel something."
        ),
        option_a=SurveyOption(
            text="Eighteen months of flat real usage",
            signals={"conviction": -1.0, "stage_lean": 1.0, "market": 0.5},
        ),
        option_b=SurveyOption(
            text="Six-week-old prototype with a pull",
            signals={"conviction": 1.0, "stage_lean": -1.0, "idea_vs_market": 0.5},
        ),
    ),
    SurveyQuestion(
        id="q03_early_vs_ontime",
        prompt="The right product two years early, or the second-best product exactly on time.",
        option_a=SurveyOption(
            text="Right product, two years early",
            signals={"idea_vs_market": 1.0, "conviction": 0.5},
        ),
        option_b=SurveyOption(
            text="Second-best product, perfect timing",
            signals={"market": 1.0, "conviction": -0.3},
        ),
    ),
    SurveyQuestion(
        id="q04_velocity_vs_durability",
        prompt=(
            "A solo founder shipping every week, or a two-person team shipping monthly "
            "whose skills genuinely complement each other."
        ),
        option_a=SurveyOption(
            text="Solo founder, weekly shipping", signals={"founder": 1.0, "conviction": 0.5}
        ),
        option_b=SurveyOption(
            text="Complementary pair, monthly shipping",
            signals={"founder": 0.5, "conviction": -0.3},
        ),
    ),
    SurveyQuestion(
        id="q05_distribution_vs_product",
        prompt=(
            "An average product with a founder who can sell anything, or an extraordinary "
            "product built by a team that has never spoken to a customer."
        ),
        option_a=SurveyOption(
            text="Average product, exceptional seller", signals={"market": 1.0, "founder": 0.5}
        ),
        option_b=SurveyOption(
            text="Exceptional product, no customer contact", signals={"idea_vs_market": 1.0}
        ),
    ),
    SurveyQuestion(
        id="q06_which_mistake",
        prompt=(
            "You will be wrong either way. Would you rather miss a fund-returner, or "
            "write a cheque that goes to zero?"
        ),
        option_a=SurveyOption(
            text="Rather miss the fund-returner", signals={"conviction": -1.0, "stage_lean": 1.0}
        ),
        option_b=SurveyOption(
            text="Rather write the cheque that dies", signals={"conviction": 1.0, "stage_lean": -1.0}
        ),
    ),
    SurveyQuestion(
        id="q07_metrics_vs_love",
        prompt=(
            "Unverifiable numbers but customers who will not shut up about it, or clean "
            "audited metrics and customers who are merely satisfied."
        ),
        option_a=SurveyOption(
            text="Unverifiable numbers, fanatical customers",
            signals={"conviction": 1.0, "market": 0.5, "stage_lean": -0.5},
        ),
        option_b=SurveyOption(
            text="Clean metrics, indifferent customers",
            signals={"conviction": -1.0, "stage_lean": 0.5},
        ),
    ),
    SurveyQuestion(
        id="q08_incumbent_vs_demand",
        prompt=(
            "The main risk is that the incumbent ships it next quarter, or the main risk "
            "is that nobody turns out to want it."
        ),
        option_a=SurveyOption(
            text="Can live with incumbent risk", signals={"idea_vs_market": 1.0, "founder": 0.5}
        ),
        option_b=SurveyOption(text="Can live with demand risk", signals={"market": 1.0}),
    ),
    SurveyQuestion(
        id="q09_adaptability_vs_persistence",
        prompt=(
            "A founder who has changed the plan three times this year, or one who has held "
            "the same plan for three years."
        ),
        option_a=SurveyOption(
            text="Changed the plan three times", signals={"founder": 0.5, "conviction": 0.5}
        ),
        option_b=SurveyOption(
            text="Held the plan for three years", signals={"founder": 0.5, "conviction": -0.5}
        ),
    ),
    SurveyQuestion(
        id="q10_insider_vs_outsider",
        prompt=(
            "A domain expert who has lived the problem for a decade, or an outsider who "
            "finds the entire industry's assumptions absurd."
        ),
        option_a=SurveyOption(text="Ten-year domain insider", signals={"founder": 1.0}),
        option_b=SurveyOption(
            text="Outsider who rejects the premises",
            signals={"idea_vs_market": 1.0, "conviction": 0.5},
        ),
    ),
    SurveyQuestion(
        id="q11_speed_vs_diligence",
        prompt=(
            "Decide in one meeting and be occasionally reckless, or take six weeks of "
            "diligence and occasionally lose the deal."
        ),
        option_a=SurveyOption(
            text="One meeting, occasionally reckless",
            signals={"conviction": 1.0, "stage_lean": -0.5},
        ),
        option_b=SurveyOption(
            text="Six weeks, occasionally too slow",
            signals={"conviction": -1.0, "stage_lean": 0.5},
        ),
    ),
    SurveyQuestion(
        id="q12_price_vs_pick",
        prompt=(
            "Your highest-conviction company at a price you think is twice what it should "
            "be, or your third-favourite at a price you think is fair."
        ),
        option_a=SurveyOption(
            text="Best company, bad price", signals={"conviction": 1.0, "founder": 0.5}
        ),
        option_b=SurveyOption(text="Third-best company, fair price", signals={"conviction": -1.0}),
    ),
]

SURVEY_BY_ID = {q.id: q for q in SURVEY_QUESTIONS}


# ---------------------------------------------------------------------------
# Input 2: past decisions — REVEALED preference
# ---------------------------------------------------------------------------


class DecisionKind(StrEnum):
    INVESTED = "invested"
    PASSED = "passed"
    WATCHED = "watched"


class PastDecision(BaseModel):
    company: str
    sector: Optional[str] = None
    stage: Optional[str] = None
    decision: DecisionKind
    decided_on: Optional[date] = None
    rationale: Optional[str] = None
    outcome: Optional[str] = None
    source_row: Optional[int] = None  # 1-based row in the uploaded file: the provenance handle


class RejectedRow(BaseModel):
    """A row we could not read. Reported, never silently dropped — a parser that
    quietly discards a third of an upload produces a confident profile of nothing."""

    row_number: int
    reason: str
    raw: str


class DecisionUploadResult(BaseModel):
    """`accepted + len(rejected) == total_rows` always holds — every row is accounted for.

    `warnings` are rows that WERE accepted but lost an optional field on the way in (an
    unreadable date, say). They are reported rather than silently degraded, but they are
    not rejections: a row whose date we cannot read still has a perfectly legible
    verdict and sector, and throwing it away would discard real revealed preference to
    protect a field no derivation reads.
    """

    accepted: int
    rejected: list[RejectedRow] = Field(default_factory=list)
    warnings: list[RejectedRow] = Field(default_factory=list)
    total_rows: int


# ---------------------------------------------------------------------------
# What we derive — every value carries where it came from
# ---------------------------------------------------------------------------


class Provenance(BaseModel):
    """Which real submissions produced a derived value. `n` is the count of underlying
    observations, and it is what drives confidence: a weight from 4 decisions has to
    say it came from 4 decisions."""

    basis: str  # "survey" | "decisions" | "profile_field"
    method: str  # one line naming the computation, in plain words
    question_ids: list[str] = Field(default_factory=list)
    decision_rows: list[int] = Field(default_factory=list)
    n: int = 0


class AxisWeights(BaseModel):
    """Relative weight on the three axes. Normalised to sum to 1 across whatever the
    submission actually supported."""

    founder: float
    market: float
    idea_vs_market: float
    provenance: Provenance
    confidence: float = Field(ge=0.0, le=1.0)


class ConvictionStyle(BaseModel):
    """-1 evidence-heavy .. +1 conviction-heavy."""

    score: float = Field(ge=-1.0, le=1.0)
    label: str
    provenance: Provenance
    confidence: float = Field(ge=0.0, le=1.0)


class Prior(BaseModel):
    """Revealed concentration in one sector or stage. `share` is over INVESTED rows
    only — a pass is not a preference for the thing passed on."""

    key: str
    count: int
    share: float
    provenance: Provenance


class RedLine(BaseModel):
    """Something disqualifying regardless of score.

    `source` is always one of:
      stated             — the user typed it. Taken at face value, confidence 1.0.
      revealed_candidate — a unanimous pass pattern over enough rows to be worth
                           showing. A CANDIDATE, never an established rule: the user
                           has to confirm it. We do not get to invent a VC's red lines.
    """

    statement: str
    source: str
    provenance: Provenance
    confidence: float = Field(ge=0.0, le=1.0)


class NotInferred(BaseModel):
    """A field we deliberately did NOT derive, and why. This model is the whole point:
    the honest answer to thin data is an empty field with a reason attached."""

    field_name: str
    reason: str


class DerivedProfile(BaseModel):
    """Derived on READ from the two raw tables — never stored as a merged blob, so the
    stated and revealed sides can always be recomputed and compared independently."""

    axis_weights_stated: Optional[AxisWeights] = None
    axis_weights_revealed: Optional[AxisWeights] = None
    conviction_style_stated: Optional[ConvictionStyle] = None
    conviction_style_revealed: Optional[ConvictionStyle] = None
    sector_priors: list[Prior] = Field(default_factory=list)
    stage_priors: list[Prior] = Field(default_factory=list)
    red_lines: list[RedLine] = Field(default_factory=list)

    survey_answered: int = 0
    survey_total: int = len(SURVEY_QUESTIONS)
    decisions_count: int = 0
    invested_count: int = 0

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    personalisation_enabled: bool = False
    personalisation_reason: str = ""
    not_inferred: list[NotInferred] = Field(default_factory=list)


class VCProfile(BaseModel):
    profile_id: UUID
    user_id: UUID
    fund_name: Optional[str] = None
    focus_sectors: list[str] = Field(default_factory=list)
    stated_red_lines: list[str] = Field(default_factory=list)
    updated_at: datetime
    derived: DerivedProfile


class ProfileUpdate(BaseModel):
    fund_name: Optional[str] = None
    focus_sectors: Optional[list[str]] = None
    stated_red_lines: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# §3 — user-authored council lenses
#
# THE DISTINCTION THIS SECTION EXISTS TO HOLD. A derived lens is an INFERENCE: the
# system read it out of survey answers or uploaded decisions, and it must name the
# profile field that justified it. An authored lens is a STATEMENT: the VC typed it,
# and there is no profile field behind it because there was never an inference. Faking
# a profile field for an authored lens would merge the stated and the derived sides of
# the profile into one blob, which is the one thing §0 forbids — the stated-vs-revealed
# gap is only computable while the two remain separable.
#
# So `LensOrigin` is a first-class field, "derived" is NOT a value any client can send,
# and the storage layer's CHECK constraint makes an authored row that claims to be
# derived impossible rather than merely discouraged.
#
# AND: nothing here is ever created implicitly. There is no default council, no seeded
# lens and no auto-accepted template. `origin` has no default precisely so that a client
# must state whether the VC typed this or knowingly accepted a template — both are real
# input, and which one it was is a fact about the user we do not get to guess.
# ---------------------------------------------------------------------------


class LensOrigin(StrEnum):
    """How a council lens came to exist.

    DERIVED is produced only by `intelligence.custom_council.derive_lenses` from profile
    fields. It is deliberately unreachable from any request body: see `AuthoredLensWrite`.
    """

    DERIVED = "derived"
    #: The VC typed this one from scratch.
    AUTHORED = "authored"
    #: The VC knowingly accepted a template and (possibly) edited it. Still real input —
    #: they chose it — but a different basis from typing it, and the record says which.
    TEMPLATE = "template"


#: The origins a user is allowed to author. "derived" is the system's word.
AUTHORABLE_ORIGINS = (LensOrigin.AUTHORED, LensOrigin.TEMPLATE)

#: An authored weight of exactly 0 is not a lens, it is a deleted lens with extra steps.
#: The council refuses it rather than storing a seat that contributes nothing.
MIN_AUTHORED_WEIGHT = 0.01


def _nonblank(value: str, field: str) -> str:
    text = (value or "").strip()
    if not text:
        raise ValueError(f"{field} cannot be blank — an unnamed council agent is not authorable")
    return text


class AuthoredLensWrite(BaseModel):
    """A create/replace body for one authored council agent.

    `origin` is required and constrained to `AUTHORABLE_ORIGINS`, so no request body can
    mint a lens that claims the system derived it.
    """

    name: str = Field(max_length=120)
    #: The quality this agent adds score for. It is not decoration: the council reads
    #: this against the company's filtered evidence graph, so a lens whose quality
    #: carries no readable term would be a seat that measures nothing.
    quality: str = Field(max_length=120)
    persona: str = Field(max_length=4000)
    weight: float = Field(ge=MIN_AUTHORED_WEIGHT, le=1.0)
    origin: LensOrigin

    @field_validator("name", "quality", "persona")
    @classmethod
    def _text(cls, v: str, info) -> str:
        return _nonblank(v, info.field_name)

    @field_validator("origin")
    @classmethod
    def _authorable(cls, v: LensOrigin) -> LensOrigin:
        if v not in AUTHORABLE_ORIGINS:
            raise ValueError(
                "origin must be 'authored' or 'template'; 'derived' is reserved for lenses "
                "the system read out of your profile and cannot be claimed by a client"
            )
        return v


class AuthoredLensPatch(BaseModel):
    """A partial edit. An omitted field is left alone rather than cleared."""

    name: Optional[str] = Field(default=None, max_length=120)
    quality: Optional[str] = Field(default=None, max_length=120)
    persona: Optional[str] = Field(default=None, max_length=4000)
    weight: Optional[float] = Field(default=None, ge=MIN_AUTHORED_WEIGHT, le=1.0)
    origin: Optional[LensOrigin] = None

    @field_validator("name", "quality", "persona")
    @classmethod
    def _text(cls, v: Optional[str], info) -> Optional[str]:
        return None if v is None else _nonblank(v, info.field_name)

    @field_validator("origin")
    @classmethod
    def _authorable(cls, v: Optional[LensOrigin]) -> Optional[LensOrigin]:
        if v is not None and v not in AUTHORABLE_ORIGINS:
            raise ValueError("origin must be 'authored' or 'template'")
        return v


class AuthoredLens(BaseModel):
    """One stored council agent, owned by a profile.

    This is MUTABLE state, like the rest of the profile and for the same reason recorded
    in migration 002: the user edits it, and an edit is not an observation about the
    world, so it does not belong in the append-only event log.
    """

    lens_id: UUID
    profile_id: UUID
    name: str
    quality: str
    persona: str
    weight: float = Field(ge=MIN_AUTHORED_WEIGHT, le=1.0)
    origin: LensOrigin
    created_at: datetime
    updated_at: datetime

    @field_validator("origin")
    @classmethod
    def _never_derived(cls, v: LensOrigin) -> LensOrigin:
        if v == LensOrigin.DERIVED:
            raise ValueError(
                "a stored authored lens can never carry origin 'derived' — it would let a "
                "lens the VC typed masquerade as one read out of their answers"
            )
        return v


# ---------------------------------------------------------------------------
# §2.3 — the stated-vs-revealed gap
# ---------------------------------------------------------------------------


class GapFinding(BaseModel):
    """A divergence between what the VC said and what the VC did. Only emitted when
    BOTH sides have real data; the magnitude is the size of the disagreement."""

    dimension: str
    stated: str
    revealed: str
    finding: str
    magnitude: float = Field(ge=0.0, le=1.0)
    provenance: Provenance
    confidence: float = Field(ge=0.0, le=1.0)


class GapUncomputable(BaseModel):
    """A dimension we could not compare, naming WHICH side was missing. A gap analysis
    that silently omits the dimensions it lacked data for reads as agreement."""

    dimension: str
    missing: str
    reason: str


class GapReport(BaseModel):
    findings: list[GapFinding] = Field(default_factory=list)
    uncomputable: list[GapUncomputable] = Field(default_factory=list)
    agreements: list[str] = Field(default_factory=list)
    computed_at: datetime
