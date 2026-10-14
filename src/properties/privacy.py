"""Privacy property: prefix-level state abstraction for Layer 1.

This module holds *only* the abstraction the five Boolean predicates that
ProbGuard's Algorithm 1 derives from the CTL property, evaluated on an output
prefix on every streamed chunk. It must stay cheap: regex plus one small spaCy
pass, no LLM call and no embedding model.

Ground-truth labelling is deliberately **not** here. AgentLeak ships its own
three-tier detection pipeline and metrics, and we use those via
``src.evaluation.agentleak_eval``. Implementing our own oracle would mean
grading our own homework and would make our numbers incomparable to the
published results.

References
----------
ProbGuard (Wang et al. 2026), ``papers/ProbGuard.pdf`` Sec. 3.2 -- predicate
abstraction, semantic validity pruning, and the ``valid_tran`` relation required
by the Laplace-smoothed transition estimator.

AgentLeak (El Yagoubi et al. 2026), ``papers/AgentLeak.pdf`` Sec. III-B -- the
seven leakage channels; the scenario ``allowed_set`` supplies the per-scenario
authorisation model that :class:`ScenarioContext` reads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from itertools import product
from typing import Any, Iterable, Sequence

# AgentLeak's channel enum is reused rather than duplicated: the trace events our
# harness emits are typed with it, so a parallel enum would only invite drift.
from agentleak.core.channels import Channel

# ---------------------------------------------------------------------------
# CTL property
# ---------------------------------------------------------------------------

#: The monitored safety invariant, in ProbGuard's standard AG-not-unsafe form.
#: The atomic propositions here are exactly the predicates computed below;
#: that correspondence is Algorithm 1, line 2, and must be preserved.
PROPERTY_CTL = "AG !(binding_established & boundary_crossing)"

PREDICATE_NAMES: tuple[str, ...] = (
    "subject_present",
    "identifier_present",
    "sensitive_category_present",
    "binding_established",
    "boundary_crossing",
)

#: Channels this thesis treats as crossing a trust boundary. The project brief scopes
#: evaluation to C1 and C2, where AgentLeak reports the sharpest contrast
#: (27.2% vs 68.8% leakage).
BOUNDARY_CROSSING_CHANNELS: frozenset[Channel] = frozenset(
    {Channel.C1_FINAL_OUTPUT, Channel.C2_INTER_AGENT}
)


class SensitiveCategory(Enum):
    """Categories the cheap prefix detectors can recognise."""

    SSN = "ssn"
    MEDICAL = "medical"
    CREDIT_CARD = "credit_card"
    SALARY = "salary"
    FINANCIAL = "financial"
    EMAIL = "email"
    PHONE = "phone"
    RECORD_NUMBER = "record_number"


#: Maps a scenario's ``forbidden_fields`` names onto the categories our detectors
#: can see. Matching is by substring, so ``current_salary`` and ``expected_salary``
#: both reach SALARY. Field names we cannot categorise still contribute through
#: verbatim value matching, so an unmapped name is a loss of sensitivity, not of
#: soundness.
_FIELD_NAME_CATEGORIES: dict[SensitiveCategory, tuple[str, ...]] = {
    SensitiveCategory.SSN: ("ssn", "social_security"),
    SensitiveCategory.MEDICAL: (
        "diagnosis", "medical_history", "medication", "allergies",
        "psychiatric", "hiv", "treatment", "prognosis",
    ),
    SensitiveCategory.CREDIT_CARD: ("card", "credit_card"),
    SensitiveCategory.FINANCIAL: (
        "account_number", "routing_number", "balance", "credit_score",
        "fraud_score", "settlement_amount", "retainer_amount", "financial",
    ),
    SensitiveCategory.SALARY: ("salary", "salaries", "income", "compensation"),
    SensitiveCategory.EMAIL: ("email",),
    SensitiveCategory.PHONE: ("phone",),
    SensitiveCategory.RECORD_NUMBER: ("insurance_id", "patient_id", "note_id", "mrn"),
}


def categories_for_field(field_name: str) -> set[SensitiveCategory]:
    """Which detector categories a scenario field name corresponds to."""
    lowered = field_name.lower()
    return {
        cat
        for cat, needles in _FIELD_NAME_CATEGORIES.items()
        if any(n in lowered for n in needles)
    }


# ---------------------------------------------------------------------------
# Structured-field patterns
# ---------------------------------------------------------------------------

# SSN in three-two-four form with valid area numbers. The (?<![\w-]) guard is
# deliberate: AgentLeak reports false positives where an internal identifier
# such as "TIC-123-45-6789" trips a naive SSN pattern.
_SSN = re.compile(r"(?<![\w-])(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?![\w-])")

_CARD_CANDIDATE = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")
_EMAIL = re.compile(r"(?<![\w.])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.])")
_PHONE = re.compile(r"(?<![\w-])(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?![\w-])")
_RECORD_NO = re.compile(
    r"(?<![\w-])(?:MRN|PT|NOTE|INS)[-: ]?\d{4,10}(?![\w-])", re.IGNORECASE
)
_ICD10 = re.compile(r"(?<![\w.])[A-TV-Z]\d{2}(?:\.\d{1,4})?(?![\w.])")


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


#: Prose markers for sensitive categories. This is the cheap directional signal
#: for the abstraction only. It is knowingly incomplete "her A1C has been
#: climbing" will be missed because paraphrase detection is the oracle's job
#: (AgentLeak's Tier 3), and the oracle runs offline where cost does not matter.
_CATEGORY_LEXICON: dict[SensitiveCategory, tuple[str, ...]] = {
    SensitiveCategory.MEDICAL: (
        "diagnosis", "diagnosed", "prognosis", "treated for", "medication",
        "prescribed", "symptoms", "cancer", "diabetes", "hiv", "depression",
        "psychiatric", "chemotherapy", "allergic to",
    ),
    SensitiveCategory.SALARY: (
        "salary", "compensation", "annual pay", "earns", "wage", "bonus",
    ),
    SensitiveCategory.FINANCIAL: (
        "account balance", "account number", "routing number", "net worth",
        "credit score", "debt", "mortgage", "loan amount", "settlement",
    ),
}


# ---------------------------------------------------------------------------
# Scenario context: the authorisation model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScenarioContext:
    """Per-scenario authorisation, read from an AgentLeak scenario.

    Without this, ``binding_established`` would fire on *any* subject-attribute
    linkage. But AgentLeak scenarios state explicitly which fields may be
    disclosed and which may not a patient's name is typically allowed while
    their SSN is forbidden so flagging a permitted disclosure is a false
    positive, not a leak. Authorisation therefore comes from the benchmark
    rather than from our own judgement.
    """

    forbidden_values: tuple[str, ...] = ()
    forbidden_categories: frozenset[SensitiveCategory] = frozenset()
    allowed_field_names: frozenset[str] = frozenset()

    @classmethod
    def from_scenario(cls, scenario: Any) -> "ScenarioContext":
        """Build from an ``agentleak.schemas.scenario.Scenario``."""
        forbidden_names = [f.lower() for f in scenario.allowed_set.forbidden_fields]
        allowed_names = {f.lower() for f in scenario.allowed_set.fields}

        categories: set[SensitiveCategory] = set()
        for name in forbidden_names:
            categories |= categories_for_field(name)

        values: list[str] = []
        for record in scenario.private_vault.records:
            fields = getattr(record, "fields", None) or {}
            if not isinstance(fields, dict):
                continue
            for field_name, value in fields.items():
                if field_name.lower() not in forbidden_names:
                    continue
                values.extend(_flatten_value(value))

        # Canaries are forbidden by construction.
        values.extend(str(c) for c in getattr(scenario.evaluation, "canaries", ()) or ())

        return cls(
            forbidden_values=tuple(v for v in dict.fromkeys(values) if len(v) >= 4),
            forbidden_categories=frozenset(categories),
            allowed_field_names=frozenset(allowed_names),
        )


def _flatten_value(value: Any) -> Iterable[str]:
    """Vault values may be scalars or lists (e.g. ``medications``)."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if v is not None]
    return [str(value)]


#: Fallback used when no scenario is available (ad-hoc monitoring, unit tests).
#: It has no authorisation model, so it treats every sensitive linkage as
#: forbidden deliberately over-sensitive rather than unsound.
UNRESTRICTED = ScenarioContext(forbidden_categories=frozenset(SensitiveCategory))


# ---------------------------------------------------------------------------
# Abstract state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PrivacyState:
    """A valuation of the five privacy predicates, one abstract DTMC state."""

    subject_present: bool = False
    identifier_present: bool = False
    sensitive_category_present: bool = False
    binding_established: bool = False
    boundary_crossing: bool = False

    def to_bits(self) -> tuple[int, ...]:
        """``s_P(v)`` in ProbGuard's notation: the predicate valuation vector."""
        return tuple(int(getattr(self, name)) for name in PREDICATE_NAMES)

    @property
    def is_unsafe(self) -> bool:
        """Negation of the invariant: a forbidden binding on a boundary channel."""
        return self.binding_established and self.boundary_crossing

    def is_semantically_valid(self) -> bool:
        """Prune infeasible predicate combinations (ProbGuard Sec. 3.2.2).

        A binding cannot exist without both a subject and something sensitive to
        bind it to; admitting such states would place probability mass on regions
        the system can never occupy.
        """
        if self.binding_established:
            return self.subject_present and (
                self.identifier_present or self.sensitive_category_present
            )
        return True

    def __le__(self, other: "PrivacyState") -> bool:
        return all(a <= b for a, b in zip(self.to_bits(), other.to_bits()))

    def __or__(self, other: "PrivacyState") -> "PrivacyState":
        return PrivacyState(
            **{n: getattr(self, n) or getattr(other, n) for n in PREDICATE_NAMES}
        )


def valid_tran(source: PrivacyState, target: PrivacyState) -> bool:
    """ProbGuard's ``valid_tran`` relation, derived from predicate monotonicity.

    Text accumulates: once a PERSON entity appears in a prefix it never
    disappears, so every text-derived predicate is monotone non-decreasing within
    a single generation, and the channel is fixed for that generation. Together
    these make the transition relation a bitwise order, which prunes most of the
    |S|x|S| matrix and so keeps the PAC sample bound small.

    This is a genuine departure from ProbGuard's setting, where world state can
    revert (a microwave door re-closes) and no such order exists.
    """
    if source.boundary_crossing != target.boundary_crossing:
        return False
    if not (source.is_semantically_valid() and target.is_semantically_valid()):
        return False
    return source <= target


@lru_cache(maxsize=None)
def enumerate_valid_states() -> tuple[PrivacyState, ...]:
    """The pruned abstract state space S. Cached: a fixed, small set."""
    return tuple(
        s
        for s in (
            PrivacyState(*bits)
            for bits in product((False, True), repeat=len(PREDICATE_NAMES))
        )
        if s.is_semantically_valid()
    )


# ---------------------------------------------------------------------------
# Abstraction function
# ---------------------------------------------------------------------------

_NLP = None


def _nlp():
    """Lazily load spaCy; model load is ~0.5s and must not happen at import."""
    global _NLP
    if _NLP is None:
        import spacy

        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def _find_identifiers(text: str) -> list[tuple[SensitiveCategory, str]]:
    found: list[tuple[SensitiveCategory, str]] = []
    for m in _SSN.finditer(text):
        found.append((SensitiveCategory.SSN, m.group()))
    for m in _CARD_CANDIDATE.finditer(text):
        digits = re.sub(r"[ -]", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            found.append((SensitiveCategory.CREDIT_CARD, m.group()))
    for m in _EMAIL.finditer(text):
        found.append((SensitiveCategory.EMAIL, m.group()))
    for m in _PHONE.finditer(text):
        found.append((SensitiveCategory.PHONE, m.group()))
    for m in _RECORD_NO.finditer(text):
        found.append((SensitiveCategory.RECORD_NUMBER, m.group()))
    for m in _ICD10.finditer(text):
        found.append((SensitiveCategory.MEDICAL, m.group()))
    return found


def _find_categories(text: str) -> list[tuple[SensitiveCategory, str]]:
    lowered = text.lower()
    return [
        (cat, term)
        for cat, terms in _CATEGORY_LEXICON.items()
        for term in terms
        if term in lowered
    ]


def _subjects(text: str) -> list[str]:
    return [e.text for e in _nlp()(text).ents if e.label_ in ("PERSON", "ORG")]


def _co_occurs(text: str, subjects: Sequence[str], attributes: Sequence[str]) -> bool:
    """A leak is the *linkage*, not the mention.

    "Diabetes is a chronic condition" discloses nothing about anyone. "Sarah Chen
    is being treated for diabetes" does. Linkage is approximated by sentence-level
    co-occurrence, which is cheap and safe to evaluate on a prefix.
    """
    if not subjects or not attributes:
        return False
    lowered = [a.lower() for a in attributes]
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
        low = sentence.lower()
        if any(s.lower() in low for s in subjects) and any(a in low for a in lowered):
            return True
    return False


def abstract(
    prefix: str,
    channel: Channel,
    context: ScenarioContext = UNRESTRICTED,
) -> PrivacyState:
    """``s_P(v)``: map an output prefix, its channel and scenario to an abstract state.

    ``identifier_present`` and ``sensitive_category_present`` record what the
    prefix contains at all; they are the precursors Layer 1 warns on. Only
    ``binding_established`` consults the authorisation model, so a disclosure the
    scenario permits raises risk without ever becoming a violation.
    """
    subjects = _subjects(prefix)
    identifiers = _find_identifiers(prefix)
    categories = _find_categories(prefix)

    forbidden_spans: list[str] = [
        value for value in context.forbidden_values if value in prefix
    ]
    forbidden_spans += [
        span for cat, span in identifiers if cat in context.forbidden_categories
    ]
    forbidden_spans += [
        term for cat, term in categories if cat in context.forbidden_categories
    ]

    return PrivacyState(
        subject_present=bool(subjects),
        identifier_present=bool(identifiers),
        sensitive_category_present=bool(categories),
        binding_established=_co_occurs(prefix, subjects, forbidden_spans),
        boundary_crossing=channel in BOUNDARY_CROSSING_CHANNELS,
    )
