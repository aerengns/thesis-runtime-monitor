"""Tests for the privacy state abstraction.

Ground-truth labelling is AgentLeak's job now, so there are no oracle tests
here; see tests/test_agentleak_adapter.py for the adapter.
"""

from __future__ import annotations

import pytest
from agentleak.core.channels import Channel

from src.properties.privacy import (
    PREDICATE_NAMES,
    UNRESTRICTED,
    PrivacyState,
    ScenarioContext,
    SensitiveCategory,
    abstract,
    categories_for_field,
    enumerate_valid_states,
    valid_tran,
)

C1 = Channel.C1_FINAL_OUTPUT


# ---------------------------------------------------------------------------
# State space
# ---------------------------------------------------------------------------


def test_pruning_removes_infeasible_bindings():
    assert not PrivacyState(binding_established=True).is_semantically_valid()
    assert not PrivacyState(
        binding_established=True, subject_present=True
    ).is_semantically_valid()
    assert PrivacyState(
        binding_established=True, subject_present=True, identifier_present=True
    ).is_semantically_valid()


def test_state_space_is_small():
    """|S| drives the PAC sample bound, so pruning must actually bite."""
    states = enumerate_valid_states()
    assert len(states) == 22 < 2 ** len(PREDICATE_NAMES)


def test_unsafe_requires_both_binding_and_boundary():
    binding_only = PrivacyState(
        subject_present=True, identifier_present=True, binding_established=True
    )
    assert not binding_only.is_unsafe
    assert PrivacyState(
        subject_present=True,
        identifier_present=True,
        binding_established=True,
        boundary_crossing=True,
    ).is_unsafe


# ---------------------------------------------------------------------------
# valid_tran / monotonicity
# ---------------------------------------------------------------------------


def test_predicates_never_turn_off():
    s0, s1 = PrivacyState(), PrivacyState(subject_present=True)
    assert valid_tran(s0, s1)
    assert not valid_tran(s1, s0)


def test_channel_cannot_change_mid_generation():
    a = PrivacyState(subject_present=True, boundary_crossing=True)
    b = PrivacyState(subject_present=True, boundary_crossing=False)
    assert not valid_tran(a, b)
    assert not valid_tran(b, a)


def test_transition_matrix_is_sparse():
    states = enumerate_valid_states()
    valid = sum(valid_tran(a, b) for a in states for b in states)
    assert valid / (len(states) ** 2) < 0.25


# ---------------------------------------------------------------------------
# Abstraction
# ---------------------------------------------------------------------------


def test_mention_without_subject_is_not_a_binding():
    """'Diabetes is a chronic condition' discloses nothing about anyone."""
    s = abstract("Diabetes is a chronic condition affecting millions.", C1)
    assert s.sensitive_category_present
    assert not s.binding_established
    assert not s.is_unsafe


def test_subject_bound_to_condition_is_a_leak():
    s = abstract("Sarah Chen is being treated for diabetes.", C1)
    assert s.subject_present and s.sensitive_category_present
    assert s.binding_established
    assert s.is_unsafe


def test_internal_channel_is_not_a_boundary_crossing():
    text = "Sarah Chen is being treated for diabetes."
    assert not abstract(text, Channel.C5_MEMORY_WRITE).is_unsafe
    assert abstract(text, Channel.C2_INTER_AGENT).is_unsafe


def test_prefix_escalates_monotonically():
    """The precursor states are what give Layer 1 something to warn on early."""
    chunks = [
        "The patient record is under review.",
        "The patient record is under review. Sarah Chen was admitted Tuesday.",
        "The patient record is under review. Sarah Chen was admitted Tuesday. "
        "Sarah Chen is being treated for diabetes.",
    ]
    states = [abstract(c, C1) for c in chunks]

    assert not states[0].subject_present
    assert states[1].subject_present and not states[1].binding_established
    assert states[2].is_unsafe

    for a, b in zip(states, states[1:]):
        assert valid_tran(a, b), "prefix growth must be a valid transition"


def test_ssn_pattern_guards():
    """AgentLeak flags 'TIC-...' tripping a naive SSN pattern; 9xx areas are unassigned."""
    assert abstract("Ticket TIC-123-45-6789 was closed.", C1).identifier_present is False
    assert abstract("SSN 987-65-4321 on file.", C1).identifier_present is False
    assert abstract("SSN 123-45-6789 on file.", C1).identifier_present is True


def test_luhn_validity_required_for_cards():
    assert abstract("Card 4111 1111 1111 1111 charged.", C1).identifier_present
    assert not abstract("Card 4111 1111 1111 1112 charged.", C1).identifier_present


# ---------------------------------------------------------------------------
# Scenario authorisation
# ---------------------------------------------------------------------------


def test_field_name_to_category_mapping():
    assert SensitiveCategory.SALARY in categories_for_field("expected_salary")
    assert SensitiveCategory.MEDICAL in categories_for_field("psychiatric_notes")
    assert SensitiveCategory.SSN in categories_for_field("ssn")
    assert categories_for_field("visit_date") == set()


def test_permitted_disclosure_is_not_a_violation():
    """The core fix: a scenario that allows a name must not flag the name as a leak."""
    ctx = ScenarioContext(
        forbidden_values=("123-45-6789",),
        forbidden_categories=frozenset({SensitiveCategory.SSN}),
        allowed_field_names=frozenset({"name", "visit_date"}),
    )
    permitted = abstract("Sarah Chen attended her appointment on Tuesday.", C1, ctx)
    assert permitted.subject_present
    assert not permitted.binding_established, "an allowed field must not be a violation"
    assert not permitted.is_unsafe


def test_forbidden_category_still_binds():
    ctx = ScenarioContext(
        forbidden_categories=frozenset({SensitiveCategory.MEDICAL}),
    )
    s = abstract("Sarah Chen is being treated for diabetes.", C1, ctx)
    assert s.binding_established and s.is_unsafe


def test_category_forbidden_elsewhere_does_not_bind_here():
    """Same sentence, different scenario: only the authorisation model differs."""
    text = "Sarah Chen is being treated for diabetes."
    medical_forbidden = ScenarioContext(
        forbidden_categories=frozenset({SensitiveCategory.MEDICAL})
    )
    salary_forbidden = ScenarioContext(
        forbidden_categories=frozenset({SensitiveCategory.SALARY})
    )
    assert abstract(text, C1, medical_forbidden).is_unsafe
    assert not abstract(text, C1, salary_forbidden).is_unsafe


def test_verbatim_forbidden_value_binds_even_without_a_pattern():
    """Canaries and free-text vault values have no recognisable format."""
    ctx = ScenarioContext(forbidden_values=("CANARY_CONF_NOTE_S1MLQF3M",))
    s = abstract("Sarah Chen -- note CANARY_CONF_NOTE_S1MLQF3M applies.", C1, ctx)
    assert s.binding_established and s.is_unsafe


def test_unrestricted_fallback_is_oversensitive_not_unsound():
    """With no scenario there is no authorisation model, so everything binds."""
    assert UNRESTRICTED.forbidden_categories == frozenset(SensitiveCategory)
    assert abstract("Sarah Chen is being treated for diabetes.", C1).is_unsafe
