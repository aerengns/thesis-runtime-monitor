"""Tests for turning streamed chunks into latched state sequences."""

from __future__ import annotations

from agentleak.core.channels import Channel

from src.monitor.abstraction import prefixes, state_sequence
from src.properties.privacy import PrivacyState, abstract, valid_tran

ON = PrivacyState(subject_present=True)
OFF = PrivacyState()


def scripted(*raw_states):
    """An abstraction returning the given states in order, ignoring the text."""
    it = iter(raw_states)
    return lambda prefix: next(it)


def test_prefixes_accumulate():
    assert list(prefixes(["I", " cannot", " help"])) == ["I", "I cannot", "I cannot help"]


def test_one_state_per_chunk_with_repeats_kept():
    seq = state_sequence(["a", "b", "c"], scripted(OFF, OFF, ON))
    assert seq.states == (OFF, OFF, ON)


def test_predicate_stays_on_after_raw_drop():
    seq = state_sequence(["a", "b", "c", "d"], scripted(OFF, ON, OFF, ON))
    assert seq.states == (OFF, ON, ON, ON)
    assert seq.raw_drops == 1


def test_no_chunks_gives_empty_sequence():
    assert state_sequence([], scripted()).states == ()


def test_real_ner_flicker_is_latched():
    """Chunks from a real qwen3 answer where spaCy dropped 'Patient X' mid-word."""
    chunks = ["I", " cannot", " summarize", " Patient", " X", "'s", " recent", " visit"]
    seq = state_sequence(chunks, lambda p: abstract(p, Channel.C1_FINAL_OUTPUT))

    for a, b in zip(seq.states, seq.states[1:]):
        assert valid_tran(a, b)
    assert seq.states[-1].subject_present
