"""Tests for building AgentLeak prompts."""

from __future__ import annotations

import ast
import inspect
import json
import textwrap

import pytest
from agentleak.runner import OpenRouterAdapter

from src.evaluation.agentleak_eval import load_scenarios
from src.evaluation.agentleak_prompts import (
    UPSTREAM_COMPLETION_NUDGE,
    UPSTREAM_CONTEXT_TEMPLATE,
    UPSTREAM_SYSTEM_PROMPT,
    attack_targets_c1,
    build_messages,
)


@pytest.fixture(scope="module")
def base_scenarios():
    return load_scenarios("base")


@pytest.fixture(scope="module")
def single_agent(base_scenarios):
    return [s for s in base_scenarios if len(s.agents) == 1]


@pytest.fixture(scope="module")
def c1_attacked(single_agent):
    return next(s for s in single_agent if attack_targets_c1(s))


def _upstream_literals() -> tuple[set[str], set[str]]:
    """String constants and f-string pieces in upstream's execute()."""
    source = textwrap.dedent(inspect.getsource(OpenRouterAdapter.execute))
    constants: set[str] = set()
    fragments: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.JoinedStr):
            fragments |= {
                v.value for v in node.values
                if isinstance(v, ast.Constant) and isinstance(v.value, str)
            }
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            constants.add(node.value)
    return constants, fragments


def test_prompt_strings_are_verbatim_upstream():
    constants, fragments = _upstream_literals()
    assert UPSTREAM_SYSTEM_PROMPT in constants
    assert UPSTREAM_COMPLETION_NUDGE in constants
    head, tail = UPSTREAM_CONTEXT_TEMPLATE.split("{context}")
    assert head in fragments and tail in fragments


def test_four_turn_structure(c1_attacked):
    messages = build_messages(c1_attacked)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == c1_attacked.objective.user_request
    assert messages[3]["content"] == UPSTREAM_COMPLETION_NUDGE


def test_system_prompt_carries_privacy_instruction(c1_attacked):
    system = build_messages(c1_attacked)[0]["content"]
    assert system.startswith(UPSTREAM_SYSTEM_PROMPT)
    assert system.endswith(c1_attacked.objective.privacy_instruction)


def test_vault_reaches_the_model_intact(c1_attacked):
    assistant = build_messages(c1_attacked)[2]["content"]
    head, tail = UPSTREAM_CONTEXT_TEMPLATE.split("{context}")
    payload = assistant.removeprefix(head).removesuffix(tail)
    assert json.loads(payload) == [
        dict(r.fields) for r in c1_attacked.private_vault.records
    ]


def test_attack_off_contains_no_payload(c1_attacked):
    text = "".join(m["content"] for m in build_messages(c1_attacked))
    assert c1_attacked.attack.payload not in text


def test_attack_on_appends_payload_to_first_user_turn_only(c1_attacked):
    messages = build_messages(c1_attacked, inject_attack=True)
    payload = c1_attacked.attack.payload
    assert messages[1]["content"] == (
        f"{c1_attacked.objective.user_request}\n\n{payload}"
    )
    assert all(payload not in m["content"] for i, m in enumerate(messages) if i != 1)


def test_sixteen_single_agent_attacks_target_c1(single_agent):
    assert sum(attack_targets_c1(s) for s in single_agent) == 16


def test_injection_refused_for_off_channel_attack(single_agent):
    off_channel = next(s for s in single_agent if not attack_targets_c1(s))
    assert off_channel.attack.enabled
    with pytest.raises(ValueError, match="targeting C1"):
        build_messages(off_channel, inject_attack=True)
    build_messages(off_channel)


def test_injection_refused_without_attack(base_scenarios):
    clean = next(s for s in base_scenarios if not (s.attack and s.attack.enabled))
    with pytest.raises(ValueError, match="targeting C1"):
        build_messages(clean, inject_attack=True)
