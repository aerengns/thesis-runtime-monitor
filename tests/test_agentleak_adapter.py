"""Tests for the AgentLeak adapter: scenario loading and oracle configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.agentleak_eval import (
    CHANNEL_ALIASES,
    FAMILY_ALIASES,
    OLLAMA_OPENAI_BASE,
    load_scenarios,
    local_judge_config,
    normalise_scenario,
    oracle_config,
)
from src.properties.privacy import ScenarioContext, SensitiveCategory


@pytest.fixture(scope="module")
def base_scenarios():
    return load_scenarios("base")


# ---------------------------------------------------------------------------
# Loader / normalisation
# ---------------------------------------------------------------------------


def test_aliases_are_derived_not_hardcoded():
    """If upstream renames an enum member the mapping must follow, not break."""
    assert CHANNEL_ALIASES["final_output"] == "C1_final_output"
    assert CHANNEL_ALIASES["inter_agent"] == "C2_inter_agent"
    assert FAMILY_ALIASES["F1"] == "prompt_instruction"
    assert FAMILY_ALIASES["F2"] == "tool_surface"
    assert len(FAMILY_ALIASES) == 6


def test_normalise_does_not_mutate_input():
    """Vendored data on disk must stay byte-identical to upstream."""
    raw = {
        "attack": {"attack_family": "F2", "target_channels": ["inter_agent"]},
        "evaluation": {"channels_to_audit": ["final_output"]},
    }
    out = normalise_scenario(raw)
    assert raw["attack"]["attack_family"] == "F2", "input was mutated"
    assert out["attack"]["attack_family"] == "tool_surface"
    assert out["evaluation"]["channels_to_audit"] == ["C1_final_output"]


def test_all_splits_validate(base_scenarios):
    """The released data uses an older encoding than the shipped schema accepts."""
    assert len(base_scenarios) == 100
    assert len(load_scenarios("difficult")) == 100


def test_unknown_split_rejected():
    with pytest.raises(ValueError, match="unknown split"):
        load_scenarios("nonexistent")


def test_base_split_shape(base_scenarios):
    """Facts the first slice relies on: a clean subset and a single-agent subset."""
    verticals = {s.vertical for s in base_scenarios}
    assert len(verticals) == 4
    assert sum(1 for s in base_scenarios if not s.attack.enabled) == 36
    assert sum(1 for s in base_scenarios if len(s.agents) == 1) == 28


# ---------------------------------------------------------------------------
# ScenarioContext built from real scenarios
# ---------------------------------------------------------------------------


def test_context_from_real_scenario(base_scenarios):
    scenario = base_scenarios[0]
    ctx = ScenarioContext.from_scenario(scenario)

    assert "ssn" in scenario.allowed_set.forbidden_fields
    assert SensitiveCategory.SSN in ctx.forbidden_categories
    assert ctx.allowed_field_names, "allowed fields should be populated"
    # Canaries are forbidden by construction.
    for canary in scenario.evaluation.canaries:
        assert canary in ctx.forbidden_values


def test_context_never_forbids_an_allowed_field(base_scenarios):
    """Sanity check across the whole split, not just one scenario."""
    for scenario in base_scenarios:
        ctx = ScenarioContext.from_scenario(scenario)
        overlap = ctx.allowed_field_names & {
            f.lower() for f in scenario.allowed_set.forbidden_fields
        }
        assert not overlap, f"{scenario.scenario_id}: {overlap} both allowed and forbidden"


def test_context_extracts_vault_values(base_scenarios):
    """At least some scenarios must yield concrete forbidden values to match on."""
    with_values = sum(
        1 for s in base_scenarios if ScenarioContext.from_scenario(s).forbidden_values
    )
    assert with_values > 50, f"only {with_values}/100 scenarios yielded vault values"


# ---------------------------------------------------------------------------
# Oracle configuration
# ---------------------------------------------------------------------------


def test_presidio_is_always_enabled():
    """Without Presidio, tiers 1 and 2 do not run at all and the oracle returns zero."""
    assert oracle_config().enable_presidio is True
    assert oracle_config(enable_judge=False).enable_presidio is True


def test_local_judge_points_at_ollama():
    cfg = local_judge_config()
    assert cfg.openrouter_base_url == OLLAMA_OPENAI_BASE
    assert cfg.openrouter_base_url.endswith("/"), "judge appends 'chat/completions'"
    assert cfg.temperature == 0.0
    assert cfg.use_cache is True


def test_judge_can_be_disabled_for_offline_runs():
    assert oracle_config(enable_judge=False).enable_llm_judge is False


def test_judge_prompts_are_wired():
    """The installed package omits these; without them Tier 3 detects almost nothing."""
    cfg = local_judge_config()
    assert cfg.system_prompt_path and Path(cfg.system_prompt_path).is_file()
    assert cfg.few_shot_prompt_path and Path(cfg.few_shot_prompt_path).is_file()
    assert Path(cfg.system_prompt_path).read_text().strip(), "system prompt is empty"


def test_missing_judge_prompt_fails_loudly(monkeypatch, tmp_path):
    """A silent fallback to no system prompt would quietly invalidate every label."""
    import src.evaluation.agentleak_eval as mod

    monkeypatch.setattr(mod, "JUDGE_PROMPT_DIR", tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="judge prompt missing"):
        mod.local_judge_config()
