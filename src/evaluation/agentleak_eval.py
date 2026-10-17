"""AgentLeak benchmark adapter: scenario loading and ground-truth labelling.

The benchmark is used as the *oracle* for the privacy property. We deliberately do
not implement our own leakage detector for evaluation purposes -- using the
benchmark's own three-tier pipeline is what makes our numbers comparable to the
published results, and avoids grading our own homework.

Two upstream quirks are worked around here, each documented on the function that
handles it: the released datasets do not validate against the shipped schema
(:func:`normalise_scenario`), and the detection pipeline needs a dependency the
package does not declare (:func:`oracle_config`).
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from agentleak.schemas.scenario import AttackFamily, Channel, Scenario
from agentleak.schemas.trace import EventType, ExecutionTrace, TraceEvent, TraceMetadata

#: Repository-relative location of the vendored datasets. See data/agentleak/PROVENANCE.md.
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "agentleak"

DATASETS = {
    "base": DATA_DIR / "scenarios_base_100.jsonl",
    "difficult": DATA_DIR / "scenarios_difficult_100.jsonl",
    "full": DATA_DIR / "scenarios_full_1000.jsonl",
}


def _channel_aliases() -> dict[str, str]:
    """``final_output`` -> ``C1_final_output``, derived from the enum values."""
    aliases: dict[str, str] = {}
    for member in Channel:
        value = member.value                      # e.g. "C1_final_output"
        short = value.split("_", 1)[1]            # e.g. "final_output"
        aliases[short] = value
    return aliases


def _family_aliases() -> dict[str, str]:
    """``F2`` -> ``tool_surface``, derived from the enum member names."""
    return {
        member.name.split("_", 1)[0]: member.value   # "F2_TOOL" -> ("F2", "tool_surface")
        for member in AttackFamily
    }


CHANNEL_ALIASES = _channel_aliases()
FAMILY_ALIASES = _family_aliases()


def _fix_channel(value: Any) -> Any:
    return CHANNEL_ALIASES.get(value, value) if isinstance(value, str) else value


def normalise_scenario(raw: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a released scenario record into the shipped schema's encoding.

    The released datasets are written in an older encoding than the pydantic
    models accept, and the package ships no loader reconciling the two
    (``utils.load_jsonl`` just returns raw dicts), so every scenario fails
    validation without this step:

    ====================  ==========================  =====================
    Field                 Released data               Schema expects
    ====================  ==========================  =====================
    channels              ``final_output``            ``C1_final_output``
    attack families       ``F2``                      ``tool_surface``
    ====================  ==========================  =====================

    Verified against pinned commit ``1301eb9``; see requirements.txt for why
    the version is pinned. Two properties keep this safe to leave in place:

    * **Forward-compatible.** The alias keys are only the short forms, so a
      record already in the new encoding passes through untouched. If upstream
      fixes the datasets this quietly becomes a no-op rather than mangling
      correct data, and ``f(f(x)) == f(x)``.
    * **Rename-resistant.** The alias tables are derived from the enum member
      names rather than hardcoded, so an upstream rename follows automatically
      instead of silently breaking.

    Returns a new dict; the input is not mutated, so the vendored data on disk
    stays byte-identical to upstream.
    """
    out = json.loads(json.dumps(raw))  # cheap deep copy of plain JSON

    attack = out.get("attack") or {}
    family = attack.get("attack_family")
    if isinstance(family, str) and family in FAMILY_ALIASES:
        attack["attack_family"] = FAMILY_ALIASES[family]
    if isinstance(attack.get("target_channels"), list):
        attack["target_channels"] = [_fix_channel(c) for c in attack["target_channels"]]

    evaluation = out.get("evaluation") or {}
    if isinstance(evaluation.get("channels_to_audit"), list):
        evaluation["channels_to_audit"] = [
            _fix_channel(c) for c in evaluation["channels_to_audit"]
        ]

    return out


def iter_scenarios(path: str | Path) -> Iterator[Scenario]:
    """Stream validated :class:`Scenario` objects from a released JSONL file."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Scenario.model_validate(normalise_scenario(json.loads(line)))


def load_scenarios(split: str = "base") -> list[Scenario]:
    """Load one of the vendored splits: ``base``, ``difficult`` or ``full``."""
    try:
        path = DATASETS[split]
    except KeyError:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(DATASETS)}") from None
    return list(iter_scenarios(path))


# ---------------------------------------------------------------------------
# Oracle configuration
# ---------------------------------------------------------------------------

#: Ollama's OpenAI-compatible endpoint. Trailing slash matters: the judge builds
#: its URL as f"{openrouter_base_url}chat/completions".
OLLAMA_OPENAI_BASE = "http://127.0.0.1:11434/v1/"

#: Ollama ignores bearer tokens, but the judge refuses to call without one.
_DUMMY_KEY = "ollama-local"

#: Where we keep our own copies of the two prompt files the judge needs.
#:
#: Those files tell the judge what counts as a leak. They exist in AgentLeak's
#: GitHub repository but are missing from the package you get with pip install,
#: and the judge does not complain when they are absent it just runs with no
#: instructions and finds almost nothing. That would make every label we
#: generate wrong, so we keep copies here. See data/agentleak/PROVENANCE.md.
JUDGE_PROMPT_DIR = DATA_DIR / "prompts"


def local_judge_config(model: str | None = None) -> "LLMJudgeConfig":  # noqa: F821
    """Tier 3 judge served by local Ollama instead of a paid API.

    ``openrouter_base_url`` points at localhost deliberately: the judge posts an
    OpenAI-compatible request to whatever that URL holds, and sends any model
    name it does not recognise down that path ``qwen3:4b-instruct`` included.

    Qwen is the default on evidence rather than convenience. AgentLeak measured a
    Qwen-2.5-7B judge at 4.8% FPR / 7.4% FNR against human labels, beside
    GPT-4o-mini's 4.1% / 6.9% (Sec. VI-B). Tier 3 decides the ~82% of leaks that
    are paraphrased, so a weak judge here would quietly corrupt every label.

    ``use_cache`` is left on, but do not rely on it between runs: despite
    ``_setup_cache`` being documented as a "disk-based cache", it only creates an
    empty directory and the cache itself is an in-memory dict that dies with the
    process. Identical judgements are deduplicated *within* one run only. At
    ~40 s per judged output, anything that labels a corpus needs to persist its
    own results, or a crash loses the whole run.
    """
    import os

    from agentleak.detection.llm_judge import LLMJudgeConfig

    os.environ.setdefault("OPENROUTER_API_KEY", _DUMMY_KEY)

    system_prompt = JUDGE_PROMPT_DIR / "llm_judge_system.md"
    few_shot = JUDGE_PROMPT_DIR / "llm_judge_few_shot.md"
    for path in (system_prompt, few_shot):
        if not path.exists():
            raise FileNotFoundError(
                f"judge prompt missing: {path}. These are vendored from the pinned "
                "commit because the installed package omits them; without them the "
                "judge runs with no system prompt and detects almost nothing."
            )

    return LLMJudgeConfig(
        model=model or os.getenv("JUDGE_MODEL_LOCAL", "qwen3:4b-instruct"),
        openrouter_base_url=OLLAMA_OPENAI_BASE,
        temperature=0.0,
        use_cache=True,
        cache_dir=str(DATA_DIR.parent / ".llm_judge_cache"),
        system_prompt_path=str(system_prompt),
        few_shot_prompt_path=str(few_shot),
    )


def oracle_config(use_local_judge: bool = True, enable_judge: bool = True):
    """A :class:`HybridConfig` for ground-truth labelling.

    **Presidio is always enabled, and must be installed.** The paper describes
    three detection tiers, but ``HybridPipeline`` has only two code paths:
    Presidio covers Tiers 1 and 2, and the LLM judge is Tier 3. There is no
    standalone Tier 1, so with Presidio off the oracle reports no leak even for
    a *canary* -- a unique marker string planted in the scenario vault (e.g.
    ``CANARY_CONF_NOTE_S1MLQF3M``) that appears nowhere else, and whose presence
    in an output is therefore proof of disclosure rather than a judgement call.

    Presidio is an optional dependency upstream (``# Presidio est optionnel``)
    and absent from its requirements.txt, yet ``enable_presidio`` defaults to
    True and the detector returns an empty list when the import failed, logging
    only a warning. Installing ``agentleak`` alone therefore yields a leak
    detector that silently finds nothing. We declare presidio-analyzer in our
    own requirements.txt, and `make check` verifies it independently.
    """
    from agentleak.detection.hybrid_pipeline import HybridConfig

    kwargs: dict[str, Any] = {
        "enable_presidio": True,
        "enable_llm_judge": enable_judge,
    }
    if enable_judge and use_local_judge:
        kwargs["llm_judge"] = local_judge_config()
    return HybridConfig(**kwargs)


def build_oracle(use_local_judge: bool = True, enable_judge: bool = True):
    """The AgentLeak three-tier pipeline, configured as this thesis uses it.

    Returns an object exposing ``detect(scenario, trace) -> HybridResult``, whose
    ``has_leakage`` is the binary ground-truth label the DTMC is learned against.

    Note that ``HybridResult.elr`` is *not* the paper's binary ELR: the
    implementation returns a fraction of vault fields leaked. Use ``has_leakage``
    for the binary notion and say which one you mean when reporting.
    """
    from agentleak.detection.hybrid_pipeline import HybridPipeline

    from src.evaluation.agentleak_judge import SchemaJudge

    config = oracle_config(use_local_judge, enable_judge)
    pipeline = HybridPipeline(config)
    if enable_judge and use_local_judge:
        pipeline._llm_judge = SchemaJudge(config.llm_judge)
    return pipeline


# ---------------------------------------------------------------------------
# Labelling final outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleLabel:
    leaked: bool
    presidio_fields: tuple[str, ...]
    judge_fields: tuple[str, ...]
    judge_verdict: bool | None = None


def final_output_trace(scenario_id: str, text: str, model: str) -> ExecutionTrace:
    """A trace holding only the final output (C1).

    The vault lookup is left out: the oracle scans every event, so including
    it would mark every run as a leak.
    """
    return ExecutionTrace(
        metadata=TraceMetadata(
            trace_id=f"{scenario_id}:{model}",
            scenario_id=scenario_id,
            framework="none",
            model=model,
            started_at=dt.datetime.now(),
        ),
        events=[
            TraceEvent(
                sequence_num=0,
                scenario_id=scenario_id,
                agent_id="A1",
                event_type=EventType.MESSAGE_OUT,
                channel=Channel.C1_FINAL_OUTPUT,
                content=text,
            )
        ],
    )


def _leaked_fields(tier: Any) -> tuple[str, ...]:
    return tuple(leak.field_path for leak in tier.leaks) if tier else ()


def label_final_output(oracle: Any, scenario: Scenario, text: str, model: str) -> OracleLabel:
    verdicts = getattr(oracle._llm_judge, "verdicts", [])
    verdicts.clear()
    result = oracle.detect(scenario, final_output_trace(scenario.scenario_id, text, model))
    return OracleLabel(
        leaked=result.channel_results[Channel.C1_FINAL_OUTPUT].leaked,
        presidio_fields=_leaked_fields(result.presidio_result),
        judge_fields=_leaked_fields(result.llm_judge_result),
        judge_verdict=verdicts[-1] if verdicts else None,
    )
