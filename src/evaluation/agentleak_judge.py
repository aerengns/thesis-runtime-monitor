"""AgentLeak's LLM judge on local Ollama, constrained to its own output format."""

from __future__ import annotations

import json
from typing import Any

import ollama
from agentleak.detection.llm_judge import LLMJudge

SEVERITIES = ["critical", "high", "medium", "low"]


def _strings(*names: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {n: {"type": "string"} for n in names},
        "required": list(names),
    }


# The JSON format requested by data/agentleak/prompts/llm_judge_system.md.
_LEAK = _strings("field_name", "leaked_content", "original_value")
_LEAK["properties"] |= {
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "severity": {"enum": SEVERITIES},
    "leak_type": {"enum": ["direct", "paraphrase", "inference", "partial", "context"]},
    "reasoning": {"type": "string"},
}
_LEAK["required"] += ["confidence", "severity", "leak_type", "reasoning"]

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"enum": [0, 1]},
        "has_leakage": {"type": "boolean"},
        "overall_severity": {"enum": SEVERITIES + ["none"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning_chain": _strings(
            "vault_analysis", "output_scan", "match_check", "context_check", "verdict"
        ),
        "leaks": {"type": "array", "items": _LEAK},
        "reasoning_summary": {"type": "string"},
    },
    "required": [
        "score", "has_leakage", "overall_severity", "confidence",
        "reasoning_chain", "leaks", "reasoning_summary",
    ],
}


def _verdict(text: str, done_reason: str | None) -> bool | None:
    if done_reason == "length":
        return None
    try:
        return json.loads(text)["score"] == 1
    except (ValueError, KeyError, TypeError):
        return None


class SchemaJudge(LLMJudge):
    """Unconstrained, qwen3:4b renamed the leak keys and the parser dropped real leaks.

    ``verdicts`` keeps the judge's own yes/no per call (None if cut off or
    unreadable), so a verdict lost in parsing stays visible.
    """

    def __init__(self, config: Any, client: ollama.Client | None = None):
        super().__init__(config)
        self._ollama = client or ollama.Client()
        self.verdicts: list[bool | None] = []

    def _call_openrouter(self, prompt: str, api_key: str) -> str:
        response = self._ollama.chat(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
            format=JUDGE_SCHEMA,
            keep_alive=0,
            options={
                "temperature": self.config.temperature,
                "seed": 0,
                "num_predict": self.config.max_tokens,
            },
        )
        text = response.message.content or ""
        self.verdicts.append(_verdict(text, response.done_reason))
        return text
