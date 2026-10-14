"""Stream a chat completion from the local model under test."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import ollama

DEFAULT_MODEL = "qwen3:4b-instruct"
MAX_TOKENS = 512


@dataclass(frozen=True)
class Chunk:
    offset: int
    text: str
    logprobs: tuple[float, ...] = ()


@dataclass(frozen=True)
class Generation:
    chunks: tuple[Chunk, ...]
    done_reason: str | None
    eval_count: int | None

    @property
    def text(self) -> str:
        return "".join(c.text for c in self.chunks)

    @property
    def truncated(self) -> bool:
        return self.done_reason == "length"


def collect(responses: Iterable[ollama.ChatResponse]) -> Generation:
    chunks: list[Chunk] = []
    offset = 0
    last = None
    for response in responses:
        last = response
        text = response.message.content or ""
        if not text:
            continue
        logprobs = tuple(lp.logprob for lp in response.logprobs or ())
        chunks.append(Chunk(offset, text, logprobs))
        offset += len(text)

    if last is None or not last.done:
        raise RuntimeError("stream ended before the model finished")
    return Generation(tuple(chunks), last.done_reason, last.eval_count)


def generate(
    messages: list[dict[str, str]],
    *,
    seed: int = 0,
    model: str = DEFAULT_MODEL,
    max_tokens: int = MAX_TOKENS,
    client: ollama.Client | None = None,
) -> Generation:
    client = client or ollama.Client()
    stream = client.chat(
        model=model,
        messages=messages,
        stream=True,
        logprobs=True,
        # Unload after each call: a reused prompt cache changes greedy output.
        keep_alive=0,
        options={"temperature": 0, "seed": seed, "num_predict": max_tokens},
    )
    return collect(stream)
