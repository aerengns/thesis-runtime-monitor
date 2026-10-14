"""Tests for streaming from the model under test, using fake responses."""

from __future__ import annotations

import pytest
from ollama import ChatResponse, Message
from ollama._types import Logprob

from src.evaluation.ollama_stream import MAX_TOKENS, collect, generate


def piece(text: str, logprob: float = -0.5) -> ChatResponse:
    return ChatResponse(
        model="fake",
        done=False,
        message=Message(role="assistant", content=text),
        logprobs=[Logprob(token=text, logprob=logprob)],
    )


def final(done_reason: str = "stop", eval_count: int = 3) -> ChatResponse:
    return ChatResponse(
        model="fake",
        done=True,
        done_reason=done_reason,
        eval_count=eval_count,
        message=Message(role="assistant", content=""),
    )


def test_chunks_record_offsets_and_join_to_text():
    generation = collect([piece("Chad"), piece(" Stewart"), piece(" is"), final()])
    assert generation.text == "Chad Stewart is"
    assert [(c.offset, c.text) for c in generation.chunks] == [
        (0, "Chad"), (4, " Stewart"), (12, " is"),
    ]


def test_logprobs_are_kept_per_chunk():
    generation = collect([piece("Hi", -0.25), final()])
    assert generation.chunks[0].logprobs == (-0.25,)


def test_empty_final_message_is_not_a_chunk():
    generation = collect([piece("Hi"), final()])
    assert len(generation.chunks) == 1


def test_stop_metadata_comes_from_final_response():
    generation = collect([piece("Hi"), final("length", 512)])
    assert generation.done_reason == "length"
    assert generation.eval_count == 512
    assert generation.truncated


def test_stream_cut_off_before_done_raises():
    with pytest.raises(RuntimeError, match="before the model finished"):
        collect([piece("Hi")])


def test_generate_requests_deterministic_capped_stream():
    class FakeClient:
        def chat(self, **kwargs):
            self.kwargs = kwargs
            return iter([piece("ok"), final()])

    client = FakeClient()
    generation = generate([{"role": "user", "content": "q"}], seed=7, client=client)

    assert generation.text == "ok"
    assert client.kwargs["stream"] is True
    assert client.kwargs["logprobs"] is True
    assert client.kwargs["keep_alive"] == 0
    assert client.kwargs["options"] == {
        "temperature": 0, "seed": 7, "num_predict": MAX_TOKENS,
    }
