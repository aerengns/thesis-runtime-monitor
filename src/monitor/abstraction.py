"""Turn a streamed output into abstract states, one per chunk."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Iterable, Iterator, Protocol, TypeVar


class Latchable(Protocol):
    def __or__(self, other): ...
    def __le__(self, other) -> bool: ...


S = TypeVar("S", bound=Latchable)


@dataclass(frozen=True)
class StateSequence(Generic[S]):
    states: tuple[S, ...]
    raw_drops: int


def prefixes(chunks: Iterable[str]) -> Iterator[str]:
    text = ""
    for chunk in chunks:
        text += chunk
        yield text


def state_sequence(chunks: Iterable[str], abstract: Callable[[str], S]) -> StateSequence[S]:
    """Latched states: a predicate stays on once any prefix has set it.

    Streamed text cannot be taken back, so "has appeared so far" is the right
    meaning. ``raw_drops`` counts steps where the unlatched predicates fell.
    """
    states: list[S] = []
    raw_drops = 0
    previous_raw = None
    for prefix in prefixes(chunks):
        raw = abstract(prefix)
        if previous_raw is not None and not previous_raw <= raw:
            raw_drops += 1
        states.append(states[-1] | raw if states else raw)
        previous_raw = raw
    return StateSequence(tuple(states), raw_drops)
