"""Show the privacy predicates escalating as an output streams in.

This is the clearest single demonstration of what Layer 1 observes: the flags
switch on one at a time, so a monitor seeing the middle row should already
suspect trouble before the final row arrives. Run via `make demo`.

No LLM and no network needed -- regex plus one small spaCy pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentleak.core.channels import Channel  # noqa: E402

from src.evaluation.agentleak_eval import load_scenarios  # noqa: E402
from src.properties.privacy import (  # noqa: E402
    PREDICATE_NAMES,
    PROPERTY_CTL,
    ScenarioContext,
    abstract,
)

CHUNKS = [
    "The patient record is under review.",
    " Dylan Lee was seen on Tuesday.",
    " Dylan Lee is being treated for diabetes.",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="base", choices=("base", "difficult", "full"))
    ap.add_argument("--index", type=int, default=0, help="which scenario to use")
    ap.add_argument(
        "--channel",
        default="C1_FINAL_OUTPUT",
        help="C1_FINAL_OUTPUT (to the user) or C5_MEMORY_WRITE (internal, not a boundary)",
    )
    args = ap.parse_args()

    scenario = load_scenarios(args.split)[args.index]
    context = ScenarioContext.from_scenario(scenario)
    channel = Channel[args.channel]

    print(f"property  : {PROPERTY_CTL}")
    print(f"scenario  : {scenario.scenario_id}  ({scenario.vertical})")
    print(f"channel   : {channel.value}")
    print(f"may reveal: {sorted(context.allowed_field_names)}")
    print(f"forbidden : {scenario.allowed_set.forbidden_fields}")
    print()
    print("  " + "  ".join(n[:11].rjust(11) for n in PREDICATE_NAMES) + "   unsafe")
    print("  " + "-" * 76)

    text = ""
    for chunk in CHUNKS:
        text += chunk
        state = abstract(text, channel, context)
        bits = "  ".join(str(b).rjust(11) for b in state.to_bits())
        print(f"  {bits}   {str(state.is_unsafe):<6}  <- {chunk.strip()!r}")

    print()
    print("Read the columns: a person is named before anything forbidden appears.")
    print("That gap is the early-warning window Layer 1 exists to exploit.")


if __name__ == "__main__":
    main()
