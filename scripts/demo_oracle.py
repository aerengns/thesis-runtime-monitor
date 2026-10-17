"""Grade finished outputs with AgentLeak's own three-tier detector.

This is the ground-truth label the DTMC will be learned against. Run via
`make oracle` (needs ollama for tier 3) or `make oracle-fast` (tiers 1-2 only).

Tier 1+2 are Presidio: verbatim and format-valid identifiers. Tier 3 is an LLM
judge and carries 82% of real leaks, because models paraphrase rather than quote.
The two example outputs below are chosen to show exactly that split.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.evaluation.agentleak_eval import (  # noqa: E402
    build_oracle,
    final_output_trace,
    load_scenarios,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument(
        "--no-judge",
        action="store_true",
        help="skip tier 3 (no ollama needed, but misses paraphrased leaks)",
    )
    args = ap.parse_args()

    scenario = load_scenarios("base")[args.index]

    # Take name, diagnosis and SSN from ONE vault record. Mixing records would
    # attribute one person's condition to another, so the paraphrase case would
    # not actually correspond to a forbidden field of that subject.
    name = diagnosis = ssn = None
    records = [
        r.fields for r in scenario.private_vault.records if isinstance(r.fields, dict)
    ]
    for fields in records:
        if ("name" in fields or "patient_name" in fields) and "diagnosis" in fields:
            name = fields.get("name") or fields.get("patient_name")
            diagnosis, ssn = fields.get("diagnosis"), fields.get("ssn")
            break
    else:  # no single record has both; fall back to the first named record
        for fields in records:
            if "name" in fields or "patient_name" in fields:
                name = fields.get("name") or fields.get("patient_name")
                diagnosis, ssn = fields.get("diagnosis"), fields.get("ssn")
                break

    print(f"scenario : {scenario.scenario_id} ({scenario.vertical})")
    print(f"forbidden: {scenario.allowed_set.forbidden_fields}")
    print(f"vault    : name={name!r} diagnosis={diagnosis!r} ssn={ssn!r}")
    print(f"tier 3   : {'OFF' if args.no_judge else 'ON (local ollama)'}\n")

    cases = [
        ("clean", "The visit went well and the patient is stable. Follow up in two weeks."),
        ("verbatim", f"Patient {name}, SSN {ssn}, diagnosis {diagnosis}."),
        ("paraphrase", f"{name} suffers from recurring severe one-sided headaches with aura."),
    ]

    oracle = build_oracle(enable_judge=not args.no_judge)

    print(f"  {'case':<12}{'leaked?':<10}{'fields':<8}{'seconds':<9}tiers that fired")
    print("  " + "-" * 64)
    for label, text in cases:
        start = time.time()
        result = oracle.detect(
            scenario, final_output_trace(scenario.scenario_id, text, "qwen3:4b-instruct")
        )
        fired = []
        if result.presidio_result and result.presidio_result.matches_count:
            fired.append("1+2 presidio")
        if result.llm_judge_result and result.llm_judge_result.matches_count:
            fired.append("3 judge")
        print(
            f"  {label:<12}{str(result.has_leakage):<10}"
            f"{len(result.all_leaks):<8}{time.time() - start:<9.1f}"
            f"{', '.join(fired) or '-'}"
        )

    if args.no_judge:
        print("\nWith tier 3 off, the paraphrase row should read False -- that is the")
        print("82% of real leaks pattern matching cannot see. Re-run `make oracle`.")


if __name__ == "__main__":
    main()
