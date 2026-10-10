"""Show which parts of the pipeline are built and which are still empty stubs.

The repository was scaffolded complete on day one, so the folder tree looks
finished while most files are empty. Run via `make status` to see the truth.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Pipeline order, not alphabetical: each stage needs the one above it.
STAGES: list[tuple[str, str, str]] = [
    ("src/properties/privacy.py", "privacy predicates", "the 5 yes/no flags read off an output prefix"),
    ("src/evaluation/agentleak_eval.py", "benchmark + oracle", "loads 1200 scenarios; grades finished outputs"),
    ("src/monitor/abstraction.py", "abstraction plumbing", "generic prefix -> state, shared across properties"),
    ("src/monitor/dtmc.py", "DTMC", "turns recorded traces into transition probabilities"),
    ("src/monitor/ctl_checker.py", "CTL check", "P(reach an unsafe state | current state)"),
    ("src/monitor/prob_monitor.py", "layer 1 monitor", "runtime: current prefix -> P(violation)"),
    ("src/interface/graded_interface.py", "graded interface", "P(violation) -> pass / flag / block"),
    ("src/enforcement/dsl_parser.py", "rule DSL parser", "parses rules/*.rules"),
    ("src/enforcement/rule_engine.py", "rule engine", "evaluates rules, dispatches actions"),
    ("src/enforcement/mitigations.py", "mitigations", "block / rewrite / redact / flag"),
    ("src/properties/factuality.py", "factuality predicates", "second property, after privacy works"),
    ("src/properties/fairness.py", "fairness predicates", "third property; needs paired outputs"),
    ("src/evaluation/openfactcheck_eval.py", "OpenFactCheck harness", "factuality evaluation"),
    ("src/evaluation/faireval_eval.py", "FairEval harness", "fairness evaluation"),
]

EXTRA = [
    ("rules/privacy.rules", "privacy rules", "DSL rules for the privacy property"),
]


def lines_of(path: Path) -> int:
    if not path.is_file():
        return -1
    return sum(1 for _ in path.open(encoding="utf-8", errors="replace"))


def main() -> None:
    print("PIPELINE STATUS  (in dependency order -- each stage needs the one above)\n")
    done = 0
    rows = STAGES + EXTRA
    for rel, name, why in rows:
        n = lines_of(ROOT / rel)
        if n > 0:
            done += 1
            mark, detail = "[built]", f"{n:>4} lines"
        elif n == 0:
            mark, detail = "[empty]", "   stub"
        else:
            mark, detail = "[none ]", " missing"
        print(f"  {mark} {detail}  {name:<24} {why}")

    print(f"\n  {done} of {len(rows)} built.")

    # Tests
    test_files = sorted((ROOT / "tests").glob("test_*.py"))
    live = [t for t in test_files if lines_of(t) > 0]
    print(f"\nTESTS\n  {len(live)} of {len(test_files)} test files written")
    for t in test_files:
        n = lines_of(t)
        print(f"  {'[built]' if n else '[empty]'} {n:>4} lines  tests/{t.name}")

    # Git
    print("\nGIT")
    try:
        head = subprocess.run(
            ["git", "-C", str(ROOT), "log", "--oneline", "-1"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--short"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        print(f"  HEAD: {head}")
        if dirty:
            print(f"  {len(dirty.splitlines())} uncommitted path(s):")
            for line in dirty.splitlines():
                print(f"      {line}")
        else:
            print("  working tree clean")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  (not a git repository, or git unavailable)")

    print("\nNEXT: the trace harness -- run scenarios through the local model and")
    print("      record streamed prefixes. Nothing below it can be built or tested")
    print("      until recorded traces exist. See docs/01-layer1-design.md.")


if __name__ == "__main__":
    main()
