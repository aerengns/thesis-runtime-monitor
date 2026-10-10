"""Environment health check: is everything this project needs actually present?

Run via `make check`. Exits non-zero if anything required is missing, so it is
usable as a precondition in other targets.
"""

from __future__ import annotations

import importlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLLAMA = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

OK, BAD, WARN = "  ok ", "FAIL ", "warn "
failures = 0
warnings = 0


def report(status: str, label: str, detail: str = "") -> None:
    global failures, warnings
    if status == BAD:
        failures += 1
    if status == WARN:
        warnings += 1
    print(f"[{status}] {label:<34} {detail}")


# --- interpreter -----------------------------------------------------------
v = sys.version_info
report(
    OK if (v.major, v.minor) >= (3, 11) else BAD,
    "python >= 3.11",
    f"{v.major}.{v.minor}.{v.micro}",
)

# --- packages --------------------------------------------------------------
for module, label in [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("lark", "lark (DSL parser, layer 2)"),
    ("spacy", "spacy"),
    ("sentence_transformers", "sentence-transformers"),
    ("agentleak", "agentleak (privacy benchmark)"),
    ("presidio_analyzer", "presidio-analyzer (oracle t1/t2)"),
    ("pytest", "pytest"),
]:
    try:
        m = importlib.import_module(module)
        report(OK, label, getattr(m, "__version__", ""))
    except ImportError as exc:
        report(BAD, label, str(exc)[:60])

# --- spaCy model -----------------------------------------------------------
try:
    import spacy

    spacy.load("en_core_web_sm")
    report(OK, "spacy model en_core_web_sm")
except Exception as exc:  # noqa: BLE001
    report(BAD, "spacy model en_core_web_sm", f"{type(exc).__name__}: run `make install`")

# --- benchmark data --------------------------------------------------------
for name, minimum in [
    ("scenarios_base_100.jsonl", 100),
    ("scenarios_difficult_100.jsonl", 100),
    ("scenarios_full_1000.jsonl", 1000),
]:
    path = ROOT / "data" / "agentleak" / name
    if not path.is_file():
        report(BAD, f"data/{name}", "missing")
    else:
        lines = sum(1 for _ in path.open(encoding="utf-8"))
        report(OK if lines >= minimum else BAD, f"data/{name}", f"{lines} scenarios")

# The installed agentleak wheel omits these; without them the tier-3 judge runs
# with no system prompt and detects almost nothing.
for name in ("llm_judge_system.md", "llm_judge_few_shot.md"):
    path = ROOT / "data" / "agentleak" / "prompts" / name
    report(
        OK if path.is_file() and path.stat().st_size > 0 else BAD,
        f"judge prompt {name}",
        "" if path.is_file() else "missing -- tier 3 would be silently crippled",
    )

# --- ollama ----------------------------------------------------------------
try:
    with urllib.request.urlopen(f"{OLLAMA}/api/version", timeout=3) as resp:
        import json

        report(OK, "ollama server", json.load(resp).get("version", "?"))
    with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=5) as resp:
        import json

        names = {m["name"] for m in json.load(resp).get("models", [])}
    wanted = os.getenv("SUT_MODEL", "qwen3:4b-instruct")
    report(
        OK if wanted in names else BAD,
        f"model {wanted}",
        "" if wanted in names else "run `ollama pull " + wanted + "`",
    )
except (urllib.error.URLError, TimeoutError, OSError):
    report(WARN, "ollama server", "not reachable -- run `make ollama` (only tier 3 needs it)")
    report(WARN, "local model", "skipped, server down")

# --- optional judge key ----------------------------------------------------
report(
    OK if os.getenv("GEMINI_API_KEY") else WARN,
    "GEMINI_API_KEY",
    "set" if os.getenv("GEMINI_API_KEY") else "unset -- only needed for judge cross-validation",
)

print()
if failures:
    print(f"{failures} required check(s) failed, {warnings} warning(s).")
    sys.exit(1)
print(f"All required checks passed ({warnings} warning(s)).")
