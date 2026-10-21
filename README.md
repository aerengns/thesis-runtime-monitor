# thesis-runtime-monitor

Master's thesis project: a runtime monitor that reads an LLM's output **while it is being
generated** and predicts whether the finished output will violate a property, applied to
**factuality**, **fairness** and **privacy**.

## Overview

The system has two layers. Layer 1 is a *probabilistic monitor* (ProbGuard-style): it
abstracts each prefix of the output stream into a symbolic state and, using a Discrete-Time
Markov Chain learned from execution traces, estimates the probability that the completed
output will violate the property. Layer 2 is a *declarative enforcement engine*
(AgentSpec-style): property-specific rules written in a small DSL that block, rewrite,
redact or flag output.

The two layers are joined by a **probability-graded interface**: outputs below a low
threshold pass through, outputs above a high threshold are blocked or rewritten, and the
band in between is flagged. That integration, plus its application to factuality, fairness
and privacy (properties neither source system addresses), is the contribution of this
thesis. Evaluation compares the combined system against three baselines (no monitoring,
enforcement only, prediction only) using OpenFactCheck, FairEval and AgentLeak.

Privacy is being built first, end to end, before factuality and fairness.

## Current status

| Part | State |
|---|---|
| Privacy predicates: 5 Boolean predicates per prefix, 22 abstract states, invariant `AG !(binding_established & boundary_crossing)` | built |
| AgentLeak scenario loading and ground-truth oracle (Presidio + LLM judge on local Ollama) | built |
| Trace pipeline: AgentLeak prompts, streamed generation, one abstract state per token, labelling of the final output | built |
| Persisting runs and the batch runner | next |
| DTMC, P(violation), graded interface | planned |
| Layer 2 rule DSL, engine and mitigations | planned |
| Factuality and fairness | planned |

`make status` prints this live from the code. The test suite has 61 tests.

## Setup

Requirements: macOS or Linux, Python 3.11, and [Ollama](https://ollama.com) (tested with
0.34; older versions do not return token logprobs). Developed on an Apple M2 with 8 GB RAM.

```bash
make install                 # venv, dependencies and the spaCy model
cp .env.example .env         # optional: only needed for the Gemini key (see below)

brew install ollama          # or https://ollama.com/download
ollama serve &               # or: make ollama
ollama pull qwen3:4b-instruct

make check                   # verify Python, packages, benchmark data and Ollama
make test                    # run the test suite
```

Manual equivalent of `make install`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The AgentLeak benchmark data is already included in `data/agentleak/`, so no download is
needed.

## Common commands

Run `make` on its own for the full list.

| Target | What it does |
|---|---|
| `make status` | which pipeline stages are built and which are still empty, plus git state |
| `make check` | environment health check; exits non-zero if something required is missing |
| `make demo` | the five privacy predicates switching on across a growing prefix (no LLM needed) |
| `make oracle` | grade example outputs with all detection tiers (starts Ollama; ~40 s per case on an M2/8 GB, and the judge keeps no cache between runs) |
| `make oracle-fast` | Presidio only: instant, but misses paraphrased leaks |
| `make scenarios` | summarise the benchmark splits |
| `make test` | run the test suite |
| `make ollama` / `make ollama-stop` | start or stop the local model server |

The Makefile is written for GNU Make 3.81 (the version macOS ships), so multi-line logic
lives in `scripts/`.

## Models

No paid API account is required.

| Role | Backend |
|---|---|
| **System under test** | Local `qwen3:4b-instruct` via Ollama |
| **Privacy judge** (AgentLeak's semantic tier) | The same local model, constrained to the JSON format AgentLeak's judge prompt asks for |
| **Cross-check and later evaluation** (OpenFactCheck, FairEval) | Gemini, via a free-tier key from [Google AI Studio](https://aistudio.google.com/apikey) in `.env` |

Generation runs at temperature 0 with a fixed seed, and the model is unloaded after every
call: a reused prompt cache changed greedy output for the same seed, so only cold runs
reproduce.

Because the system under test is a 4B local model, absolute scores are not comparable to
figures published with the benchmarks. The evaluation is internally consistent: all four
configurations run against the same model.

## Notes on the AgentLeak benchmark

AgentLeak is pinned to commit `1301eb9` and its data is vendored (see
`data/agentleak/PROVENANCE.md`). Six problems in the released code are worked around and
documented where they are handled:

1. The released datasets do not validate against the package's own schema
   (`normalise_scenario` in `src/evaluation/agentleak_eval.py`).
2. Presidio is required but not declared as a dependency; without it nothing is detected.
3. The judge's prompt files are missing from the installed package (vendored in
   `data/agentleak/prompts/`).
4. The judge's "disk-based" cache only lives in memory, so labels must be persisted by us.
5. The runner never injects the attack payload into the prompt
   (`src/evaluation/agentleak_prompts.py` adds an explicit switch).
6. The judge's verdict can be silently lost in parsing (`src/evaluation/agentleak_judge.py`
   constrains its output format and records its verdict).

## Layout

| Path | Contents |
|---|---|
| `src/properties/` | Predicates per property (`privacy.py` is built) |
| `src/monitor/` | Layer 1: state abstraction (built), DTMC, CTL checking, monitor |
| `src/evaluation/` | Benchmark loading, prompts, streaming, oracle and judge |
| `src/interface/` | Probability → enforcement action bridge |
| `src/enforcement/` | Layer 2: rule DSL parser, rule engine, mitigation actions |
| `rules/` | DSL rule files, one per property |
| `tests/` | pytest suite |
| `scripts/` | Helper scripts invoked by the Makefile |
| `data/agentleak/` | Vendored AgentLeak benchmark and judge prompts |
| `docs/` | Design notes (`01-layer1-design.md`: what ProbGuard and AgentSpec specify) |
| `papers/` | The five reference papers (ProbGuard, AgentSpec, OpenFactCheck, FairEval, AgentLeak) |
| `notebooks/` | Exploratory analysis |
