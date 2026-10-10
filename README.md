# thesis-runtime-monitor

A runtime monitoring system for LLM behaviour targeting three human-centred properties:
**factuality**, **fairness**, and **privacy**. The system has two layers. Layer 1 is a
*probabilistic monitor* (ProbGuard-style): from a prefix of an LLM output stream or agent
action trace it estimates the probability that the completed output will violate a property
threshold, using a Discrete-Time Markov Chain learned from execution traces and checked
against CTL specifications. Layer 2 is a *declarative enforcement engine* (AgentSpec-style):
property-specific mitigation rules written in a small DSL, dispatching actions such as block,
rewrite, redact, and flag.

The two layers are joined by a **probability-graded mitigation interface**: outputs below a
low threshold pass through untouched, outputs above a high threshold are blocked or rewritten,
and the band in between is flagged or warned. That integration, plus the application of this
architecture to factuality, fairness, and privacy properties neither source system addresses
is the contribution of this thesis. Evaluation compares the combined system against three
baselines (no monitoring, enforcement only, prediction only) using OpenFactCheck for factuality,
FairEval for fairness, and AgentLeak for privacy.

## Quick start

Everything routine is a `make` target. Run `make` on its own for the full list.

```bash
make install     # venv + dependencies + spaCy model
cp .env.example .env
make check       # verify python, packages, benchmark data, ollama
make status      # what is built vs still an empty stub
make demo        # watch the privacy predicates escalate as text streams in
make test        # run the test suite
```

| Target | What it does |
|---|---|
| `make status` | which pipeline stages are built, which are empty, plus git state |
| `make check` | environment health check; exits non-zero if something required is missing |
| `make demo` | the five privacy predicates switching on across a growing prefix (no LLM needed) |
| `make oracle` | grade example outputs with all three detection tiers (starts ollama; ~40 s per case on an M2/8 GB with Qwen3 4B, so ~2 min for the three cases — the judge has no on-disk cache, so every run pays this again) |
| `make oracle-fast` | tiers 1–2 only: instant, but misses paraphrased leaks |
| `make scenarios` | summarise the benchmark splits |
| `make ollama` / `ollama-stop` | start or stop the local model server |

Scripts behind these targets live in `scripts/`; the Makefile is written for
GNU Make 3.81 (the version macOS ships), so multi-line logic lives in the
scripts rather than in recipes.

## Setup (manual equivalent of `make install`)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env
```

### LLM backends

No paid API account is required. The two roles are served by different backends:

| Role | Backend | Why |
|---|---|---|
| **System under test** — generates the traces Layer 1 learns its DTMC from, and performs Layer 2 rewrites | Local `qwen3:4b-instruct` via [Ollama](https://ollama.com) | Free at unlimited volume, deterministic under a fixed seed, and exposes token logprobs for state abstraction |
| **Judge** — OpenFactCheck / FairEval scoring | Gemini, free-tier key from [Google AI Studio](https://aistudio.google.com/apikey) | Higher quality where it matters; evaluation volume fits the free rate limits |

```bash
brew install ollama          # or https://ollama.com/download
ollama serve &
ollama pull qwen3:4b-instruct
```

Token logprobs are only returned by Ollama's OpenAI-compatible endpoint
(`/v1/chat/completions` with `logprobs: true`), not by the native `/api/generate`.
The `openai` package in `requirements.txt` exists to talk to that local endpoint.

Note that the Gemini API key is a **free-tier developer key from AI Studio**, which is
separate from and not obtainable via a consumer Gemini or Claude subscription.

Because the system under test is a 4B local model rather than a frontier model, absolute
scores are not comparable to figures published alongside OpenFactCheck, FairEval, or
AgentLeak. The evaluation is internally consistent: all four configurations (no monitoring,
enforcement only, prediction only, combined) run against the same SUT.

## Layout

| Path | Contents |
|---|---|
| `src/monitor/` | Layer 1 — DTMC, CTL checking, state abstraction, orchestrator |
| `src/enforcement/` | Layer 2 — rule DSL parser, rule engine, mitigation actions |
| `src/interface/` | Probability → enforcement action bridge |
| `src/properties/` | Detection signals for factuality, fairness, privacy |
| `src/evaluation/` | Benchmark harnesses (OpenFactCheck, FairEval, AgentLeak) |
| `rules/` | DSL rule files, one per property |
| `tests/` | pytest suite |
| `notebooks/` | Exploratory analysis |
| `scripts/` | Helper scripts invoked by the Makefile |
| `data/agentleak/` | Vendored AgentLeak benchmark — see its `PROVENANCE.md` |
| `papers/` | The five reference papers (ProbGuard, AgentSpec, OpenFactCheck, FairEval, AgentLeak) |

Status: scaffolding only — no logic implemented yet.
