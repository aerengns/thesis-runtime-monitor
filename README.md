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
architecture to factuality, fairness, and privacy — properties neither source system addresses —
is the contribution of this thesis. Evaluation compares the combined system against three
baselines (no monitoring, enforcement only, prediction only) using OpenFactCheck for factuality,
FairEval for fairness, and AgentLeak for privacy.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env   # then add your OpenAI API key
```

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
| `papers/` | The five reference papers (ProbGuard, AgentSpec, OpenFactCheck, FairEval, AgentLeak) |

Status: scaffolding only — no logic implemented yet.
