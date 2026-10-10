# AgentLeak benchmark data — provenance

Vendored so that thesis results reproduce even if upstream changes or disappears.

| | |
|---|---|
| Source | https://github.com/Privatris/AgentLeak |
| Pinned commit | `1301eb9245a03bbee8694ee2b74accbc8f6fefe6` (2026-09-07) |
| Upstream path | `agentleak_data/datasets/` |
| License | MIT — Copyright (c) 2025 Faouzi El Yagoubi, Ranwa Al Mallah, Polytechnique Montréal |
| Paper | `papers/AgentLeak.pdf` |

Files copied verbatim, unmodified:

| File | Records | Notes |
|---|---|---|
| `scenarios_base_100.jsonl` | 100 | balanced 25/vertical, all `difficulty: medium`; 36 with no attack, 28 single-agent |
| `scenarios_difficult_100.jsonl` | 100 | harder split |
| `scenarios_full_1000.jsonl` | 1000 | full benchmark |

Also copied verbatim, from `agentleak/config/configs/prompts/` at the same commit:

| File | Why vendored |
|---|---|
| `prompts/llm_judge_system.md` | Not shipped in the installed package |
| `prompts/llm_judge_few_shot.md` | Not shipped in the installed package |

The installed wheel omits these non-Python data files, and `LLMJudge` falls back to
**no system prompt** when it cannot find them — silently gutting Tier 3, which carries
82% of real leaks. Verified: with the prompts absent a paraphrased diagnosis went
undetected; with them wired in, the same text is correctly flagged.

The `agentleak` package itself is **not** vendored; it is installed from the same
pinned commit (see `requirements.txt`).

All personal data in these scenarios is synthetic — generated with Faker and
pre-anonymised by the benchmark authors. No real personal data is present.
