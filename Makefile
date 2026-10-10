# Thesis runtime monitor -- common tasks.
#
#   make            show this help
#   make status     what is built and what is still an empty stub
#
# Written for GNU Make 3.81 (the version macOS ships), so no .ONESHELL:
# anything multi-line lives in scripts/ and is called from here.

PY      := .venv/bin/python
PIP     := .venv/bin/pip
PYTEST  := $(PY) -m pytest
OLLAMA_LOG := /tmp/thesis-ollama.log
SUT     := qwen3:4b-instruct

.DEFAULT_GOAL := help
.PHONY: help status check test test-v install venv deps model data demo oracle \
        oracle-fast scenarios ollama ollama-stop clean clean-cache

## ---------------------------------------------------------------- orientation

help: ## Show this help
	@echo "Thesis runtime monitor -- available targets:"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Start with: make check   (is the environment sane?)"
	@echo "            make status  (what is actually built?)"
	@echo "            make demo    (see the monitor's predicates in action)"

status: ## What is built vs still an empty stub, plus git state
	@$(PY) scripts/project_status.py

check: ## Verify python, packages, spacy model, benchmark data, ollama
	@$(PY) scripts/check_env.py

## ---------------------------------------------------------------- setup

install: venv deps model ## Full setup: venv, dependencies, spacy model
	@echo
	@echo "Setup done. Next: cp .env.example .env, then 'make check'."

venv: ## Create the Python 3.11 virtualenv
	@test -d .venv || python3.11 -m venv .venv
	@$(PIP) install --quiet --upgrade pip
	@echo "venv ready: $$($(PY) --version)"

deps: ## Install requirements.txt into the venv
	@$(PIP) install -r requirements.txt

model: ## Download the spaCy English model (needed for name detection)
	@$(PY) -m spacy download en_core_web_sm

data: ## Re-download the AgentLeak benchmark data (already vendored in data/)
	@echo "Benchmark data is vendored in data/agentleak/ -- see its PROVENANCE.md."
	@echo "Only re-fetch if those files are missing; 'make check' will tell you."

## ---------------------------------------------------------------- run

demo: ## Watch the privacy predicates escalate as text streams in (no LLM needed)
	@$(PY) scripts/demo_abstraction.py

oracle: ollama ## Grade example outputs with all 3 detection tiers (needs ollama, ~30s/case)
	@$(PY) scripts/demo_oracle.py

oracle-fast: ## Grade with tiers 1-2 only: instant, but misses paraphrased leaks
	@$(PY) scripts/demo_oracle.py --no-judge

scenarios: ## Summarise the benchmark splits
	@$(PY) -c "from src.evaluation.agentleak_eval import load_scenarios as L; \
	import collections as c; \
	s=L('base'); \
	print('base split: %d scenarios' % len(s)); \
	print('  verticals  :', dict(c.Counter(x.vertical.value for x in s))); \
	print('  no attack  : %d  (clean baseline set)' % sum(1 for x in s if not x.attack.enabled)); \
	print('  single-agent: %d  (the C1-only first slice)' % sum(1 for x in s if len(x.agents)==1)); \
	print('other splits: difficult=%d full=%d' % (len(L('difficult')), len(L('full'))))"

## ---------------------------------------------------------------- test

test: ## Run the test suite
	@$(PYTEST) tests/ -q

test-v: ## Run the test suite, verbose, with test names
	@$(PYTEST) tests/ -v

## ---------------------------------------------------------------- ollama

ollama: ## Start the local model server if it is not already running
	@curl -sf http://127.0.0.1:11434/api/version >/dev/null 2>&1 \
	  && echo "ollama already running" \
	  || ( echo "starting ollama (log: $(OLLAMA_LOG))..." ; \
	       OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 \
	         nohup ollama serve > $(OLLAMA_LOG) 2>&1 & \
	       for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do \
	         curl -sf http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break || sleep 1; \
	       done ; \
	       curl -sf http://127.0.0.1:11434/api/version >/dev/null 2>&1 \
	         && echo "ollama up" || ( echo "ollama failed to start; see $(OLLAMA_LOG)" ; exit 1 ) )
	@ollama list | grep -q "$(SUT)" || ( echo "pulling $(SUT) (~2.5GB, one time)..." ; ollama pull $(SUT) )

ollama-stop: ## Stop the local model server
	@pkill -f "ollama serve" 2>/dev/null && echo "ollama stopped" || echo "ollama was not running"

## ---------------------------------------------------------------- housekeeping

clean: clean-cache ## Remove caches (keeps the venv)

clean-cache: ## Remove __pycache__ and pytest cache
	@find . -path ./.venv -prune -o -name '__pycache__' -type d -print0 2>/dev/null \
	  | xargs -0 rm -rf 2>/dev/null || true
	@rm -rf .pytest_cache
	@echo "caches removed"
