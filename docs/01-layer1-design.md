# Design note 1 — What ProbGuard and AgentSpec actually specify

Notes taken from `papers/ProbGuard.pdf` (§3) and `papers/agentspec_llm_enforcement_icse26.pdf`
(§3) before implementing either layer. Records the decisions the source papers force on us
and, more importantly, the places where this thesis must depart from them.

## 1. ProbGuard derives the abstraction from the property

The single most important detail, from Algorithm 1 (line 2):

```
P_ψ ← {φ_s1, φ_s2, ..., φ_sn | φ_si ∈ ψ}     ⊲ Derive predicates
```

The predicate set is **extracted from the atomic propositions appearing in the CTL property
ψ**. The abstract state space is then every Boolean valuation over those predicates,
`S ⊆ {0,1}^n`, with semantically invalid combinations pruned.

Consequence for us: we do not design the abstraction function and the property separately.
**Writing the property defines the abstraction.** Our design work per property is really the
design of a small predicate vocabulary.

ProbGuard's monitored invariant has one standard form:

```
ψ = AG ¬unsafe
```

CTL is more expressive and the framework supports arbitrary formulas, but the paper itself
uses this invariant throughout. We adopt it too.

## 2. The DTMC and its estimator

`M = (S_M, P_M)`, transitions estimated from empirical counts over abstracted traces, with
**valid-transition-aware Laplace smoothing** (α = 1) applied *only* to semantically valid
transitions:

```
P̂(s_j | s_i) = (n_ij + α) / (n_i + k_i · α)   if valid_tran(s_i, s_j)
             = 0                                otherwise
```

where `k_i` is the number of valid transitions out of `s_i`. Invalid transitions keep exactly
zero mass, which is what preserves logical consistency of the chain. A `valid_tran` relation
is therefore a required input, not an optional refinement — we must define it per property.

## 3. The PAC bound answers "how many traces?"

Algorithm 1 loops until the sample count satisfies a bound derived from |S|, the error ε, and
the confidence δ (with δ' = δ/|S|). This gives a principled, computable stopping criterion for
trace collection rather than an arbitrary "collect 1000 and hope". Worth reporting in the
evaluation chapter — it is a genuine formal-methods contribution that costs us nothing extra.

Practical note: the bound grows with |S|, so a **small predicate vocabulary is not just
simpler, it is cheaper to certify**. Prefer 4–6 predicates per property over 15.

## 4. Runtime monitoring loop

Algorithm 2 is short:

```
P_safe = P_M̂[ψ | π]          ⊲ conditional satisfaction probability given prefix π
if P_safe < θ: intervene
```

Note the polarity: ProbGuard thresholds on the probability that the run stays **safe**, while
our project brief's graded interface is written in terms of P(violation). They are complements,
`P_viol = 1 − P_safe`. We will standardise on **P(violation)** throughout the codebase and
convert once, at the monitor boundary, to avoid sign errors.

## 5. AgentSpec rule structure

A rule is a three-tuple `r = (η_r, P_r, E_r)`:

| Part | Meaning |
|---|---|
| `η_r` | triggering event — one of *state change*, *action* (before execution), or *agent_finish* |
| `P_r` | set of predicate functions, evaluated as `p(u, τ_i) → B`; the rule fires only if **all** hold |
| `E_r` | ordered sequence of enforcement functions |

Concrete syntax is `rule / trigger / check / enforce / end`. AgentSpec parses with ANTLR4; the
project brief specifies Lark, which is equivalent for this grammar and is the better fit for a Python
codebase. No reason to change that decision.

## 6. Where this thesis must depart from both papers

**This is the actual research contribution, and it is not a detail.**

ProbGuard abstracts *concrete environment states* `v ∈ V` — microwave door open, obstacle
distance, vehicle speed. Its predicates read a structured world model. We are monitoring **LLM
output text**, where no such world model exists. Our `v` has to be defined.

So the state abstraction must be built from predicates that are:

1. **Computable on a prefix**, not only on a finished output. The whole value proposition is
   early warning, which means the predicate must be evaluable on a partial generation.
2. **Monotone-ish or at least stable.** A predicate that flickers as tokens arrive produces a
   noisy chain and destroys the transition estimates.
3. **Cheap.** It runs on every streamed chunk.

This is the design risk of the project and where the novelty sits. It should be stated
explicitly in the thesis as the adaptation of ProbGuard from world-state monitoring to
output-stream monitoring.

## 7. The integration point, concretely

Both layers consume **Boolean predicates over the same state**. ProbGuard's abstraction is a
vector of predicate valuations; AgentSpec's `check` clause is a conjunction of predicates.

So we define **one predicate library per property**, and both layers import it:

```
src/properties/privacy.py   →  has_person, has_identifier, has_sensitive_category, ...
        │                              │
        ├──────────────► Layer 1 abstraction (the bit-vector)
        └──────────────► Layer 2 `check` clauses (the rule conditions)
```

The graded interface then passes both the scalar `P(violation)` **and** the predicate valuation
to the rule engine, which lets a rule condition on *why* the risk is high, not just how high it
is. That shared vocabulary is what makes "integration" a real architectural claim rather than
two systems in a trenchcoat.

## 8. Open questions to resolve with the supervisor

- Do we monitor token-streamed output (C1-style, single model) only, or also multi-step agent
  traces (C2-style, inter-agent)? AgentLeak defines both channels; the second is closer to
  ProbGuard's original setting but needs an agent harness.
- Is the `AG ¬unsafe` invariant sufficient for all three properties, or does fairness need a
  comparative/counterfactual formulation? Fairness is a property of a *pair* of outputs (SNSR
  compares demographic variants), which does not fit a single-trace invariant cleanly. This is
  a known wrinkle to raise early.
