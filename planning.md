# Planning: System Prompt Saturation

## Motivation & Novelty Assessment

### Why This Research Matters
Production system prompts routinely run to thousands of tokens and dozens of rules (personas, formatting,
safety, tool policies). Practitioners assume every rule added is a rule followed. If per-rule compliance decays with
length, rule count, rule position, or rule conflicts, prompt authors need quantitative guidance on how many rules a
model of a given size can carry and where to put the important ones.

### Gap in Existing Work
The resource-finder stage produced no `literature_review.md`, `papers/`, `datasets/` or `code/` in this workspace
(and no API keys are configured), so the review below is from background knowledge. Related work:
IFEval (Zhou et al., 2023) measures compliance with 1–3 verifiable instructions in the *user* turn; "Lost in the
Middle" (Liu et al., 2023) shows U-shaped position effects for *retrieval*; ManyIFEval / "Curse of instructions"
(Harada et al., 2024/25) shows that the probability of following *all* instructions falls with their count;
FollowBench / ComplexBench add constraints incrementally. What is missing is a single controlled study that, in the
**system prompt**, separates (a) total length (holding rule count fixed), (b) rule count, (c) rule position, and
(d) contradictory rules, and does so across a size ladder of open models with *per-rule* (not all-or-nothing) metrics.

### Our Novel Contribution
A factorial, programmatically-scored benchmark of 40 mutually compatible verifiable system-prompt rules plus
4 conflict pairs, run on 9 open instruction models from 0.5B to 14B, with paired designs (same user queries across
conditions) and baseline (no-rule) rates to correct for rules that are satisfied by chance.

### Experiment Justification
- **Exp A (rule count N ∈ {0,1,2,5,10,15,20,30,40})**: tests the saturation claim (sub-3B saturate at ~10 rules,
  7B+ sustain ~20). N=0 gives base rates for every rule.
- **Exp B (length, N=5 fixed, neutral filler to 0/500/1k/2k/4k/8k tokens)**: isolates length from rule count to test
  "2–5% per doubling beyond 500 tokens".
- **Position analysis (Exp A N≥10 and Exp B)**: rules are randomly ordered/inserted, so relative position is
  randomized; tests "first/last 20% followed 15–25% more often than middle".
- **Exp D (conflicts)**: X-only vs Y-only vs X+Y for 4 contradictory pairs, holding rule count fixed; tests
  "conflict drops compliance on both by 20–30%".

## Research Question
How does per-rule compliance with system-prompt rules depend on prompt length, rule count, rule position and
rule conflicts, and does the saturation point scale with model size?

## Hypothesis Decomposition
- H1: per-rule compliance falls 2–5 pp (relative) per doubling of system prompt length beyond 500 tokens (fixed N).
- H2: rules in the first and last 20% of the prompt are followed 15–25% (relative) more often than middle rules.
- H3: adding a contradicting rule lowers compliance on each member of the pair by 20–30% (relative).
- H4: per-rule compliance in models <3B degrades past ~10 rules; ≥7B models hold up to ~20 rules.

## Proposed Methodology
- Models (local, vLLM, greedy, temperature 0): Qwen2.5-Instruct 0.5B/1.5B/3B/7B/14B(AWQ), Llama-3.2-Instruct 1B/3B,
  Llama-3.1-8B-Instruct, Qwen3-8B (non-thinking). Real models; no simulation. No API key available.
- User queries: 100 open-ended instructions from Alpaca-cleaned (no input field), same 100 in every cell (paired).
- Rules: 40 verifiable rules (keyword inclusion, start/end phrases, formatting, exclusion, punctuation).
  Each scored by a deterministic Python checker. Random subset and order per item (seeded).
- Filler: Wikipedia (WikiText-103) paragraphs presented as reference material; rules inserted as standalone
  "Response rule:" paragraphs at random paragraph boundaries.
- Metrics: per-rule compliance (mean of binary checks), baseline-corrected lift over N=0, all-rules-followed rate.
- Statistics: item-clustered bootstrap 95% CIs; mixed/fixed-effect logistic regressions with rule fixed effects
  (log2 length slope; position bins; conflict indicator); McNemar-style paired tests for conflict; Holm correction
  across the hypothesis families. α = 0.05.

## Expected Outcomes
Support for H1: negative log2-length slope of 2–5% relative beyond 500 tokens. H2: U-shape with edge/middle ratio
1.15–1.25. H3: conflict ratio 0.70–0.80. H4: knee at ~10 rules for small, ~20 for large models.
Any other pattern (no slope, monotone position effect, a model winning one side of a conflict) refutes/qualifies.

## Timeline
Setup+prompt build 45 min; generation 2–4 h (shared GPUs); scoring/analysis 1 h; report 45 min.

## Potential Challenges
Shared GPUs (one 48 GB GPU reserved with a holder; small models on ~5 GB slices); checkers may be too strict/lenient
(audit samples); filler content could confound (use neutral encyclopedic text, randomized per item).

## Success Criteria
All four hypotheses tested with CIs on ≥6 models; report states clearly which are supported.

## Deviations from the plan (recorded during execution)
- **Added Exp E** (N ∈ {10,20,40} rules interleaved in ~4k tokens of padding) after the first model ran, to separate
  rule count from prompt length. **Added Exp F** (a 5-rule block at the top/middle/bottom of 1k/4k-token prompts) to
  separate distance-from-end from interleaving. Both use separate RNG streams, so A/B/D prompts are unchanged.
- **Quantization.** Shared GPUs had only 5–13 GB free for most of the session. The 3B, 8B and 14B models (and
  Qwen3-8B) were therefore run as AWQ int4 checkpoints. The 0.5B–1.5B and 7B models are bf16.
- **Coverage.**
  - Qwen3-8B and Qwen2.5-14B completed only Exps A and D; their long-prompt runs could not get enough KV-cache
    memory in time.
  - Qwen2.5-7B lacks Exps E and F.
- **Exp D metrics.** The hard conflicts are mutually exclusive by construction, so the analysis added "winner"
  (X-only / Y-only / neither) rates and word-count distributions to the planned relative-drop metric.
