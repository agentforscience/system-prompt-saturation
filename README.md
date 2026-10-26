# System Prompt Saturation

This project measures how often open LLMs follow individual **system-prompt rules** as four things change:
prompt length, number of rules, where a rule sits, and whether rules contradict each other. It uses 9 local
instruction models (0.5B–14.7B; Qwen2.5, Llama-3.x and Qwen3), 40 rules checked automatically, 4 contradictory
rule pairs, and the same 100 user queries in every condition. Full write-up: **[REPORT.md](REPORT.md)**.

## Key findings
- **Length (rule count fixed at 5):** per-rule compliance falls about **7 pp (≈15% relative) per doubling** of
  system-prompt length beyond 500 tokens, pooled over 7 models. That is 3–6× steeper than the hypothesized 2–5%.
  Qwen2.5-7B goes from 0.83 to 0.24 between ~100 and ~8k tokens.
- **Position:** in a numbered list there is only a *primacy* effect: the first 20% get +5 pp, the last 20% get
  −3 pp, and the edges-vs-middle ratio is 1.04, not the predicted 1.15–1.25. When rules are scattered through long
  text, the effect is U-shaped (ratio 1.36). A rule **block placed at the end** of a long prompt is followed
  +11 pp (1k tokens) to +21 pp (4k tokens) more than the same block in the middle, and any contiguous block beats rules scattered through the text.
- **Conflicts:** adding the contradicting rule cuts compliance on each rule of the pair by **30–83%**, not 20–30%.
  For length conflicts, models usually compromise and satisfy *neither* rule (58–90% of responses). "Explain
  thoroughly" beats "be concise".
- **Rule count / size:** there is no sharp saturation knee. Every model degrades gradually from about 5 rules.
  - 7–8B Qwen2.5/Llama keep ~85% of their chance-corrected compliance at 10 rules and ~70% at 20.
  - Sub-3B models keep 58–73% at 10 rules and 36–56% at 20.
  - Qwen3-8B is the exception (0.92 at 10 rules, 0.80 at 40), and Qwen2.5-14B (AWQ) is no better than 7B.

## Reproduce
```bash
uv venv --python 3.12 && source .venv/bin/activate && uv sync      # vllm 0.32, pandas, statsmodels, ...
export HF_HOME=/data/transformers HF_HUB_OFFLINE=1                 # models/datasets from the local HF cache
python src/build_prompts.py                                         # -> data/prompts.jsonl (3,600 prompts)
CUDA_VISIBLE_DEVICES=0 python src/generate.py --model Qwen/Qwen2.5-7B-Instruct --name qwen2.5-7b \
    --gpu-gb 20 --kv-gb 4                                           # -> results/gen/qwen2.5-7b.jsonl (resumable)
# (src/run_gpu_queue.sh <gpu> <jobfile> runs a list of models with retries; job files are in logs/jobs_*.txt)
cd src && python score.py && python analyze.py                      # -> results/tables, figures, stats.json
python rerun_check.py qwen2.5-1.5b rerun_qwen2.5-1.5b               # reproducibility check
```
Generation is greedy (temperature 0, seed 42, ≤1,024 new tokens). The model checkpoints are listed in REPORT.md §3.

## Layout
```
planning.md              research plan + motivation
REPORT.md                full report with results
src/rules.py             40-rule bank, 4 conflict pairs, deterministic checkers
src/build_prompts.py     Exps A (count), B (length), D (conflict), E (count @4k), F (block placement)
src/generate.py          vLLM generation (shared-GPU memory handoff, resumable, per-model lock)
src/score.py             per-(response, rule) scoring + N=0 base rates
src/analyze.py           bootstrap CIs, fixed-effect regressions, McNemar tests, figures
src/hold_gpu.py, src/run_gpu_queue.sh   shared-GPU job management
data/                    prompts.jsonl, queries.json
results/gen/             raw model outputs (one jsonl per model)
results/tables/          all result tables (csv); results/analysis_output.txt = printed analysis
figures/                 expA_count, expB_length, position, expD_conflict, expE_count_long, expF_placement
logs/                    generation logs and job files
```
