"""Reproducibility check: compare two independent greedy runs of the same model on Exp A.

Usage: python src/rerun_check.py qwen2.5-1.5b rerun_qwen2.5-1.5b  -> results/rerun_check.txt
"""
import json
import os
import sys

import numpy as np

from rules import check

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
a_name, b_name = sys.argv[1], sys.argv[2]
P = {json.loads(l)["uid"]: json.loads(l) for l in open(os.path.join(ROOT, "data", "prompts.jsonl"))}
load = lambda n: {json.loads(l)["uid"]: json.loads(l)["text"] for l in open(os.path.join(ROOT, "results", "gen", f"{n}.jsonl"))}
A, B = load(a_name), load(b_name)
uids = sorted(u for u in B if u in A and u.startswith("A|"))
same_text = np.mean([A[u] == B[u] for u in uids])
ra, rb = [], []
for u in uids:
    for r in P[u]["rules"]:
        ra.append(check(r, A[u]))
        rb.append(check(r, B[u]))
ra, rb = np.array(ra), np.array(rb)
lines = [f"{a_name} vs {b_name}: {len(uids)} Exp-A prompts, {len(ra)} rule checks",
         f"identical response text: {same_text:.3f}",
         f"rule-level agreement: {np.mean(ra == rb):.3f}",
         f"mean compliance run1 {ra.mean():.4f}  run2 {rb.mean():.4f}  diff {100 * (rb.mean() - ra.mean()):+.2f} pp"]
open(os.path.join(ROOT, "results", "rerun_check.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
