"""Score every generation: one row per (model, prompt, rule) -> results/rule_scores.parquet.

Also scores every bank rule on the no-rule (Exp A, N=0) responses to get per-model chance/base rates
-> results/base_rates.csv.
"""
import glob
import json
import os

import pandas as pd

from rules import ALL_CHECKERS, BANK, CONFLICTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFLICT_OF = {}
for pair, (x, y) in CONFLICTS.items():
    CONFLICT_OF[x[0]] = (pair, "X")
    CONFLICT_OF[y[0]] = (pair, "Y")


def main():
    prompts = {json.loads(l)["uid"]: json.loads(l) for l in open(os.path.join(ROOT, "data", "prompts.jsonl"))}
    rows, base, gens = [], [], []
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "gen", "*.jsonl"))):
        model = os.path.basename(path)[:-6]
        if model.endswith("_smoke"):
            continue
        for l in open(path):
            g = json.loads(l)
            p = prompts[g["uid"]]
            t = g["text"]
            words = len(t.split())
            gens.append(dict(model=model, uid=g["uid"], exp=p["exp"], cond=p["cond"], item=p["item"],
                             n_rules=len(p["rules"]), sys_tokens=g["n_prompt_tokens"], out_tokens=g["n_tokens"],
                             finish=g["finish"], words=words))
            for j, (r, pos) in enumerate(zip(p["rules"], p["pos"])):
                pair, role = CONFLICT_OF.get(r, (None, None))
                rows.append(dict(model=model, uid=g["uid"], exp=p["exp"], cond=p["cond"], item=p["item"],
                                 n_rules=len(p["rules"]), filler=p.get("filler_budget", 0), rule=r, rank=j, pos=pos,
                                 pair=p.get("pair"), conflict_role=role, followed=int(ALL_CHECKERS[r](t)),
                                 truncated=int(g["finish"] == "length"), prompt_tokens=g["n_prompt_tokens"]))
            if p["exp"] == "A" and p["cond"] == "N0":
                for r in ALL_CHECKERS:
                    base.append(dict(model=model, item=p["item"], rule=r, followed=int(ALL_CHECKERS[r](t))))
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(ROOT, "results", "rule_scores.parquet"))
    pd.DataFrame(gens).to_parquet(os.path.join(ROOT, "results", "generations_meta.parquet"))
    b = pd.DataFrame(base).groupby(["model", "rule"]).followed.mean().unstack(0)
    b.to_csv(os.path.join(ROOT, "results", "base_rates.csv"))
    print(df.groupby(["model", "exp"]).followed.agg(["count", "mean"]).unstack(1).round(3).to_string())
    print("\nbase rates (N=0), mean over bank rules:")
    print(b.loc[[r[0] for r in BANK]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
