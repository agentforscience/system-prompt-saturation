"""Build all experimental prompts -> data/prompts.jsonl.

Exp A (count):    N rules in a numbered list, N in {0,1,2,5,10,15,20,30,40}; random subset+order per item.
Exp B (length):   N=5 rules, each a standalone "Response rule:" paragraph inserted at a random paragraph boundary of
                  neutral WikiText-103 reference text; filler budget in {0,500,1000,2000,4000,8000} tokens.
Exp D (conflict): 4 background keyword rules + target rules, for each conflict pair: X-only (+1 neutral rule),
                  Y-only (+1 neutral rule), X+Y. Rule order randomized.
The same 100 user queries (item i -> query i) are used in every cell (paired design).
"""
import json
import os
import random
import re

import pandas as pd
from transformers import AutoTokenizer

from rules import BANK, CONFLICTS, CONFLICT_BASE_POOL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUB = "/data/transformers/hub"
N_ITEMS = 100
SEED = 42
COUNTS = [0, 1, 2, 5, 10, 15, 20, 30, 40]
FILLER_TOKENS = [0, 500, 1000, 2000, 4000, 8000]
E_COUNTS = [10, 20, 40]
E_FILLER = 4000
F_FILLER = [1000, 4000]

PREAMBLE = "You are a helpful assistant."
LIST_HEADER = "Follow these rules in every response:"
B_HEADER = ("You are a helpful assistant. The reference material below is provided for background only. "
            "Rules for your responses are marked \"Response rule:\" and appear throughout this prompt; follow all of them.")


def load_queries(rng):
    snap = os.path.join(HUB, "datasets--yahma--alpaca-cleaned", "snapshots")
    path = os.path.join(snap, os.listdir(snap)[0], "alpaca_data_cleaned.json")
    data = json.load(open(path))
    ok = re.compile(r"^(Explain|Describe|Write|What|How|Why|Give|Tell|Discuss|Summarize|Suggest|Compare|Create|Generate)\b")
    bad = re.compile(r"(sentence|word|rewrite|translate|code|program|function|poem|haiku|list of \d|\d+ words|tweet|"
                     r"classify|title|headline|slogan|edit|grammar|haiku|limerick|acronym|synonym|antonym)", re.I)
    cands = [d["instruction"].strip() for d in data
             if not d["input"].strip() and ok.match(d["instruction"]) and not bad.search(d["instruction"])
             and 30 <= len(d["instruction"]) <= 160 and len(d["output"].split()) >= 80]
    cands = sorted(set(cands))
    return rng.sample(cands, N_ITEMS)


def load_filler_paragraphs():
    snap = os.path.join(HUB, "datasets--Salesforce--wikitext", "snapshots")
    path = os.path.join(snap, os.listdir(snap)[0], "wikitext-103-raw-v1", "validation-00000-of-00001.parquet")
    lines = pd.read_parquet(path)["text"].tolist()
    paras = []
    for l in lines:
        l = l.strip()
        if not l or l.startswith("="):
            continue
        l = re.sub(r" @(.)@ ", r"\1", l)
        l = re.sub(r" ([.,;:)'])", r"\1", l).replace("( ", "(")
        if 60 <= len(l.split()) <= 220 and "!" not in l and "?" not in l:
            paras.append(l)
    return paras


def build_filler(rng, paras, tok, budget):
    """Consecutive WikiText paragraphs (random start) until `budget` tokens."""
    out, n = [], 0
    i = rng.randrange(len(paras))
    while n < budget:
        p = paras[i % len(paras)]
        k = len(tok.encode(p))
        if n + k > budget * 1.05 and out:
            break
        out.append(p)
        n += k
        i += 1
    return out


def rule_list(rule_texts):
    return "\n".join(f"{j + 1}. {t}" for j, t in enumerate(rule_texts))


def interleave(rng, fill, ids, text):
    """Insert each rule as a 'Response rule:' paragraph at a random paragraph boundary of the filler.
    Shuffles `ids` in place (final order = order of appearance). Returns (system prompt, relative char positions)."""
    n = len(ids)
    slots = sorted(rng.choices(range(len(fill) + 1), k=n)) if fill else [0] * n
    rng.shuffle(ids)
    blocks = []
    k = 0
    for s in range(len(fill) + 1):
        while k < n and slots[k] == s:
            blocks.append(("rule", ids[k]))
            k += 1
        if s < len(fill):
            blocks.append(("fill", fill[s]))
    parts, offsets = [B_HEADER], {}
    for kind, v in blocks:
        if kind == "rule":
            offsets[v] = len("\n\n".join(parts)) + 2
            parts.append(f"Response rule: {text[v]}")
        else:
            parts.append(v)
    sysp = "\n\n".join(parts)
    return sysp, [offsets[r] / len(sysp) for r in ids]


def main():
    rng = random.Random(SEED)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    queries = load_queries(rng)
    paras = load_filler_paragraphs()
    print(f"{len(queries)} queries, {len(paras)} filler paragraphs")
    rows = []
    bank_ids = [r[0] for r in BANK]
    text = {r[0]: r[2] for r in BANK}

    # ---- Exp A: rule count
    for n in COUNTS:
        for i in range(N_ITEMS):
            ids = rng.sample(bank_ids, n)
            sysp = PREAMBLE if n == 0 else f"{PREAMBLE} {LIST_HEADER}\n{rule_list([text[r] for r in ids])}"
            # relative position of each rule = index / (n-1) in the list (0 = first, 1 = last)
            pos = [j / (n - 1) if n > 1 else 0.5 for j in range(n)]
            rows.append(dict(exp="A", cond=f"N{n}", n_rules=n, item=i, rules=ids, pos=pos, system=sysp))

    # ---- Exp B: length with fixed N=5
    for L in FILLER_TOKENS:
        for i in range(N_ITEMS):
            ids = rng.sample(bank_ids, 5)
            fill = build_filler(rng, paras, tok, L)
            sysp, pos = interleave(rng, fill, ids, text)
            rows.append(dict(exp="B", cond=f"L{L}", n_rules=5, filler_budget=L, item=i, rules=ids, pos=pos, system=sysp))

    # ---- Exp D: conflicts
    for pair, (x, y) in CONFLICTS.items():
        for i in range(N_ITEMS):
            base = rng.sample(CONFLICT_BASE_POOL, 5)  # 4 background + 1 neutral stand-in
            for cond in ("X", "Y", "XY"):
                ids_txt = [(b, text[b]) for b in base[:4]]
                if cond in ("X", "XY"):
                    ids_txt.append((x[0], x[1]))
                if cond in ("Y", "XY"):
                    ids_txt.append((y[0], y[1]))
                if cond != "XY":
                    ids_txt.append((base[4], text[base[4]]))
                r2 = random.Random(f"{pair}-{i}")  # same order skeleton across the 3 conditions
                r2.shuffle(ids_txt)
                ids = [a for a, _ in ids_txt]
                sysp = f"{PREAMBLE} {LIST_HEADER}\n{rule_list([t for _, t in ids_txt])}"
                pos = [j / (len(ids) - 1) for j in range(len(ids))]
                rows.append(dict(exp="D", cond=f"{pair}:{cond}", pair=pair, item=i, rules=ids, pos=pos, system=sysp))

    # ---- Exp E (added after the first model started; separate RNG so A/B/D are unchanged):
    # rule count at a fixed ~4000-token filler length, rules interleaved as in Exp B.
    rng_e = random.Random(SEED + 1)
    for n in E_COUNTS:
        for i in range(N_ITEMS):
            ids = rng_e.sample(bank_ids, n)
            fill = build_filler(rng_e, paras, tok, E_FILLER)
            sysp, pos = interleave(rng_e, fill, ids, text)
            rows.append(dict(exp="E", cond=f"N{n}", n_rules=n, filler_budget=E_FILLER, item=i, rules=ids, pos=pos, system=sysp))

    # ---- Exp F (added later; separate RNG): N=5 rules as ONE contiguous numbered block placed at the top,
    # middle or bottom of a prompt padded with reference material to ~1000 or ~4000 tokens.
    rng_f = random.Random(SEED + 2)
    for L in F_FILLER:
        for place in ("top", "middle", "bottom"):
            for i in range(N_ITEMS):
                ids = rng_f.sample(bank_ids, 5)
                fill = build_filler(rng_f, paras, tok, L)
                block = f"{LIST_HEADER}\n{rule_list([text[r] for r in ids])}"
                ref = "Reference material (background only):"
                h = len(fill) // 2
                if place == "top":
                    parts = [PREAMBLE + " " + block, ref] + fill
                elif place == "bottom":
                    parts = [PREAMBLE, ref] + fill + [block]
                else:
                    parts = [PREAMBLE, ref] + fill[:h] + [block] + fill[h:]
                sysp = "\n\n".join(parts)
                start = sysp.index(block)
                pos = [(start + block.index(text[r])) / len(sysp) for r in ids]
                rows.append(dict(exp="F", cond=f"L{L}:{place}", n_rules=5, filler_budget=L, place=place, item=i,
                                 rules=ids, pos=pos, system=sysp))

    for k, r in enumerate(rows):
        r["uid"] = f"{r['exp']}|{r['cond']}|{r['item']}"
        r["query"] = queries[r["item"]]
        r["sys_tokens_qwen"] = len(tok.encode(r["system"]))
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    with open(os.path.join(ROOT, "data", "prompts.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(ROOT, "data", "queries.json"), "w") as f:
        json.dump(queries, f, indent=1)
    df = pd.DataFrame(rows)
    print(df.groupby(["exp", "cond"]).sys_tokens_qwen.agg(["count", "mean", "min", "max"]).to_string())


if __name__ == "__main__":
    main()
