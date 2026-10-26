"""Statistical analysis + figures for all hypotheses. Reads results/rule_scores.parquet (from score.py).

Outputs: results/tables/*.csv, results/stats.json, figures/*.png
Uncertainty: item-clustered bootstrap (resample the 100 user queries; 2000 resamples, seed 0) unless noted.
"""
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy import stats

from rules import BANK, CONFLICTS

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "results", "tables")
FIG = os.path.join(ROOT, "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
B_RESAMPLES = 2000
RNG = np.random.default_rng(0)

# Model metadata: display order, parameter count (B), family
MODELS = {
    "qwen2.5-0.5b": (0.5, "Qwen2.5"), "llama3.2-1b": (1.2, "Llama-3"), "qwen2.5-1.5b": (1.5, "Qwen2.5"),
    "qwen2.5-3b": (3.1, "Qwen2.5"), "llama3.2-3b": (3.2, "Llama-3"), "qwen2.5-7b": (7.6, "Qwen2.5"),
    "llama3.1-8b": (8.0, "Llama-3"), "qwen3-8b": (8.2, "Qwen3"), "qwen2.5-14b": (14.7, "Qwen2.5"),
}
BANK_IDS = [r[0] for r in BANK]


def size_class(m):
    p = MODELS[m][0]
    return "<3B" if p < 3 else ("3B" if p < 7 else ">=7B")


def boot_items(df, stat, n=B_RESAMPLES):
    """Item-clustered bootstrap of stat(df) -> (point, lo, hi). stat takes a dataframe."""
    items = df["item"].unique()
    groups = {i: g for i, g in df.groupby("item")}
    point = stat(df)
    vals = []
    for _ in range(n):
        samp = RNG.choice(items, len(items), replace=True)
        d = pd.concat([groups[i] for i in samp], ignore_index=True)
        vals.append(stat(d))
    vals = np.array(vals)
    return point, np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5), vals


def item_boot_means(per_item, n=B_RESAMPLES):
    """Fast bootstrap for statistics that are means over items: per_item is an (items x k) array of per-item
    means; returns bootstrap distribution of column means (n x k)."""
    idx = RNG.integers(0, per_item.shape[0], size=(n, per_item.shape[0]))
    return np.nanmean(per_item[idx], axis=1)


def main():
    df = pd.read_parquet(os.path.join(ROOT, "results", "rule_scores.parquet"))
    df = df[df.model.isin(MODELS)]
    meta_all = pd.read_parquet(os.path.join(ROOT, "results", "generations_meta.parquet"))
    # Each experiment uses the models whose generations for that experiment are complete (partial files from
    # still-running jobs are skipped).
    EXPECTED = {"A": 900, "B": 600, "D": 1200, "E": 300, "F": 600}
    cnt = meta_all.groupby(["model", "exp"]).uid.nunique()
    complete = {e: [m for m in MODELS if cnt.get((m, e), 0) >= n] for e, n in EXPECTED.items()}
    keep = pd.Series([m in complete[e] for m, e in zip(df.model, df.exp)], index=df.index)
    df = df[keep]
    models = [m for m in MODELS if any(m in v for v in complete.values())]
    print("complete:", complete)
    base = pd.read_csv(os.path.join(ROOT, "results", "base_rates.csv"), index_col=0)
    df["base"] = [base.at[r, m] if (r in base.index and m in base.columns) else np.nan for r, m in zip(df.rule, df.model)]
    out = {"models": models}

    # ------------------------------------------------------------------ Exp A: rule count
    A = df[df.exp == "A"]
    rowsA = []
    for m in complete['A']:
        d = A[A.model == m]
        for n, g in d.groupby("n_rules"):
            per_item = g.groupby("item").followed.mean()
            bm = item_boot_means(per_item.values[:, None])[:, 0]
            allf = g.groupby("item").followed.min()
            lift = (g.followed - g.base).groupby(g.item).mean()
            rowsA.append(dict(model=m, n_rules=n, compliance=per_item.mean(), lo=np.percentile(bm, 2.5),
                              hi=np.percentile(bm, 97.5), all_followed=allf.mean(), base=g.base.mean(),
                              lift=lift.mean(), sys_tokens=g.prompt_tokens.mean()))
    TA = pd.DataFrame(rowsA)
    TA.to_csv(os.path.join(TAB, "expA_count.csv"), index=False)

    # Saturation point: largest N whose compliance is within 5 pp of the model's N<=2 level, and a hinge fit.
    sat = []
    for m in complete['A']:
        t = TA[TA.model == m].set_index("n_rules")
        ref = t.loc[[1, 2], "compliance"].mean()
        d = A[(A.model == m)].copy()
        # hinge regression: compliance ~ a + b1*N + b2*max(0, N-k); pick k minimizing SSE (linear prob. model)
        best = None
        for k in [2, 5, 10, 15, 20, 30]:
            d["h"] = np.maximum(0, d.n_rules - k)
            f = smf.ols("followed ~ n_rules + h + C(rule)", data=d).fit()
            if best is None or f.ssr < best[1]:
                best = (k, f.ssr, f)
        k, _, f = best
        first_drop = next((n for n in [5, 10, 15, 20, 30, 40] if t.loc[n, "compliance"] < ref - 0.05), None)
        # chance-corrected retention: (compliance - base rate) / (reference - base rate), base = N=0 rate of the
        # same rules
        ret = {n: (t.loc[n, "compliance"] - t.loc[n, "base"]) / (ref - t.loc[[1, 2], "base"].mean()) for n in
               [5, 10, 15, 20, 30, 40]}
        n90 = next((n for n in [5, 10, 15, 20, 30, 40] if ret[n] < 0.9), ">40")
        n75 = next((n for n in [5, 10, 15, 20, 30, 40] if ret[n] < 0.75), ">40")
        sat.append(dict(model=m, params_b=MODELS[m][0], size_class=size_class(m), ref_N1_2=ref,
                        comp_N10=t.loc[10, "compliance"], comp_N20=t.loc[20, "compliance"],
                        comp_N40=t.loc[40, "compliance"], first_N_5pp_below_ref=first_drop,
                        retention_N10=ret[10], retention_N20=ret[20], retention_N40=ret[40],
                        first_N_retention_below_90=n90, first_N_retention_below_75=n75,
                        hinge_k=k, slope_before=f.params["n_rules"], slope_after=f.params["n_rules"] + f.params["h"]))
    TS = pd.DataFrame(sat)
    TS.to_csv(os.path.join(TAB, "expA_saturation.csv"), index=False)
    out["saturation"] = TS.to_dict("records")

    # ------------------------------------------------------------------ Exp B: length at fixed N=5
    B = df[df.exp == "B"]
    rowsB, slopes = [], []
    for m in complete['B']:
        d = B[B.model == m]
        for L, g in d.groupby("filler"):
            per_item = g.groupby("item").followed.mean()
            bm = item_boot_means(per_item.values[:, None])[:, 0]
            rowsB.append(dict(model=m, filler=L, sys_tokens=g.prompt_tokens.mean(), compliance=per_item.mean(),
                              lo=np.percentile(bm, 2.5), hi=np.percentile(bm, 97.5),
                              truncated=g.truncated.mean()))
        # per-doubling slope beyond ~500 tokens: OLS on log2(prompt tokens) with rule fixed effects,
        # item-cluster-robust SEs. Relative slope = slope / mean compliance at the ~500-token condition.
        for label, sub in [("all", d), ("ge500", d[d.filler >= 500])]:
            sub = sub.assign(l2=np.log2(sub.prompt_tokens))
            f = smf.ols("followed ~ l2 + C(rule)", data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub.item})
            ref = d[d.filler == 500].followed.mean()
            lg = smf.logit("followed ~ l2 + C(rule)", data=sub).fit(disp=0, cov_type="cluster",
                                                                     cov_kwds={"groups": sub.item}) \
                if sub.followed.nunique() > 1 else None
            slopes.append(dict(model=m, range=label, slope_pp_per_doubling=100 * f.params["l2"],
                               ci_lo=100 * f.conf_int().loc["l2", 0], ci_hi=100 * f.conf_int().loc["l2", 1],
                               p=f.pvalues["l2"], rel_pct_per_doubling=100 * f.params["l2"] / ref,
                               logit_or_per_doubling=float(np.exp(lg.params["l2"])) if lg is not None else np.nan,
                               comp_500=ref, comp_8000=d[d.filler == 8000].followed.mean()))
    TB = pd.DataFrame(rowsB)
    TB.to_csv(os.path.join(TAB, "expB_length.csv"), index=False)
    TBs = pd.DataFrame(slopes)
    TBs["p_holm"] = np.nan
    msk = TBs.range == "ge500"
    TBs.loc[msk, "p_holm"] = multipletests(TBs.loc[msk, "p"], method="holm")[1]
    TBs.to_csv(os.path.join(TAB, "expB_slopes.csv"), index=False)
    # pooled across models (model x rule fixed effects)
    sub = B[B.filler >= 500].assign(l2=lambda x: np.log2(x.prompt_tokens))
    f = smf.ols("followed ~ l2 + C(rule):C(model)" + (" + C(model)" if sub.model.nunique() > 1 else ""), data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub.item})
    out["expB_pooled_ge500"] = dict(slope_pp=100 * f.params["l2"], lo=100 * f.conf_int().loc["l2", 0],
                                    hi=100 * f.conf_int().loc["l2", 1], p=f.pvalues["l2"],
                                    rel_pct=100 * f.params["l2"] / B[B.filler == 500].followed.mean())

    # ------------------------------------------------------------------ Position (A N>=10, B L>=1000, E)
    P = df[((df.exp == "A") & (df.n_rules >= 10)) | ((df.exp == "B") & (df.filler >= 1000)) | (df.exp == "E")].copy()
    P["bin"] = pd.cut(P.pos, [-0.01, 0.2, 0.8, 1.01], labels=["first20", "middle", "last20"])
    P["dec"] = np.minimum((P.pos * 10).astype(int), 9)
    P["src"] = P.exp.map({"A": "list (Exp A)", "B": "interleaved, N=5 (Exp B)", "E": "interleaved, 4k tok (Exp E)"})
    rowsP = []
    for m in models + ["ALL"]:
        for src in ["ALL"] + sorted(P.src.unique()):
            d = P if m == "ALL" else P[P.model == m]
            d = d if src == "ALL" else d[d.src == src]
            if d.empty:
                continue
            # per-item means by bin -> bootstrap ratio of edges to middle
            pi = d.pivot_table(index="item", columns="bin", values="followed", aggfunc="mean", observed=False)
            pi = pi[["first20", "middle", "last20"]].values
            bm = item_boot_means(pi)
            edge = lambda a: (a[..., 0] + a[..., 2]) / 2
            ratio_b = edge(bm) / bm[:, 1]
            pt = np.nanmean(pi, axis=0)
            # regression with rule (and condition) fixed effects, cluster by item
            f = smf.ols("followed ~ C(bin, Treatment('middle')) + C(rule) + C(cond)" +
                        (" + C(model)" if d.model.nunique() > 1 else ""), data=d).fit(
                cov_type="cluster", cov_kwds={"groups": d.item})
            rowsP.append(dict(model=m, source=src, n=len(d), first20=pt[0], middle=pt[1], last20=pt[2],
                              edge_mid_ratio=edge(pt) / pt[1], ratio_lo=np.nanpercentile(ratio_b, 2.5),
                              ratio_hi=np.nanpercentile(ratio_b, 97.5),
                              fe_first_pp=100 * f.params["C(bin, Treatment('middle'))[T.first20]"],
                              fe_first_p=f.pvalues["C(bin, Treatment('middle'))[T.first20]"],
                              fe_last_pp=100 * f.params["C(bin, Treatment('middle'))[T.last20]"],
                              fe_last_p=f.pvalues["C(bin, Treatment('middle'))[T.last20]"]))
    TP = pd.DataFrame(rowsP)
    TP.to_csv(os.path.join(TAB, "position.csv"), index=False)
    dec = P.groupby(["src", "dec"]).followed.mean().unstack(0)
    dec.to_csv(os.path.join(TAB, "position_deciles.csv"))
    decm = P[P.exp == "A"].groupby(["model", "dec"]).followed.mean().unstack(0)  # Exp A: run by every model

    # ------------------------------------------------------------------ Exp D: conflicts
    D = df[df.exp == "D"].copy()
    D["c"] = D.cond.str.split(":").str[1]
    rowsD = []
    for m in complete['D'] + ["ALL"]:
        for pair, (x, y) in CONFLICTS.items():
            for role, rid, alone in [("X", x[0], "X"), ("Y", y[0], "Y")]:
                d = D[(D.pair == pair) & (D.rule == rid)]
                d = d if m == "ALL" else d[d.model == m]
                a = d[d.c == alone].set_index(["model", "item"]).followed
                b = d[d.c == "XY"].set_index(["model", "item"]).followed
                j = pd.concat([a.rename("alone"), b.rename("both")], axis=1).dropna()
                n01 = int(((j.alone == 1) & (j.both == 0)).sum())
                n10 = int(((j.alone == 0) & (j.both == 1)).sum())
                p = stats.binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0
                per_item = j.groupby(level="item").mean().values
                bm = item_boot_means(per_item)
                rel_b = (bm[:, 1] - bm[:, 0]) / bm[:, 0]
                rowsD.append(dict(model=m, pair=pair, role=role, rule=rid, alone=j.alone.mean(), both=j.both.mean(),
                                  diff_pp=100 * (j.both.mean() - j.alone.mean()),
                                  rel_change=(j.both.mean() - j.alone.mean()) / j.alone.mean() if j.alone.mean() else np.nan,
                                  rel_lo=np.nanpercentile(rel_b, 2.5), rel_hi=np.nanpercentile(rel_b, 97.5),
                                  mcnemar_p=p, n=len(j)))
    TD = pd.DataFrame(rowsD)
    msk = TD.model != "ALL"
    TD.loc[msk, "p_holm"] = multipletests(TD.loc[msk, "mcnemar_p"], method="holm")[1]
    TD.to_csv(os.path.join(TAB, "expD_conflict.csv"), index=False)
    # both members followed? and "who wins" in XY
    win = []
    for m in complete['D']:
        for pair, (x, y) in CONFLICTS.items():
            d = D[(D.model == m) & (D.pair == pair) & (D.c == "XY")]
            w = d.pivot_table(index="item", columns="rule", values="followed")
            win.append(dict(model=m, pair=pair, X_only=((w[x[0]] == 1) & (w[y[0]] == 0)).mean(),
                            Y_only=((w[x[0]] == 0) & (w[y[0]] == 1)).mean(),
                            both=((w[x[0]] == 1) & (w[y[0]] == 1)).mean(), neither=((w[x[0]] == 0) & (w[y[0]] == 0)).mean()))
    TW = pd.DataFrame(win)
    TW.to_csv(os.path.join(TAB, "expD_winner.csv"), index=False)
    # background (non-conflict) rules: does adding a conflict hurt the other rules? compare XY vs mean(X,Y)
    bg = D[D.conflict_role.isna()]
    TBG = bg.groupby(["model", "pair", "c"]).followed.mean().unstack(2)
    TBG.to_csv(os.path.join(TAB, "expD_background.csv"))
    # word counts in the soft length pair
    meta = pd.read_parquet(os.path.join(ROOT, "results", "generations_meta.parquet"))
    wc = meta[meta.cond.str.startswith("length")].groupby(["model", "cond"]).words.median().unstack(1)
    wc["N0_baseline"] = meta[meta.cond == "N0"].groupby("model").words.median()
    wc.to_csv(os.path.join(TAB, "expD_wordcounts.csv"))

    # ------------------------------------------------------------------ Exp E: count at ~4k tokens vs Exp A
    E = df[df.exp == "E"]
    rowsE = []
    for m in complete['B']:
        for n in [10, 20, 40]:
            a = A[(A.model == m) & (A.n_rules == n)].followed.mean()
            e = E[(E.model == m) & (E.n_rules == n)].followed.mean() if not E.empty else np.nan
            rowsE.append(dict(model=m, n_rules=n, list_short=a, interleaved_4k=e))
        b5 = B[(B.model == m) & (B.filler == 4000)].followed.mean()
        rowsE.append(dict(model=m, n_rules=5, list_short=A[(A.model == m) & (A.n_rules == 5)].followed.mean(),
                          interleaved_4k=b5))
    TE = pd.DataFrame(rowsE).sort_values(["model", "n_rules"])
    TE.to_csv(os.path.join(TAB, "expE_count_at_4k.csv"), index=False)

    # ------------------------------------------------------------------ Exp F: rule block placement in padded prompts
    F = df[df.exp == "F"].copy()
    F["place"] = F.cond.str.split(":").str[1]
    rowsF = []
    for m in complete['F'] + ["ALL"]:
        d = F if m == "ALL" else F[F.model == m]
        for L in sorted(d.filler.unique()):
            for place in ["top", "middle", "bottom"]:
                g = d[(d.filler == L) & (d.place == place)]
                if g.empty:
                    continue
                per_item = g.groupby("item").followed.mean()
                bm = item_boot_means(per_item.values[:, None])[:, 0]
                rowsF.append(dict(model=m, filler=L, place=place, compliance=per_item.mean(),
                                  lo=np.percentile(bm, 2.5), hi=np.percentile(bm, 97.5)))
            bb = B[(B.filler == L)] if m == "ALL" else B[(B.model == m) & (B.filler == L)]
            bb = bb[bb.model.isin(d.model.unique())]
            rowsF.append(dict(model=m, filler=L, place="interleaved (Exp B)", compliance=bb.followed.mean(),
                              lo=np.nan, hi=np.nan))
    TF = pd.DataFrame(rowsF)
    TF.to_csv(os.path.join(TAB, "expF_block_placement.csv"), index=False)
    # edge (top/bottom) vs middle placement test with rule FE, per filler length, pooled models
    fF = []
    for L in sorted(F.filler.unique()):
        d = F[F.filler == L]
        f = smf.ols("followed ~ C(place, Treatment('middle')) + C(rule)" + (" + C(model)" if d.model.nunique() > 1 else ""),
                    data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d.item})
        for pl in ["top", "bottom"]:
            k = f"C(place, Treatment('middle'))[T.{pl}]"
            fF.append(dict(filler=L, place=pl, vs_middle_pp=100 * f.params[k], lo=100 * f.conf_int().loc[k, 0],
                           hi=100 * f.conf_int().loc[k, 1], p=f.pvalues[k],
                           rel=f.params[k] / d[d.place == "middle"].followed.mean()))
    TFf = pd.DataFrame(fF)
    TFf.to_csv(os.path.join(TAB, "expF_place_tests.csv"), index=False)

    # ------------------------------------------------------------------ per-rule difficulty (Exp A, N=1 vs 40)
    pr = A[A.n_rules > 0].groupby(["rule", "n_rules"]).followed.mean().unstack(1)
    pr["base"] = base.loc[pr.index].mean(axis=1)
    pr.sort_values(1).to_csv(os.path.join(TAB, "per_rule_expA.csv"))

    json.dump(out, open(os.path.join(ROOT, "results", "stats.json"), "w"), indent=1, default=float)

    # ------------------------------------------------------------------ figures
    cmap = plt.get_cmap("viridis")
    col = {m: cmap(i / max(1, len(models) - 1)) for i, m in enumerate(models)}
    ls = {"Qwen2.5": "-", "Llama-3": "--", "Qwen3": ":"}

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for m in models:
        t = TA[(TA.model == m) & (TA.n_rules > 0)]
        ax[0].plot(t.n_rules, t.compliance, ls[MODELS[m][1]], marker="o", color=col[m], label=m)
        ax[0].fill_between(t.n_rules, t.lo, t.hi, color=col[m], alpha=0.12)
        ax[1].plot(t.n_rules, t.all_followed, ls[MODELS[m][1]], marker="o", color=col[m], label=m)
    ax[0].set(xlabel="number of rules in system prompt (N)", ylabel="per-rule compliance",
              title="Exp A: per-rule compliance vs rule count", xscale="log")
    ax[1].set(xlabel="number of rules (N)", ylabel="fraction of responses following ALL rules",
              title="Exp A: all-rules-followed rate", xscale="log")
    for a in ax:
        a.set_xticks([1, 2, 5, 10, 20, 40]); a.set_xticklabels([1, 2, 5, 10, 20, 40]); a.grid(alpha=0.3)
    ax[0].legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "expA_count.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in models:
        t = TB[TB.model == m]
        ax.plot(t.sys_tokens, t.compliance, ls[MODELS[m][1]], marker="o", color=col[m], label=m)
        ax.fill_between(t.sys_tokens, t.lo, t.hi, color=col[m], alpha=0.12)
    ax.set(xscale="log", xlabel="system prompt length (tokens, log scale)", ylabel="per-rule compliance (5 rules)",
           title="Exp B: compliance vs system prompt length (rule count fixed at 5)")
    ax.axvline(500, color="gray", lw=0.8, ls=":")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "expB_length.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for s in dec.columns:
        ax[0].plot((dec.index + 0.5) / 10, dec[s], marker="o", label=s)
    ax[0].set(xlabel="relative position of rule in system prompt", ylabel="compliance (all models pooled)",
              title="Position effect by prompt structure"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    for m in models:
        ax[1].plot((decm.index + 0.5) / 10, decm[m], ls[MODELS[m][1]], marker=".", color=col[m], label=m)
    ax[1].set(xlabel="relative position in rule list", ylabel="compliance (Exp A, N>=10)", title="Position effect by model (numbered list)")
    ax[1].legend(fontsize=7, ncol=2); ax[1].grid(alpha=0.3)
    for a in ax:
        a.axvspan(0, 0.2, color="gray", alpha=0.08); a.axvspan(0.8, 1.0, color="gray", alpha=0.08)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "position.png"), dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, len(CONFLICTS), figsize=(16, 4.2), sharey=True)
    for a, (pair, (x, y)) in zip(axs, CONFLICTS.items()):
        t = TD[(TD.pair == pair) & (TD.model != "ALL")]
        xs = np.arange(len(complete["D"]))
        for k, (role, c) in enumerate([("X", "tab:blue"), ("Y", "tab:orange")]):
            tt = t[t.role == role].set_index("model").loc[complete["D"]]
            a.bar(xs + (k - 0.5) * 0.4, tt.alone, 0.4, color=c, alpha=0.35, label=f"{tt.rule.iloc[0]} alone")
            a.bar(xs + (k - 0.5) * 0.4, tt.both, 0.4, color=c, fill=False, edgecolor=c, hatch="//",
                  label=f"{tt.rule.iloc[0]} + conflict")
        a.set_xticks(xs); a.set_xticklabels(complete["D"], rotation=70, fontsize=7); a.set_title(pair)
        a.legend(fontsize=6)
    axs[0].set_ylabel("compliance")
    fig.suptitle("Exp D: compliance on each rule of a conflicting pair, alone (solid) vs with its contradiction (hatched)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "expD_conflict.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in models:
        t = TE[TE.model == m]
        ax.plot(t.n_rules, t.list_short, ls[MODELS[m][1]], marker="o", color=col[m], label=f"{m}")
        ax.plot(t.n_rules, t.interleaved_4k, ls[MODELS[m][1]], marker="x", color=col[m], alpha=0.6)
    ax.set(xscale="log", xlabel="number of rules", ylabel="per-rule compliance",
           title="Rule count: short list (o) vs interleaved in ~4k-token prompt (x)")
    ax.set_xticks([5, 10, 20, 40]); ax.set_xticklabels([5, 10, 20, 40]); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "expE_count_long.png"), dpi=150); plt.close(fig)

    if not TF.empty:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for a, L in zip(axs, sorted(TF.filler.unique())):
            t = TF[(TF.filler == L) & (TF.model != "ALL")].pivot(index="model", columns="place", values="compliance")
            t = t.loc[[m for m in models if m in t.index], ["top", "middle", "bottom", "interleaved (Exp B)"]]
            t.plot.bar(ax=a, width=0.8, color=["#4c72b0", "#dd8452", "#55a868", "#bbbbbb"])
            a.set(title=f"Exp F: 5-rule block placement, ~{L}-token system prompt", ylabel="per-rule compliance", xlabel="")
            a.tick_params(axis="x", rotation=60); a.grid(alpha=0.3, axis="y"); a.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "expF_placement.png"), dpi=150); plt.close(fig)

    pd.set_option("display.width", 250)
    print("== Exp A\n", TA.pivot(index="n_rules", columns="model", values="compliance").round(3)[complete["A"]].to_string())
    print("== saturation\n", TS.round(3).to_string())
    print("== Exp B\n", TB.pivot(index="filler", columns="model", values="compliance").round(3)[complete["B"]].to_string())
    print("== Exp B slopes\n", TBs.round(3).to_string())
    print("== pooled B", out["expB_pooled_ge500"])
    print("== position\n", TP[TP.source == "ALL"].round(3).to_string())
    print("== position by source (ALL models)\n", TP[TP.model == "ALL"].round(3).to_string())
    print("== Exp D\n", TD[TD.model == "ALL"].round(3).to_string())
    print("== Exp D per model rel change\n", TD[TD.model != "ALL"].pivot_table(index="model", columns=["pair", "role"], values="rel_change").round(2).to_string())
    print("== winners\n", TW.round(2).to_string())
    print("== background\n", TBG.round(3).to_string())
    print("== wordcounts\n", wc.to_string())
    print("== Exp E\n", TE.round(3).to_string())
    print("== Exp F\n", TF.pivot_table(index=["model", "filler"], columns="place", values="compliance").round(3).to_string())
    print("== Exp F tests\n", TFf.round(4).to_string())


if __name__ == "__main__":
    main()
