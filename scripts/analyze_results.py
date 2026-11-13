"""
Analysis for the LLM-unlearning project (Phase 3).
Reads results/per_question/*.csv (script version v2) and writes every table used in the paper
to results/analysis/. Run from the repository root:

    python scripts/analyze_results.py

Definitions follow the Phase 2 methodology (Steps 8 to 10):
  K = (S_model - S_control) / (S_original - S_control)
  G = [(S_after - S_before) - (S_control_after - S_control_before)] / (S_original - S_control)
For C3 (paraphrase) and C4 (mixed prompts), K after the condition uses the original and control
models measured on the same paraphrased / mixed prompts. For C1 and C2 it uses the C0 references.
"""
import argparse, glob, json, os, itertools
import numpy as np, pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--inp", default="results/per_question")
ap.add_argument("--out", default="results/analysis")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
RNG_SEED, NBOOT = 0, 10000
K_THR, G_THR, U_FLOOR = 0.10, 0.50, 0.90
CONDS = ["C1-low", "C1-high", "C2", "C3", "C4"]
MEM = ["answer_probability", "extraction_strength", "exact_memorization", "rouge_l", "truth_ratio", "min_k_percent"]
ALLM = MEM + ["token_entropy"]
METHODS = ["GradDiff", "NPO", "SimNPO", "RMU"]

# ---------------------------------------------------------------- load
d = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(os.path.join(a.inp, "*.csv"))))
assert d.notebook_version.nunique() == 1 and d.config_hash.nunique() == 1, "mixed script/config versions"
d["ck"] = d.method + "_" + d.config_id.astype(int).astype(str).str.zfill(2)
# C4 has two prompt orders per question: average them for the question-level score
q = d.groupby(["ck", "method", "condition", "metric_name", "question_id"], as_index=False).metric_value.mean()

def agg(g):   # truth ratio is heavy-tailed (ratios of small probabilities) -> median; others -> mean
    return g.median() if g.name[2] == "truth_ratio" else g.mean()
S = q.groupby(["ck", "condition", "metric_name"]).metric_value.apply(agg).unstack("condition")
S = S.reset_index().rename(columns={"metric_name": "metric"})
util = d[d.condition != "C4"].groupby(["ck", "condition"]).utility_metric.first().unstack()

def val(ck, m, c):
    r = S[(S.ck == ck) & (S.metric == m)]
    return float(r[c].iloc[0]) if len(r) and c in r and pd.notna(r[c].iloc[0]) else np.nan

O, C = "ORIGINAL_00", "CONTROL_00"
cks = [c for c in sorted(S.ck.unique()) if c not in (O, C)]
method_of = {c: c.rsplit("_", 1)[0] for c in cks}
u0 = util.loc[O, "C0"]

# ---------------------------------------------------------------- reference table
ref = []
for m in ALLM:
    for c in ["C0"] + CONDS:
        o, k = val(O, m, c), val(C, m, c)
        if not np.isnan(o): ref.append(dict(metric=m, condition=c, original=o, control=k, span=o - k))
ref = pd.DataFrame(ref); ref.to_csv(f"{a.out}/reference_models.csv", index=False)

# ---------------------------------------------------------------- K and G per checkpoint, metric, condition
rows = []
for ck in cks:
    for m in ALLM:
        o0, c0, s0 = val(O, m, "C0"), val(C, m, "C0"), val(ck, m, "C0")
        D0 = o0 - c0
        Kb = (s0 - c0) / D0
        for c in CONDS:
            s1 = val(ck, m, c)
            if np.isnan(s1): continue
            o1, c1 = val(O, m, c), val(C, m, c)
            if c in ("C3", "C4"): Ka = (s1 - c1) / (o1 - c1)      # same-prompt references
            else:                 Ka = (s1 - c0) / D0             # C0 references
            Graw = (s1 - s0) / D0
            G = ((s1 - s0) - (c1 - c0)) / D0
            rows.append(dict(ck=ck, method=method_of[ck], metric=m, condition=c, S_before=s0, S_after=s1,
                             original=o0, control=c0, control_after=c1, K_before=Kb, K_after=Ka,
                             G=G, G_no_control=Graw,
                             utility_before=util.loc[ck, "C0"], utility_ratio=util.loc[ck, "C0"] / u0,
                             utility_after=util.loc[ck, c] if c in util.columns else np.nan))
T = pd.DataFrame(rows)
T["forgotten_K"] = T.K_before <= K_THR
T["forgotten_verdict"] = T.forgotten_K & (T.utility_ratio >= U_FLOOR)
T["flip"] = T.forgotten_K & (T.K_after > K_THR)
T["recovered"] = T.G >= G_THR
T.to_csv(f"{a.out}/checkpoint_summary.csv", index=False)

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0, c - h), min(1, c + h))

# ---------------------------------------------------------------- baseline verdicts (each metric alone, before any condition)
base = T[T.condition == "C1-high"]   # one row per ck x metric; K_before does not depend on condition
bl = []
for m in MEM:
    x = base[base.metric == m]
    bl.append(dict(metric=m, original=x.original.iloc[0], control=x.control.iloc[0],
                   span=x.original.iloc[0] - x.control.iloc[0],
                   n_K_forgotten=int(x.forgotten_K.sum()), n_verdict_forgotten=int(x.forgotten_verdict.sum()),
                   checkpoints_K_forgotten=" ".join(x[x.forgotten_K].ck)))
pd.DataFrame(bl).to_csv(f"{a.out}/baseline_verdicts.csv", index=False)

# ---------------------------------------------------------------- SQ1: flip rates
fr = []
for m in MEM:
    for c in CONDS:
        x = T[(T.metric == m) & (T.condition == c)]
        if not len(x): continue
        n, k = int(x.forgotten_K.sum()), int(x.flip.sum()); lo, hi = wilson(k, n)
        fr.append(dict(metric=m, condition=c, n_forgotten=n, n_flipped=k, flip_rate=k / n if n else np.nan,
                       wilson_lo=lo, wilson_hi=hi, flipped=" ".join(x[x.flip].ck)))
FR = pd.DataFrame(fr); FR.to_csv(f"{a.out}/flip_rates.csv", index=False)

# recovered (G >= 0.5) per condition, answer probability (ROUGE-L for C4)
def primary(c): return "rouge_l" if c == "C4" else "answer_probability"
rec = []
for c in CONDS:
    x = T[(T.metric == primary(c)) & (T.condition == c)]
    for thr in (0.25, 0.50, 0.75):
        k = int((x.G >= thr).sum()); kn = int((x.G_no_control >= thr).sum()); lo, hi = wilson(k, len(x))
        rec.append(dict(condition=c, metric=primary(c), threshold=thr, n=len(x), n_recovered=k, rate=k / len(x),
                        wilson_lo=lo, wilson_hi=hi, n_recovered_no_control=kn,
                        recovered=" ".join(x[x.G >= thr].ck), mean_G=x.G.mean(), median_G=x.G.median(),
                        mean_G_no_control=x.G_no_control.mean(), control_shift=(x.G_no_control - x.G).mean()))
REC = pd.DataFrame(rec); REC.to_csv(f"{a.out}/recovery_by_condition.csv", index=False)

# ---------------------------------------------------------------- Cochran's Q and McNemar (exploratory)
def cochran_q(M):
    M = np.asarray(M); k = M.shape[1]; cs, rs, N = M.sum(axis=0), M.sum(axis=1), M.sum()
    den = k * N - (rs ** 2).sum()
    if den == 0: return np.nan, np.nan
    Q = (k - 1) * (k * (cs ** 2).sum() - N ** 2) / den
    return Q, 1 - stats.chi2.cdf(Q, k - 1)
def mcnemar_exact(x, y):
    b, c = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    return b, c, (1.0 if b + c == 0 else stats.binomtest(min(b, c), b + c, 0.5).pvalue)
def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for i, j in enumerate(o):
        run = max(run, (len(p) - i) * p[j]); adj[j] = min(1, run)
    return adj
tests = []
for label, metric, conds, col, subset in [
        ("flip, answer probability (K-forgotten checkpoints)", "answer_probability", ["C1-low", "C1-high", "C2", "C3"], "flip", True),
        ("flip, ROUGE-L (K-forgotten checkpoints)", "rouge_l", CONDS, "flip", True),
        ("recovered G>=0.5, answer probability (all checkpoints)", "answer_probability", ["C1-low", "C1-high", "C2", "C3"], "recovered", False)]:
    x = T[T.metric == metric]
    if subset: x = x[x.forgotten_K]
    M = x.pivot(index="ck", columns="condition", values=col)[conds].astype(int)
    Q, p = cochran_q(M.values)
    tests.append(dict(test="Cochran Q", outcome=label, pair="", n=len(M), stat=Q, p=p, p_holm=np.nan, b=np.nan, c=np.nan))
    pr = [(c1, c2) + mcnemar_exact(M[c1].values, M[c2].values) for c1, c2 in itertools.combinations(conds, 2)]
    adj = holm([r[4] for r in pr])
    for r, pa in zip(pr, adj):
        tests.append(dict(test="McNemar exact", outcome=label, pair=f"{r[0]} vs {r[1]}", n=len(M), stat=np.nan, p=r[4], p_holm=pa, b=r[2], c=r[3]))
pd.DataFrame(tests).to_csv(f"{a.out}/condition_tests.csv", index=False)

# ---------------------------------------------------------------- SQ2: does each pre-condition metric predict recovery?
rng = np.random.default_rng(RNG_SEED)
def auroc(score, y):
    y = np.asarray(y).astype(bool); s = np.asarray(score)
    if y.sum() == 0 or (~y).sum() == 0: return np.nan
    r = stats.rankdata(s); return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())
pre = base.pivot(index="ck", columns="metric", values="K_before")       # pre-condition K for every metric
pred = []; boot_store = {}
for c in CONDS:
    g = T[(T.metric == primary(c)) & (T.condition == c)].set_index("ck").G.reindex(pre.index)
    y = (g >= G_THR).values
    idx = rng.integers(0, len(g), size=(NBOOT, len(g)))
    for m in ALLM:
        x = pre[m].values
        rho, p = stats.spearmanr(x, g.values)
        bs = np.array([stats.spearmanr(x[i], g.values[i])[0] for i in idx]); bs = bs[~np.isnan(bs)]
        ok = y.sum() >= 5 and (~y).sum() >= 5
        au = auroc(x, y) if ok else np.nan
        if ok:
            ba = np.array([auroc(x[i], y[i]) for i in idx]); ba = ba[~np.isnan(ba)]
            alo, ahi = np.percentile(ba, [2.5, 97.5])
        else: alo = ahi = np.nan
        boot_store[(c, m)] = bs
        pred.append(dict(condition=c, recovery_metric=primary(c), predictor=m, n=len(g), spearman=rho,
                         ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), p_value=p,
                         n_recovered=int(y.sum()), auroc=au, auroc_lo=alo, auroc_hi=ahi,
                         auroc_status="computed" if ok else "not computed (<5 in a class)"))
PRED = pd.DataFrame(pred); PRED.to_csv(f"{a.out}/metric_predictiveness.csv", index=False)

# same analysis restricted to the K-forgotten checkpoints (answer probability), C1-high only
fk = base[(base.metric == "answer_probability") & base.forgotten_K].ck
g = T[(T.metric == "answer_probability") & (T.condition == "C1-high")].set_index("ck").G.reindex(fk)
sub = []
for m in ALLM:
    rho, p = stats.spearmanr(pre.loc[fk, m].values, g.values)
    sub.append(dict(predictor=m, n=len(fk), spearman=rho, p_value=p))
pd.DataFrame(sub).to_csv(f"{a.out}/metric_predictiveness_forgotten_only_C1high.csv", index=False)

# ---------------------------------------------------------------- SQ3: by method, and leave-one-method-out
bym = []
for c in CONDS:
    x = T[(T.metric == primary(c)) & (T.condition == c)]
    for me in METHODS:
        y = x[x.method == me]; n = int(y.forgotten_K.sum()); k = int(y.flip.sum()); lo, hi = wilson(k, n)
        bym.append(dict(condition=c, method=me, n=len(y), n_K_forgotten=n, n_flipped=k, flip_lo=lo, flip_hi=hi,
                        n_recovered=int(y.recovered.sum()), mean_G=y.G.mean(), min_G=y.G.min(), max_G=y.G.max()))
pd.DataFrame(bym).to_csv(f"{a.out}/by_method.csv", index=False)
lomo = []
x = T[(T.metric == "answer_probability") & (T.condition == "C1-high")]
for me in ["none"] + METHODS:
    y = x[x.method != me]; n = int(y.forgotten_K.sum()); k = int(y.flip.sum()); lo, hi = wilson(k, n)
    lomo.append(dict(dropped_method=me, n_K_forgotten=n, n_flipped=k, flip_rate=k / n if n else np.nan, wilson_lo=lo, wilson_hi=hi,
                     n_recovered=int(y.recovered.sum()), n=len(y),
                     spearman_K_vs_G=stats.spearmanr(y.K_before, y.G)[0]))
pd.DataFrame(lomo).to_csv(f"{a.out}/leave_one_method_out_C1high.csv", index=False)

# ---------------------------------------------------------------- ablations
# A1 is in recovery_by_condition.csv (with / without control). A2 = C1-low vs C1-high rows there. A6 = thresholds there.
x = T[T.metric == "answer_probability"]
A2 = x[x.condition.isin(["C1-low", "C1-high"])].pivot(index="ck", columns="condition", values="G")
A2["difference"] = A2["C1-high"] - A2["C1-low"]; A2.to_csv(f"{a.out}/ablation_A2_low_vs_high.csv")
# A3: which K-forgotten checkpoints are caught (flip) by which condition; unique catches. Answer probability; C4 by ROUGE-L.
F = x[x.forgotten_K].pivot(index="ck", columns="condition", values="flip").astype(int)
Fr = T[(T.metric == "rouge_l") & T.forgotten_K].pivot(index="ck", columns="condition", values="flip").astype(int)
a3 = []
for label, M in [("answer_probability", F), ("rouge_l", Fr)]:
    conds = list(M.columns); any_all = int((M.sum(axis=1) > 0).sum())
    a3.append(dict(metric=label, dropped="none", n_K_forgotten=len(M), caught_by_any=any_all, uniquely_caught=np.nan))
    for c in conds:
        rest = M.drop(columns=c)
        a3.append(dict(metric=label, dropped=c, n_K_forgotten=len(M), caught_by_any=int((rest.sum(axis=1) > 0).sum()),
                       uniquely_caught=int(((M[c] == 1) & (rest.sum(axis=1) == 0)).sum())))
pd.DataFrame(a3).to_csv(f"{a.out}/ablation_A3_drop_one_condition.csv", index=False)
# A4: single metrics vs mean of two standardised metrics (C0 values) as predictors of G under C1-high
z = (pre[MEM] - pre[MEM].mean()) / pre[MEM].std()
g = T[(T.metric == "answer_probability") & (T.condition == "C1-high")].set_index("ck").G.reindex(pre.index)
a4 = [dict(predictor=m, kind="single", spearman=stats.spearmanr(pre[m], g)[0]) for m in MEM]
for m1, m2 in itertools.combinations(MEM, 2):
    a4.append(dict(predictor=f"{m1} + {m2}", kind="mean of two", spearman=stats.spearmanr((z[m1] + z[m2]) / 2, g)[0]))
A4 = pd.DataFrame(a4); A4["abs_spearman"] = A4.spearman.abs()
A4.sort_values("abs_spearman", ascending=False).to_csv(f"{a.out}/ablation_A4_single_vs_pair.csv", index=False)
# A7: K thresholds and utility floors (answer probability)
a7 = []
for kt in (0.05, 0.10, 0.20):
    for uf in (None, 0.85, 0.90, 0.95):
        b = base[base.metric == "answer_probability"]
        f = b[(b.K_before <= kt) & ((b.utility_ratio >= uf) if uf else True)].ck
        r = dict(K_threshold=kt, utility_floor="none" if uf is None else uf, n_forgotten=len(f), forgotten=" ".join(f))
        for c in ["C1-low", "C1-high", "C2", "C3"]:
            y = x[(x.condition == c) & x.ck.isin(f)]; r[f"flips_{c}"] = int((y.K_after > kt).sum())
        a7.append(r)
pd.DataFrame(a7).to_csv(f"{a.out}/ablation_A7_thresholds.csv", index=False)

# ---------------------------------------------------------------- mixed-prompt order, and item-level analysis (exploratory)
mo = d[d.condition == "C4"].groupby(["ck", "prompt_order"]).metric_value.mean().unstack()
mo["difference"] = mo.forget_first - mo.retain_first; mo.to_csv(f"{a.out}/mixed_prompt_order.csv")

w = q[q.metric_name.isin(ALLM)].pivot_table(index=["ck", "question_id"], columns=["metric_name", "condition"], values="metric_value")
it = []
span = val(O, "answer_probability", "C0") - val(C, "answer_probability", "C0")
ctrl = w.loc[C]
for c in ["C1-low", "C1-high", "C2"]:
    cshift = ctrl[("answer_probability", c)] - ctrl[("answer_probability", "C0")]          # per-question control change
    parts = []
    for ck in cks:
        ww = w.loc[ck]
        gq = ((ww[("answer_probability", c)] - ww[("answer_probability", "C0")]) - cshift.reindex(ww.index)) / span
        f = pd.DataFrame({m: ww[(m, "C0")] for m in ALLM}); f["Gq"] = gq; f["ck"] = ck; parts.append(f)
    I = pd.concat(parts); y = I.Gq >= G_THR
    for m in ALLM:
        it.append(dict(condition=c, predictor=m, n_items=len(I), n_recovered_items=int(y.sum()),
                       auroc=auroc(I[m].values, y.values), spearman=stats.spearmanr(I[m], I.Gq)[0]))
    I.to_csv(f"{a.out}/item_level_{c}.csv")
IT = pd.DataFrame(it); IT["auroc_distance_from_chance"] = (IT.auroc - 0.5).abs()
IT.to_csv(f"{a.out}/item_level_predictiveness.csv", index=False)

# ---------------------------------------------------------------- compact json of headline numbers
b = base[base.metric == "answer_probability"].set_index("ck")
out = dict(n_checkpoints=len(cks), original_utility=float(u0),
           answer_probability=dict(original=val(O, "answer_probability", "C0"), control=val(C, "answer_probability", "C0")),
           n_K_forgotten=int(b.forgotten_K.sum()), n_verdict_forgotten=int(b.forgotten_verdict.sum()),
           K_forgotten=list(b[b.forgotten_K].index), max_utility_ratio_among_K_forgotten=float(b[b.forgotten_K].utility_ratio.max()))
json.dump(out, open(f"{a.out}/headline.json", "w"), indent=1)
print(json.dumps(out, indent=1)); print("tables written to", a.out)
