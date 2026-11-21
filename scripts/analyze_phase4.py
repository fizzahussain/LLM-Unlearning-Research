"""
Phase 4 analyses. Run from the repository root after analyze_results.py:

    python scripts/analyze_phase4.py

Reads
  results/per_question/              main experiment, relearning seed 0 (Phase 3)
  results/per_question_seed1, _seed2 relearning repeated with seeds 1 and 2 (C1-low, C1-high only)
  results/per_question_textcheck/    re-run of 8 models at C0 and C1-high, used as a reproducibility check
  results/generations/               generated answers of those 8 models
  results/general_utility/           general-knowledge utility at C0 (optional; skipped if absent)
  data/openunlearning_eval/          OpenUnlearning's published evaluation logs for the two reference models
Writes every table to results/analysis_phase4/.

Definitions are those of analyze_results.py:
  K = (S - S_control) / (S_original - S_control), K-forgotten if K <= 0.10
  G = [(S_after - S_before) - (S_control_after - S_control_before)] / (S_original - S_control), recovered if G >= 0.50
Conditions C0, C2, C3 and C4 involve no training and greedy decoding, so they do not depend on the seed;
only C1-low and C1-high are repeated.
"""
import argparse, glob, json, os, re
import numpy as np, pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--out", default="results/analysis_phase4")
a = ap.parse_args()
R = lambda p: os.path.join(a.root, p)
os.makedirs(R(a.out), exist_ok=True)
OUT = lambda f: os.path.join(R(a.out), f)
K_THR, G_THR, U_FLOOR = 0.10, 0.50, 0.90
O, C = "ORIGINAL_00", "CONTROL_00"
MEM = ["answer_probability", "extraction_strength", "exact_memorization", "rouge_l", "truth_ratio", "min_k_percent"]
ALLM = MEM + ["token_entropy"]
METHODS = ["GradDiff", "NPO", "SimNPO", "RMU"]

def load(folder):
    fs = sorted(glob.glob(R(os.path.join(folder, "*.csv"))))
    if not fs: return None
    d = pd.concat(pd.read_csv(f) for f in fs)
    d["ck"] = d.method + "_" + d.config_id.astype(int).astype(str).str.zfill(2)
    return d

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))

def scores(d):
    """checkpoint x condition mean score per metric (median for truth_ratio, as in analyze_results.py)"""
    q = d.groupby(["ck", "condition", "metric_name", "question_id"], as_index=False).metric_value.mean()
    med = q[q.metric_name == "truth_ratio"].groupby(["ck", "condition", "metric_name"]).metric_value.median()
    mean = q[q.metric_name != "truth_ratio"].groupby(["ck", "condition", "metric_name"]).metric_value.mean()
    return pd.concat([mean, med])

main = load("results/per_question")
assert main is not None and main.notebook_version.nunique() == 1
S0 = scores(main)
cks = sorted(c for c in main.ck.unique() if c not in (O, C))
method_of = {c: c.rsplit("_", 1)[0] for c in cks}
util = main[main.condition != "C4"].groupby(["ck", "condition"]).utility_metric.first().unstack()
u_ratio = util["C0"] / util.loc[O, "C0"]

# ------------------------------------------------------------------ 1. relearning seeds
seed_data = {0: main[main.condition.isin(["C1-low", "C1-high"])]}
for s in (1, 2):
    d = load(f"results/per_question_seed{s}")
    if d is not None:
        assert d.config_hash.iloc[0] == main.config_hash.iloc[0], "seed runs used a different configuration"
        seed_data[s] = d
rows = []
for s, d in seed_data.items():
    Ss = scores(d)
    for m in ALLM:
        o0, c0 = S0[(O, "C0", m)], S0[(C, "C0", m)]; D = o0 - c0
        for cond in ("C1-low", "C1-high"):
            c1 = Ss[(C, cond, m)]
            for ck in cks:
                b, aft = S0[(ck, "C0", m)], Ss[(ck, cond, m)]
                rows.append(dict(seed=s, metric=m, condition=cond, ck=ck, method=method_of[ck], K_before=(b - c0) / D,
                                 K_after=(aft - c0) / D, G=((aft - b) - (c1 - c0)) / D, control_gain=(c1 - c0) / D))
SD = pd.DataFrame(rows)
SD["forgotten_K"] = SD.K_before <= K_THR
SD["flip"] = SD.forgotten_K & (SD.K_after > K_THR)
SD["recovered"] = SD.G >= G_THR
SD.to_csv(OUT("seeds_checkpoint_level.csv"), index=False)

summ = []
for (s, m, cond), x in SD.groupby(["seed", "metric", "condition"]):
    n, k = int(x.forgotten_K.sum()), int(x.flip.sum()); lo, hi = wilson(k, n)
    summ.append(dict(seed=s, metric=m, condition=cond, n_forgotten=n, n_flipped=k, flip_rate=k / n if n else np.nan,
                     wilson_lo=lo, wilson_hi=hi, n_recovered=int(x.recovered.sum()), mean_G=x.G.mean(),
                     control_gain=x.control_gain.iloc[0], flipped=" ".join(sorted(x[x.flip].ck)),
                     recovered=" ".join(sorted(x[x.recovered].ck))))
SS = pd.DataFrame(summ); SS.to_csv(OUT("seeds_summary.csv"), index=False)
agg = SS.groupby(["metric", "condition"]).agg(seeds=("seed", "count"), flip_mean=("n_flipped", "mean"), flip_sd=("n_flipped", "std"),
        flip_min=("n_flipped", "min"), flip_max=("n_flipped", "max"), n_forgotten=("n_forgotten", "first"),
        flip_rate_mean=("flip_rate", "mean"), flip_rate_sd=("flip_rate", "std"),
        recovered_mean=("n_recovered", "mean"), recovered_sd=("n_recovered", "std"),
        mean_G_mean=("mean_G", "mean"), mean_G_sd=("mean_G", "std"),
        control_gain_mean=("control_gain", "mean"), control_gain_sd=("control_gain", "std")).reset_index()
agg.to_csv(OUT("seeds_aggregate.csv"), index=False)

# per-checkpoint G over seeds (answer probability) and between-seed agreement
gp = SD[SD.metric == "answer_probability"].pivot_table(index=["condition", "ck"], columns="seed", values="G")
gp["mean"] = gp.mean(axis=1); gp["sd"] = gp[[c for c in gp.columns if isinstance(c, (int, np.integer))]].std(axis=1)
gp.reset_index().to_csv(OUT("seeds_G_per_checkpoint.csv"), index=False)
agree = []
for cond in ("C1-low", "C1-high"):
    w = gp.loc[cond][[c for c in gp.columns if isinstance(c, (int, np.integer))]]
    for i in w.columns:
        for j in w.columns:
            if i < j: agree.append(dict(condition=cond, seed_a=i, seed_b=j, spearman=stats.spearmanr(w[i], w[j])[0],
                                        max_abs_diff=(w[i] - w[j]).abs().max()))
pd.DataFrame(agree).to_csv(OUT("seeds_agreement.csv"), index=False)

# ------------------------------------------------------------------ 2. predictiveness: seeds, and leave-one-checkpoint-out
pre = SD[(SD.seed == 0) & (SD.condition == "C1-high")].pivot(index="ck", columns="metric", values="K_before")
pr = []
for s in seed_data:
    for cond in ("C1-low", "C1-high"):
        g = SD[(SD.seed == s) & (SD.metric == "answer_probability") & (SD.condition == cond)].set_index("ck").G.reindex(pre.index)
        for m in ALLM:
            rho = stats.spearmanr(pre[m], g)[0]
            loo = [stats.spearmanr(pre[m].drop(k), g.drop(k))[0] for k in pre.index]
            pr.append(dict(seed=s, condition=cond, predictor=m, spearman=rho, loo_min=min(loo), loo_max=max(loo),
                           without_RMU_06=stats.spearmanr(pre[m].drop("RMU_06"), g.drop("RMU_06"))[0]))
PR = pd.DataFrame(pr); PR.to_csv(OUT("predictiveness_seeds_and_loo.csv"), index=False)
PR.groupby(["condition", "predictor"]).agg(spearman_mean=("spearman", "mean"), spearman_sd=("spearman", "std"),
        loo_min=("loo_min", "min"), loo_max=("loo_max", "max"), without_RMU_06_mean=("without_RMU_06", "mean")).reset_index() \
  .to_csv(OUT("predictiveness_seed_aggregate.csv"), index=False)

# bootstrap 95% intervals for Spearman (seed 0, C1-high), for the dot plot
rng = np.random.default_rng(0); idx = rng.integers(0, len(pre), size=(10000, len(pre)))
g0 = SD[(SD.seed == 0) & (SD.metric == "answer_probability") & (SD.condition == "C1-high")].set_index("ck").G.reindex(pre.index).values
bt = []
for m in ALLM:
    x = pre[m].values; bs = np.array([stats.spearmanr(x[i], g0[i])[0] for i in idx]); bs = bs[~np.isnan(bs)]
    bt.append(dict(predictor=m, spearman=stats.spearmanr(x, g0)[0], ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
pd.DataFrame(bt).to_csv(OUT("predictiveness_C1high_bootstrap.csv"), index=False)

# ------------------------------------------------------------------ 3. reproducibility: re-run of 8 models against Phase 3
tc = load("results/per_question_textcheck")
if tc is not None:
    k = ["checkpoint_id", "condition", "question_id", "metric_name"]
    m = tc.merge(main[k + ["metric_value"]], on=k, suffixes=("_rerun", "_phase3"))
    m["abs_diff"] = (m.metric_value_rerun - m.metric_value_phase3).abs()
    pd.DataFrame([dict(models=tc.ck.nunique(), conditions=" ".join(sorted(tc.condition.unique())), values_compared=len(m),
                       max_abs_diff=m.abs_diff.max(), mean_abs_diff=m.abs_diff.mean(),
                       identical_share=(m.abs_diff == 0).mean())]).to_csv(OUT("reproducibility_rerun.csv"), index=False)

# ------------------------------------------------------------------ 4. external validation against OpenUnlearning's own logs
ou_dir = R("data/openunlearning_eval")
pa = load("results/per_question_partA")
if os.path.isdir(ou_dir) and pa is not None:
    val = []
    pairs = [("answer_probability", "forget_Q_A_Prob", "prob"), ("exact_memorization", "exact_memorization", "score"),
             ("extraction_strength", "extraction_strength", "score"), ("rouge_l", "forget_Q_A_ROUGE", "rougeL_recall"),
             ("truth_ratio", "forget_truth_ratio", "score")]
    for model, sub in [("ORIGINAL", "tofu_Llama-3.2-1B-Instruct_full/evals_forget10"), ("CONTROL", "tofu_Llama-3.2-1B-Instruct_retain90")]:
        E = json.load(open(os.path.join(ou_dir, sub, "TOFU_EVAL.json")))
        ours = pd.concat([main[(main.method == model) & (main.condition == "C0")], pa[(pa.method == model) & (pa.condition == "C0")]])
        ours = ours.assign(idx=ours.question_id.str[4:].astype(int).astype(str))
        for mine, theirs, key in pairs:
            o = ours[ours.metric_name == mine].set_index("idx").metric_value
            t = pd.Series({i: v[key] for i, v in E[theirs]["value_by_index"].items() if v and v.get(key) is not None})
            j = pd.concat([o, t], axis=1, keys=["ours", "ou"]).dropna()
            val.append(dict(model=model, metric=mine, n_questions=len(j), ours_mean=j.ours.mean(), openunlearning_mean=j.ou.mean(),
                            pearson=stats.pearsonr(j.ours, j.ou)[0], mean_abs_diff=(j.ours - j.ou).abs().mean()))
        val.append(dict(model=model, metric="model_utility (OpenUnlearning)", n_questions=np.nan, ours_mean=np.nan,
                        openunlearning_mean=E["model_utility"]["agg_value"], pearson=np.nan, mean_abs_diff=np.nan))
    pd.DataFrame(val).to_csv(OUT("validation_vs_openunlearning.csv"), index=False)

# ------------------------------------------------------------------ 5. question types (keyword rule, applied in this order)
RULES = [("name", r"full name|name of|what is the name|who is the|named"),
         ("birth", r"\bborn\b|birth|date of birth|birthplace"),
         ("family", r"father|mother|parent"),
         ("awards", r"award|prize|honou?r|recogni"),
         ("genre", r"genre"),
         ("works", r"book|novel|title|work|publication|written|wrote|series")]
def qtype(q):
    q = q.lower()
    for t, p in RULES:
        if re.search(p, q): return t
    return "other"
qtext = pd.read_csv(R("data/processed/part_b_evaluation.csv"))[["question_id", "question"]]
qtext["qtype"] = qtext.question.map(qtype)
ap0 = main[main.metric_name == "answer_probability"].pivot_table(index=["ck", "question_id"], columns="condition", values="metric_value")
span = S0[(O, "C0", "answer_probability")] - S0[(C, "C0", "answer_probability")]
ctl = ap0.loc[C]; it = []
for ck in cks:
    w = ap0.loc[ck]
    gq = ((w["C1-high"] - w["C0"]) - (ctl["C1-high"] - ctl["C0"]).reindex(w.index)) / span
    it.append(pd.DataFrame(dict(ck=ck, question_id=w.index, Gq=gq.values)))
IT = pd.concat(it).merge(qtext, on="question_id")
flipped = sorted(SD[(SD.seed == 0) & (SD.metric == "answer_probability") & (SD.condition == "C1-high") & SD.flip].ck)
qt = []
for t, x in IT.groupby("qtype"):
    y = x[x.ck.isin(flipped)]
    qt.append(dict(qtype=t, n_questions=x.question_id.nunique(), share_recovered_all=(x.Gq >= G_THR).mean(),
                   share_recovered_flipped=(y.Gq >= G_THR).mean(), mean_Gq_flipped=y.Gq.mean()))
pd.DataFrame(qt).sort_values("share_recovered_flipped", ascending=False).to_csv(OUT("recovery_by_question_type.csv"), index=False)

# ------------------------------------------------------------------ 6. qualitative examples (largest per-question gain, flipped checkpoints)
gen = sorted(glob.glob(R("results/generations/*.csv")))
if gen:
    G_ = pd.concat(pd.read_csv(f) for f in gen)
    G_["model"] = G_.checkpoint_id.str.split("/").str[-1]
    w = G_.pivot_table(index=["model", "question_id"], columns="condition", values="generated_answer", aggfunc="first").reset_index()
    ref = G_.drop_duplicates("question_id").set_index("question_id")[["question", "reference_answer"]]
    w = w.join(ref, on="question_id")
    w.to_csv(OUT("generations_side_by_side.csv"), index=False)
    name = {c: c for c in cks}; ex = []
    for ck in flipped:
        x = IT[IT.ck == ck].sort_values("Gq", ascending=False).head(3)
        for r in x.itertuples(): ex.append(dict(ck=ck, question_id=r.question_id, Gq=r.Gq, qtype=r.qtype))
    pd.DataFrame(ex).to_csv(OUT("qualitative_candidates.csv"), index=False)

# ------------------------------------------------------------------ 7. general-knowledge utility (optional) and refusals
gu = load("results/general_utility")
if gu is not None:
    s = gu.groupby(["ck", "author", "metric_name"]).metric_value.mean().unstack(["author", "metric_name"])
    gk = (s[("real_authors", "answer_probability")] + s[("real_authors", "rouge_l")] + s[("world_facts", "answer_probability")] + s[("world_facts", "rouge_l")]) / 4
    U = pd.DataFrame(dict(general_utility=gk, general_ratio=gk / gk[O], proxy_ratio=u_ratio.reindex(gk.index)))
    U["K_before"] = SD[(SD.seed == 0) & (SD.metric == "answer_probability") & (SD.condition == "C1-high")].set_index("ck").K_before.reindex(U.index)
    U["forgotten_and_useful_proxy"] = (U.K_before <= K_THR) & (U.proxy_ratio >= U_FLOOR)
    U["forgotten_and_useful_general"] = (U.K_before <= K_THR) & (U.general_ratio >= U_FLOOR)
    U.to_csv(OUT("utility_general_vs_proxy.csv"))
    gtxt = sorted(glob.glob(R("results/generations_general/*retain_sample.csv")))
    if gtxt:
        T = pd.concat(pd.read_csv(f) for f in gtxt); T["model"] = T.checkpoint_id.str.split("/").str[-1]
        pat = r"i don'?t know|i'?m not sure|not aware|no (?:information|record|data)|cannot (?:provide|answer)|unable to|sorry"
        T["refusal"] = T.generated_answer.fillna("").str.lower().str.contains(pat, regex=True)
        T.groupby("model").refusal.mean().rename("refusal_rate_retain").to_csv(OUT("refusal_rate_retain.csv"))

# ------------------------------------------------------------------ 8. compute budget from the run logs
logs = sorted(glob.glob(R("logs/*.csv")))
cb = []
for f in logs:
    try:
        l = pd.read_csv(f)
        if {"start_time", "end_time", "status"} <= set(l.columns):
            dur = (pd.to_datetime(l.end_time) - pd.to_datetime(l.start_time)).dt.total_seconds()
            cb.append(dict(log=os.path.basename(f), runs=len(l), ok=int((l.status == "ok").sum()), gpu_hours=dur.sum() / 3600))
    except Exception:
        pass
pd.DataFrame(cb).to_csv(OUT("compute_budget.csv"), index=False)
print("phase-4 tables written to", R(a.out))
