"""
Figures for the Phase 3 paper. Reads results/analysis/*.csv (made by analyze_results.py)
and writes PNG files to figures/. Run from the repository root:

    python scripts/make_figures.py

Design rules: bars start at zero; rates use the full 0-100% scale and counts the full 0-24 scale;
no dual axes; uncertainty shown where a rate is estimated. The four unlearning methods use a
colour-blind-safe palette (Okabe-Ito) AND different marker shapes, so every figure can also be
read in black and white.
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--inp", default="results/analysis"); ap.add_argument("--out", default="figures")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
serif = [f.name for f in font_manager.fontManager.ttflist if f.name in ("Times New Roman", "Liberation Serif", "DejaVu Serif")]
plt.rcParams.update({"font.family": serif[0] if serif else "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.8, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "legend.frameon": False, "axes.titlesize": 10.5})
MK = {"GradDiff": "o", "NPO": "s", "SimNPO": "^", "RMU": "D"}
COL = {"GradDiff": "#0072B2", "NPO": "#E69F00", "SimNPO": "#009E73", "RMU": "#CC79A7"}
BLUE, LIGHT, GREY = "#0072B2", "#9ecae1", "#bdbdbd"
LAB = {"C1-low": "C1 low\nrelearning", "C1-high": "C1 high\nrelearning", "C2": "C2 4-bit\nquantization",
       "C3": "C3\nparaphrase", "C4": "C4 mixed\nprompts"}
CONDS = ["C1-low", "C1-high", "C2", "C3", "C4"]
T = pd.read_csv(f"{a.inp}/checkpoint_summary.csv")
AP = T[T.metric == "answer_probability"]
def pts(ax, x, y, me, s=34, label=None):
    ax.scatter(x, y, marker=MK[me], s=s, facecolor=COL[me], edgecolor="black", linewidth=0.6, label=label, zorder=3)

# ---- Figure 1: baseline verdicts - how many checkpoints each metric alone calls forgotten
BV = pd.read_csv(f"{a.inp}/baseline_verdicts.csv").sort_values("n_K_forgotten")
nm = {"answer_probability": "Answer probability", "extraction_strength": "Extraction Strength", "exact_memorization": "Exact Memorization",
      "rouge_l": "ROUGE-L", "truth_ratio": "Truth Ratio", "min_k_percent": "Min-K% Prob"}
yy = np.arange(len(BV))
fig, ax = plt.subplots(figsize=(6.3, 2.9))
ax.barh(yy, BV.n_K_forgotten, height=0.55, color=BLUE, edgecolor="black", linewidth=0.6, label="K ≤ 0.10", zorder=2)
ax.barh(yy, BV.n_verdict_forgotten, height=0.55, color="black", edgecolor="black", linewidth=0.6, label="K ≤ 0.10 and utility ≥ 90%", zorder=3)
for y0, r in zip(yy, BV.itertuples()):
    ax.text(r.n_K_forgotten + 0.3, y0, f"{int(r.n_K_forgotten)}  (with utility: {int(r.n_verdict_forgotten)})", va="center", fontsize=8.5)
ax.set_yticks(yy); ax.set_yticklabels([nm[m] for m in BV.metric]); ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 4))
ax.set_xlabel("Checkpoints judged forgotten before any condition (out of 24)")
ax.xaxis.grid(True, color="#d9d9d9", lw=0.6); ax.set_axisbelow(True)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8.5)
fig.savefig(f"{a.out}/fig1_baseline_verdicts_by_metric.png"); plt.close(fig)

# ---- Figure 2: forgetting versus utility before any condition
b = AP[AP.condition == "C1-high"]
fig, ax = plt.subplots(figsize=(6.3, 3.9))
ax.axhspan(0.90, 1.08, xmin=0, xmax=(0.10 + 0.25) / 1.30, color="#e6e6e6", zorder=0)
ax.axvline(0.10, color="black", lw=0.8, ls="--"); ax.axhline(0.90, color="black", lw=0.8, ls="--")
for me in MK:
    g = b[b.method == me]; pts(ax, g.K_before, g.utility_ratio, me, s=44, label=me)
ax.text(-0.238, 0.99, "forgotten and useful:\nno checkpoint", fontsize=8, va="center")
ax.text(0.115, 0.03, "K = 0.10", fontsize=8); ax.text(0.62, 0.865, "utility = 90%", fontsize=8)
ax.set_xlim(-0.25, 1.05); ax.set_ylim(0, 1.08)
ax.set_xlabel("Residual knowledge K before any condition (answer probability)")
ax.set_ylabel("Utility relative to original model")
ax.legend(title="Unlearning method", loc="lower right", fontsize=8.5, title_fontsize=8.5)
fig.savefig(f"{a.out}/fig2_forgetting_vs_utility.png"); plt.close(fig)

# ---- Figure 3: verdict-flip rate per condition
F = pd.read_csv(f"{a.inp}/flip_rates.csv")
rows = [F[(F.metric == "answer_probability") & (F.condition == c)].iloc[0] for c in CONDS[:4]]
rows.append(F[(F.metric == "rouge_l") & (F.condition == "C4")].iloc[0])
fig, ax = plt.subplots(figsize=(6.3, 3.3))
x = np.arange(len(rows)); y = np.array([r.flip_rate for r in rows]) * 100
lo = np.array([r.wilson_lo for r in rows]) * 100; hi = np.array([r.wilson_hi for r in rows]) * 100
ax.bar(x, y, width=0.5, color=[BLUE] * 4 + [GREY], edgecolor="black", linewidth=0.6, zorder=2)
ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor="black", elinewidth=0.9, capsize=3, zorder=3)
for i, r in enumerate(rows):
    ax.text(i + 0.07, hi[i] + 2, f"{int(r.n_flipped)}/{int(r.n_forgotten)}", ha="left", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([LAB[r.condition] for r in rows], fontsize=8.5)
ax.set_ylim(0, 100); ax.set_ylabel("Verdict-flip rate (%)"); ax.yaxis.grid(True, color="#d9d9d9", lw=0.6); ax.set_axisbelow(True)
ax.text(0.99, 0.97, "Blue bars: answer probability.  Grey bar: ROUGE-L\n(answer probability is not available for C4).\nError bars: Wilson 95% interval.",
        transform=ax.transAxes, ha="right", va="top", fontsize=8)
fig.savefig(f"{a.out}/fig3_flip_rate_by_condition.png"); plt.close(fig)

# ---- Figure 4: control-adjusted recovery G for every checkpoint and condition
fig, ax = plt.subplots(figsize=(6.3, 3.6)); off = {"GradDiff": -0.27, "NPO": -0.09, "SimNPO": 0.09, "RMU": 0.27}
for i, c in enumerate(CONDS):
    g = T[(T.metric == ("rouge_l" if c == "C4" else "answer_probability")) & (T.condition == c)]
    for me in MK:
        gg = g[g.method == me]
        pts(ax, i + off[me] + np.linspace(-0.05, 0.05, len(gg)), gg.G, me, s=26, label=me if i == 0 else None)
ax.axhline(0.5, color="black", lw=0.8, ls="--"); ax.axhline(0, color="black", lw=0.6)
ax.text(4.45, 0.52, "recovered (G ≥ 0.50)", ha="right", fontsize=8)
ax.set_xticks(range(len(CONDS))); ax.set_xticklabels([LAB[c] for c in CONDS], fontsize=8.5)
ax.set_ylabel("Control-adjusted recovery G"); ax.set_ylim(-0.8, 0.8)
ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8.5, handletextpad=0.2, columnspacing=1.2)
fig.savefig(f"{a.out}/fig4_recovery_G_by_condition.png"); plt.close(fig)

# ---- Figure 5: low versus high relearning budget, K before and after, K-forgotten checkpoints
lo_ = AP[(AP.condition == "C1-low") & AP.forgotten_K].set_index("ck"); hi_ = AP[(AP.condition == "C1-high")].set_index("ck").loc[lo_.index]
fig, ax = plt.subplots(figsize=(6.3, 3.9)); yy = np.arange(len(lo_))[::-1]
for y0, ck in zip(yy, lo_.index):
    ax.plot([lo_.K_before[ck], hi_.K_after[ck]], [y0, y0], color="#8c8c8c", lw=1, zorder=1)
ax.scatter(lo_.K_before, yy, marker="o", s=34, facecolor="white", edgecolor="black", linewidth=0.8, label="Before", zorder=3)
ax.scatter(lo_.K_after, yy, marker="s", s=30, facecolor=LIGHT, edgecolor="black", linewidth=0.8, label="After low budget (20 samples, 1 epoch)", zorder=3)
ax.scatter(hi_.K_after, yy, marker="D", s=30, facecolor=BLUE, edgecolor="black", linewidth=0.8, label="After high budget (40 samples, 3 epochs)", zorder=3)
ax.axvline(0.10, color="black", lw=0.8, ls="--"); ax.text(0.105, len(lo_) - 0.35, "K = 0.10", fontsize=8)
ax.set_yticks(yy); ax.set_yticklabels([c.replace("_0", " ") for c in lo_.index], fontsize=8.5)
ax.set_xlabel("Residual knowledge K (answer probability)"); ax.set_ylim(-0.7, len(lo_) + 0.2)
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.72), fontsize=8.5)
fig.savefig(f"{a.out}/fig5_low_vs_high_relearning.png"); plt.close(fig)

# ---- Figure 6: Spearman correlation between each pre-condition metric and recovery (diverging scale, grey midpoint)
P = pd.read_csv(f"{a.inp}/metric_predictiveness.csv")
order = ["answer_probability", "extraction_strength", "exact_memorization", "rouge_l", "min_k_percent", "truth_ratio", "token_entropy"]
names = ["Answer probability", "Extraction Strength", "Exact Memorization", "ROUGE-L", "Min-K% Prob", "Truth Ratio", "Token entropy"]
M = P.pivot(index="predictor", columns="condition", values="spearman").loc[order, CONDS]
div = LinearSegmentedColormap.from_list("div", ["#b35806", "#f1a340", "#eeeeee", "#67a9cf", "#2166ac"])
fig, ax = plt.subplots(figsize=(6.6, 3.4))
im = ax.imshow(M.values, cmap=div, vmin=-1, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M.values[i, j]
        ax.text(j, i, f"{v:+.2f}".replace("-", "−"), ha="center", va="center", fontsize=9, color="white" if abs(v) > 0.7 else "black")
ax.set_xticks(range(len(CONDS))); ax.set_xticklabels([LAB[c] for c in CONDS], fontsize=8.5)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_xticks(np.arange(-.5, len(CONDS)), minor=True); ax.set_yticks(np.arange(-.5, len(names)), minor=True)
ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="both", length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[-1, -0.5, 0, 0.5, 1]); cb.set_label("Spearman ρ", fontsize=9); cb.outline.set_visible(False)
fig.savefig(f"{a.out}/fig6_spearman_metric_by_condition.png"); plt.close(fig)

# ---- Figure 7: three pre-condition metrics against recovery under high-budget relearning
hi7 = AP[AP.condition == "C1-high"].set_index("ck")
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.7), sharey=True)
for ax, (m, lab) in zip(axes, [("answer_probability", "Answer probability"), ("truth_ratio", "Truth Ratio"), ("token_entropy", "Token entropy")]):
    kb = T[(T.metric == m) & (T.condition == "C1-high")].set_index("ck").K_before.reindex(hi7.index)
    for me in MK:
        sel = hi7.method == me; pts(ax, kb[sel], hi7.G[sel], me, s=30, label=me)
    ax.axhline(0.5, color="black", lw=0.8, ls="--"); ax.axhline(0, color="black", lw=0.5)
    rho = stats.spearmanr(kb, hi7.G)[0]
    ax.set_title(f"{lab}\nρ = {rho:+.2f}".replace("-", "−"), fontsize=9.5)
    ax.set_xlabel("K before the condition", fontsize=9)
axes[0].set_ylabel("Recovery G (C1-high)"); axes[0].set_ylim(-0.2, 0.8)
axes[2].set_xscale("symlog", linthresh=1.0); axes[2].set_xlabel("K before (symmetric log scale)", fontsize=9)
axes[0].legend(fontsize=7.5, loc="upper right", handletextpad=0.2)
fig.tight_layout(); fig.savefig(f"{a.out}/fig7_metric_vs_recovery_scatter.png"); plt.close(fig)

# ---- Figure 8: ablations A1 and A6 - recovered checkpoints under C1-high by threshold, with and without the control adjustment
R = pd.read_csv(f"{a.inp}/recovery_by_condition.csv"); R = R[R.condition == "C1-high"].sort_values("threshold")
fig, ax = plt.subplots(figsize=(6.3, 2.9)); x = np.arange(len(R)); w = 0.32
b1 = ax.bar(x - w / 2, R.n_recovered, width=w, color=BLUE, edgecolor="black", linewidth=0.6, label="With control adjustment (G)", zorder=2)
b2 = ax.bar(x + w / 2, R.n_recovered_no_control, width=w, color="white", edgecolor="black", linewidth=0.6, hatch="///", label="Without control adjustment", zorder=2)
for bars in (b1, b2):
    for bb in bars: ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.4, f"{int(bb.get_height())}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([f"G ≥ {t:.2f}" for t in R.threshold]); ax.set_ylim(0, 24); ax.set_yticks(range(0, 25, 4))
ax.set_ylabel("Recovered checkpoints (out of 24)"); ax.set_xlabel("Recovery threshold, high-budget relearning (C1-high)")
ax.yaxis.grid(True, color="#d9d9d9", lw=0.6); ax.set_axisbelow(True); ax.legend(loc="upper right", fontsize=8.5)
fig.savefig(f"{a.out}/fig8_ablation_control_and_threshold.png"); plt.close(fig)

# ---- Figure 9: ablation A5 - recovery measured on the relearning authors (part A) against the held-out authors (part B)
p5 = f"{a.inp}/ablation_A5_partA_vs_partB.csv"
if os.path.exists(p5):
    A5 = pd.read_csv(p5); A5 = A5[A5.condition == "C1-high"]
    fig, ax = plt.subplots(figsize=(3.9, 3.7))
    ax.plot([-0.2, 0.8], [-0.2, 0.8], color="#8c8c8c", lw=0.9, ls="--", zorder=1); ax.text(0.33, 0.22, "equal recovery", fontsize=8, rotation=45, color="#555555")
    for me in MK:
        g = A5[A5.method == me]; pts(ax, g.G_partB, g.G_partA, me, s=36, label=me)
    ax.set_xlim(-0.2, 0.8); ax.set_ylim(-0.2, 0.8); ax.set_aspect("equal")
    ax.set_xlabel("G on held-out authors (part B)"); ax.set_ylabel("G on relearning authors (part A)")
    ax.legend(fontsize=8, loc="upper left", handletextpad=0.2)
    fig.savefig(f"{a.out}/fig9_ablation_A5_partA_vs_partB.png"); plt.close(fig)
print("figures written to", a.out)
