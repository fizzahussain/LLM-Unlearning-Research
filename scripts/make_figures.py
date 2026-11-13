"""
Figures for the Phase 3 paper. Reads results/analysis/*.csv (made by analyze_results.py)
and writes PNG files to figures/. Run from the repository root:

    python scripts/make_figures.py

All figures are greyscale; unlearning methods are told apart by marker shape.
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ap = argparse.ArgumentParser()
ap.add_argument("--inp", default="results/analysis"); ap.add_argument("--out", default="figures")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
serif = [f.name for f in font_manager.fontManager.ttflist if f.name in ("Times New Roman", "Liberation Serif", "DejaVu Serif")]
plt.rcParams.update({"font.family": serif[0] if serif else "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.8, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "legend.frameon": False, "axes.titlesize": 10.5, "axes.titleweight": "bold"})
MK = {"GradDiff": "o", "NPO": "s", "SimNPO": "^", "RMU": "D"}
FILL = {"GradDiff": "black", "NPO": "white", "SimNPO": "#9a9a9a", "RMU": "#4d4d4d"}
LAB = {"C1-low": "C1 low\nrelearning", "C1-high": "C1 high\nrelearning", "C2": "C2 4-bit\nquantization",
       "C3": "C3\nparaphrase", "C4": "C4 mixed\nprompts"}
T = pd.read_csv(f"{a.inp}/checkpoint_summary.csv")
AP = T[T.metric == "answer_probability"]
def short(ck): return ck.replace("GradDiff", "GD").replace("SimNPO", "Sim").replace("_0", "-")

# ---- Figure 1: forgetting versus utility before any condition
b = AP[AP.condition == "C1-high"]
fig, ax = plt.subplots(figsize=(6.3, 3.9))
ax.axhspan(0.90, 1.08, xmin=0, xmax=(0.10 + 0.25) / 1.30, color="#e6e6e6", zorder=0)
ax.axvline(0.10, color="black", lw=0.8, ls="--"); ax.axhline(0.90, color="black", lw=0.8, ls="--")
for me, g in b.groupby("method"):
    ax.scatter(g.K_before, g.utility_ratio, marker=MK[me], s=42, facecolor=FILL[me], edgecolor="black", linewidth=0.8, label=me, zorder=3)
ax.text(-0.238, 0.99, "forgotten and useful:\nno checkpoint", fontsize=8, va="center")
ax.text(0.115, 0.03, "K = 0.10", fontsize=8); ax.text(0.62, 0.865, "utility = 90%", fontsize=8)
ax.set_xlim(-0.25, 1.05); ax.set_ylim(0, 1.08)
ax.set_xlabel("Residual knowledge K before any condition (answer probability)")
ax.set_ylabel("Utility relative to original model")
ax.legend(title="Unlearning method", loc="lower right", fontsize=8.5, title_fontsize=8.5)
fig.savefig(f"{a.out}/fig1_forgetting_vs_utility.png"); plt.close(fig)

# ---- Figure 2: verdict-flip rate per condition
F = pd.read_csv(f"{a.inp}/flip_rates.csv")
rows = [F[(F.metric == "answer_probability") & (F.condition == c)].iloc[0] for c in ["C1-low", "C1-high", "C2", "C3"]]
rows.append(F[(F.metric == "rouge_l") & (F.condition == "C4")].iloc[0])
fig, ax = plt.subplots(figsize=(6.3, 3.3))
x = np.arange(len(rows)); y = np.array([r.flip_rate for r in rows]) * 100
lo = np.array([r.wilson_lo for r in rows]) * 100; hi = np.array([r.wilson_hi for r in rows]) * 100
ax.bar(x, y, width=0.5, color=["#4d4d4d"] * 4 + ["#b3b3b3"], edgecolor="black", linewidth=0.8, zorder=2)
ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor="black", elinewidth=0.9, capsize=3, zorder=3)
for i, r in enumerate(rows):
    ax.text(i + 0.07, hi[i] + 2, f"{int(r.n_flipped)}/{int(r.n_forgotten)}", ha="left", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([LAB[r.condition] for r in rows], fontsize=8.5)
ax.set_ylim(0, 75); ax.set_ylabel("Verdict-flip rate (%)"); ax.yaxis.grid(True, color="#d9d9d9", lw=0.6); ax.set_axisbelow(True)
ax.text(0.99, 0.97, "Dark bars: answer probability.  Light bar: ROUGE-L\n(answer probability is not available for C4).\nError bars: Wilson 95% interval.",
        transform=ax.transAxes, ha="right", va="top", fontsize=8)
fig.savefig(f"{a.out}/fig2_flip_rate_by_condition.png"); plt.close(fig)

# ---- Figure 3: control-adjusted recovery G for every checkpoint and condition
fig, ax = plt.subplots(figsize=(6.3, 3.6))
conds = ["C1-low", "C1-high", "C2", "C3", "C4"]; off = {"GradDiff": -0.27, "NPO": -0.09, "SimNPO": 0.09, "RMU": 0.27}
for i, c in enumerate(conds):
    g = T[(T.metric == ("rouge_l" if c == "C4" else "answer_probability")) & (T.condition == c)]
    for me, gg in g.groupby("method"):
        jit = np.linspace(-0.05, 0.05, len(gg))
        ax.scatter(i + off[me] + jit, gg.G, marker=MK[me], s=26, facecolor=FILL[me], edgecolor="black", linewidth=0.7,
                   label=me if i == 0 else None, zorder=3)
ax.axhline(0.5, color="black", lw=0.8, ls="--"); ax.axhline(0, color="black", lw=0.6)
ax.text(4.45, 0.52, "recovered (G ≥ 0.50)", ha="right", fontsize=8)
ax.set_xticks(range(len(conds))); ax.set_xticklabels([LAB[c] for c in conds], fontsize=8.5)
ax.set_ylabel("Control-adjusted recovery G"); ax.set_ylim(-0.8, 0.8)
ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8.5, handletextpad=0.2, columnspacing=1.2)
fig.savefig(f"{a.out}/fig3_recovery_G_by_condition.png"); plt.close(fig)

# ---- Figure 4: Spearman correlation between each pre-condition metric and recovery
P = pd.read_csv(f"{a.inp}/metric_predictiveness.csv")
order = ["answer_probability", "extraction_strength", "exact_memorization", "rouge_l", "min_k_percent", "truth_ratio", "token_entropy"]
names = ["Answer probability", "Extraction Strength", "Exact Memorization", "ROUGE-L", "Min-K% Prob", "Truth Ratio", "Token entropy"]
M = P.pivot(index="predictor", columns="condition", values="spearman").loc[order, conds]
fig, ax = plt.subplots(figsize=(6.3, 3.4))
ax.imshow(np.abs(M.values), cmap="Greys", vmin=0, vmax=1.15, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M.values[i, j]
        ax.text(j, i, f"{v:+.2f}".replace("-", "−"), ha="center", va="center", fontsize=9, color="white" if abs(v) > 0.6 else "black")
ax.set_xticks(range(len(conds))); ax.set_xticklabels([LAB[c] for c in conds], fontsize=8.5)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
for s in ax.spines.values(): s.set_visible(False)
ax.set_xticks(np.arange(-.5, len(conds)), minor=True); ax.set_yticks(np.arange(-.5, len(names)), minor=True)
ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="both", length=0)
fig.savefig(f"{a.out}/fig4_spearman_metric_by_condition.png"); plt.close(fig)

# ---- Figure 5: low versus high relearning budget, K before and after, K-forgotten checkpoints
lo_ = AP[(AP.condition == "C1-low") & AP.forgotten_K].set_index("ck"); hi_ = AP[(AP.condition == "C1-high")].set_index("ck").loc[lo_.index]
fig, ax = plt.subplots(figsize=(6.3, 3.9))
yy = np.arange(len(lo_))[::-1]
for y0, ck in zip(yy, lo_.index):
    ax.plot([lo_.K_before[ck], hi_.K_after[ck]], [y0, y0], color="#8c8c8c", lw=1, zorder=1)
ax.scatter(lo_.K_before, yy, marker="o", s=34, facecolor="white", edgecolor="black", linewidth=0.8, label="Before", zorder=3)
ax.scatter(lo_.K_after, yy, marker="s", s=30, facecolor="#9a9a9a", edgecolor="black", linewidth=0.8, label="After low budget (20 samples, 1 epoch)", zorder=3)
ax.scatter(hi_.K_after, yy, marker="D", s=30, facecolor="black", edgecolor="black", linewidth=0.8, label="After high budget (40 samples, 3 epochs)", zorder=3)
ax.axvline(0.10, color="black", lw=0.8, ls="--"); ax.text(0.105, len(lo_) - 0.35, "K = 0.10", fontsize=8)
ax.set_yticks(yy); ax.set_yticklabels([c.replace("_0", " ") for c in lo_.index], fontsize=8.5)
ax.set_xlabel("Residual knowledge K (answer probability)"); ax.set_ylim(-0.7, len(lo_) + 0.2)
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.72), fontsize=8.5)
fig.savefig(f"{a.out}/fig5_low_vs_high_relearning.png"); plt.close(fig)
print("figures written to", a.out)
