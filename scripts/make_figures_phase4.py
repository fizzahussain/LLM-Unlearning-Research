"""
Figures for the Phase 4 paper. Run from the repository root after analyze_results.py and analyze_phase4.py:

    python scripts/make_figures_phase4.py

Writes vector PDF (for LaTeX) and 300-dpi PNG versions to figures/phase4/.

Design rules (see the paper's "Visualization design" paragraph):
  * every figure answers one question, stated in its caption;
  * values are encoded by position on a common scale (Cleveland and McGill); rates use the full 0-100 % axis;
  * uncertainty is shown wherever a quantity is estimated (Wilson or bootstrap 95 % intervals, seed ranges);
  * no dual axes, no 3-D, no decorative fill; light reference lines only;
  * the four unlearning methods use the Okabe-Ito colours AND different marker shapes with black outlines,
    so the figures remain readable without colour (validated with a CVD palette checker).
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

ap = argparse.ArgumentParser()
ap.add_argument("--p3", default="results/analysis"); ap.add_argument("--p4", default="results/analysis_phase4")
ap.add_argument("--out", default="figures/phase4")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
serif = [f.name for f in font_manager.fontManager.ttflist if f.name in ("Times New Roman", "Nimbus Roman", "TeX Gyre Termes", "Liberation Serif")]
plt.rcParams.update({"font.family": "serif", "font.serif": serif + ["DejaVu Serif"], "font.size": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "legend.frameon": False, "pdf.fonttype": 42, "mathtext.fontset": "stix"})
W1 = 3.45   # IEEE single-column width in inches
MK = {"GradDiff": "o", "NPO": "s", "SimNPO": "^", "RMU": "D"}
COL = {"GradDiff": "#0072B2", "NPO": "#E69F00", "SimNPO": "#009E73", "RMU": "#CC79A7"}
GRID = dict(color="#e3e3e3", lw=0.5)
def save(fig, name):
    fig.savefig(f"{a.out}/{name}.pdf"); fig.savefig(f"{a.out}/{name}.png", dpi=300); plt.close(fig)
def method_handles(filled_note=False):
    h = [Line2D([], [], marker=MK[m], color="none", markerfacecolor=COL[m], markeredgecolor="black", markeredgewidth=0.5, markersize=5, label=m) for m in MK]
    return h

T3 = pd.read_csv(f"{a.p3}/checkpoint_summary.csv")
SD = pd.read_csv(f"{a.p4}/seeds_checkpoint_level.csv")
SS = pd.read_csv(f"{a.p4}/seeds_summary.csv")
F3 = pd.read_csv(f"{a.p3}/flip_rates.csv")

# ------------------------------------------------------------------ Fig. 2: is any checkpoint forgotten AND useful? (two utility measures)
b = T3[(T3.metric == "answer_probability") & (T3.condition == "C1-high")].set_index("ck")
GU = pd.read_csv(f"{a.p4}/utility_general_vs_proxy.csv", index_col=0) if os.path.exists(f"{a.p4}/utility_general_vs_proxy.csv") else None
panels = [("utility_ratio", "(a) TOFU retain questions")] + ([("general", "(b) General knowledge")] if GU is not None else [])
if GU is not None: b["general"] = GU.general_ratio.reindex(b.index)
fig, axs = plt.subplots(1, len(panels), figsize=(W1, 2.35), sharey=True, squeeze=False); axs = axs[0]
for ax, (col, title) in zip(axs, panels):
    ax.fill_between([-0.25, 0.10], 0.90, 1.08, color="#eeeeee", zorder=0, lw=0)
    ax.axvline(0.10, color="black", lw=0.6, ls="--"); ax.axhline(0.90, color="black", lw=0.6, ls="--")
    for m in MK:
        g = b[b.method == m]
        ax.scatter(g.K_before, g[col], marker=MK[m], s=18, c=COL[m], edgecolors="black", linewidths=0.5, zorder=3)
    n = int(((b.K_before <= 0.10) & (b[col] >= 0.90)).sum())
    ax.annotate(f"shaded target:\n{n} of 24", xy=(-0.07, 0.99), xytext=(0.28, 0.50), fontsize=6.5, va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color="#555555"))
    ax.set_xlim(-0.25, 1.05); ax.set_ylim(0, 1.08); ax.set_xticks([0, 0.5, 1.0]); ax.set_title(title, fontsize=7.5, pad=3)
axs[0].set_ylabel("Utility / original model")
axs[-1].legend(handles=method_handles(), loc="lower right", ncol=1, handletextpad=0.1, borderaxespad=0.1, fontsize=6.3)
fig.supxlabel(r"Residual knowledge $K$ before any condition (answer probability)", fontsize=8)
fig.tight_layout(w_pad=0.6, pad=0.2)
save(fig, "fig2_forgetting_vs_utility")

# ------------------------------------------------------------------ Fig. 3: which condition overturns a "forgotten" verdict?
rows = []
for cond, lab in [("C1-low", "C1 relearning, low"), ("C1-high", "C1 relearning, high"), ("C2", "C2 4-bit quantization"),
                  ("C3", "C3 paraphrased questions"), ("C4", "C4 mixed prompts$^{\\dagger}$")]:
    metric = "rouge_l" if cond == "C4" else "answer_probability"
    r = F3[(F3.metric == metric) & (F3.condition == cond)].iloc[0]
    seeds = SS[(SS.metric == metric) & (SS.condition == cond)].flip_rate if cond.startswith("C1") else None
    rows.append(dict(lab=lab, k=int(r.n_flipped), n=int(r.n_forgotten), p=r.flip_rate, lo=r.wilson_lo, hi=r.wilson_hi,
                     smin=seeds.min() if seeds is not None else np.nan, smax=seeds.max() if seeds is not None else np.nan))
R3 = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(W1, 1.75)); y = np.arange(len(R3))[::-1]
for yi, r in zip(y, R3.itertuples()):
    ax.plot([r.lo * 100, r.hi * 100], [yi, yi], color="black", lw=0.8, zorder=2)
    ax.plot(r.p * 100, yi, "o", ms=5, mfc="#0072B2", mec="black", mew=0.5, zorder=3)
    if not np.isnan(r.smin):
        ax.plot([r.smin * 100, r.smax * 100], [yi + 0.22, yi + 0.22], color="#0072B2", lw=2.2, solid_capstyle="butt", zorder=2)
    ax.text(101, yi, f"{r.k}/{r.n}", va="center", fontsize=7)
ax.set_yticks(y); ax.set_yticklabels(R3.lab); ax.set_xlim(0, 100); ax.set_ylim(-0.6, len(R3) - 0.3)
ax.set_xlabel("Verdict-flip rate among K-forgotten checkpoints (%)")
ax.xaxis.grid(True, **GRID); ax.set_axisbelow(True)
ax.legend(handles=[Line2D([], [], marker="o", color="black", lw=0.8, mfc="#0072B2", mec="black", ms=4, label="seed 0 with Wilson 95% CI"),
                   Line2D([], [], color="#0072B2", lw=2.2, label="range over seeds 0, 1, 2")],
          loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=6.5)
save(fig, "fig3_flip_rate_by_condition")

# ------------------------------------------------------------------ Fig. 4: which checkpoints recover, and how stable is it across seeds?
g = SD[(SD.metric == "answer_probability") & (SD.condition == "C1-high")]
st = g.groupby(["ck", "method"]).agg(mean=("G", "mean"), lo=("G", "min"), hi=("G", "max"), kf=("forgotten_K", "first")).reset_index()
st["cfg"] = st.ck.str[-2:].astype(int)
order = [m for m in ["GradDiff", "NPO", "SimNPO", "RMU"]]
fig, ax = plt.subplots(figsize=(W1, 2.3)); xt, xl, x0 = [], [], 0
for m in order:
    s = st[st.method == m].sort_values("cfg")
    xs = x0 + np.arange(len(s))
    for xi, r in zip(xs, s.itertuples()):
        ax.plot([xi, xi], [r.lo, r.hi], color="black", lw=0.8, zorder=2)
        ax.scatter(xi, r.mean, marker=MK[m], s=24, c=COL[m] if r.kf else "white", edgecolors=COL[m] if not r.kf else "black",
                   linewidths=0.9 if not r.kf else 0.5, zorder=3)
    xt.append(xs.mean()); xl.append(m); x0 += len(s) + 1
ax.axhline(0.5, color="black", lw=0.6, ls="--"); ax.axhline(0, color="#9a9a9a", lw=0.5)
ax.text(-0.6, 0.52, "recovered ($G \\geq 0.50$)", ha="left", fontsize=6.5)
ax.set_xticks(xt); ax.set_xticklabels(xl); ax.set_xlim(-0.8, x0 - 1.2)
ax.set_ylabel(r"Recovery $G$ under C1-high"); ax.set_ylim(-0.2, 0.8)
ax.yaxis.grid(True, **GRID); ax.set_axisbelow(True)
ax.legend(handles=[Line2D([], [], marker="o", color="none", mfc="#777777", mec="black", mew=0.5, ms=4.5, label="K-forgotten before"),
                   Line2D([], [], marker="o", color="none", mfc="white", mec="#777777", mew=0.9, ms=4.5, label="not K-forgotten")],
          loc="upper center", bbox_to_anchor=(0.55, 1.0), fontsize=6.5, handletextpad=0.2)
save(fig, "fig4_recovery_by_checkpoint")

# ------------------------------------------------------------------ Fig. 5: does a pre-attack metric predict recovery?
P = pd.read_csv(f"{a.p4}/predictiveness_C1high_bootstrap.csv")
nm = {"answer_probability": "Answer probability", "extraction_strength": "Extraction Strength", "exact_memorization": "Exact Memorization",
      "rouge_l": "ROUGE-L", "truth_ratio": "Truth Ratio", "min_k_percent": "Min-K% Prob", "token_entropy": "Token entropy"}
P = P.assign(lab=P.predictor.map(nm)).sort_values("spearman")
fig, ax = plt.subplots(figsize=(W1, 1.9)); y = np.arange(len(P))
for yi, r in zip(y, P.itertuples()):
    sig = r.ci_hi < 0 or r.ci_lo > 0
    ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color="black", lw=0.8, zorder=2)
    ax.plot(r.spearman, yi, "o", ms=5, mfc="#0072B2" if sig else "white", mec="black" if sig else "#0072B2", mew=0.6 if sig else 1.0, zorder=3)
    ax.text(1.02, yi, f"{r.spearman:+.2f}".replace("-", "−"), va="center", fontsize=7)
ax.axvline(0, color="black", lw=0.6)
ax.set_yticks(y); ax.set_yticklabels(P.lab); ax.set_xlim(-1, 1); ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_xlabel(r"Spearman $\rho$ between pre-condition $K$ and recovery $G$ (C1-high)")
ax.xaxis.grid(True, **GRID); ax.set_axisbelow(True)
save(fig, "fig5_metric_predictiveness")

# ------------------------------------------------------------------ Fig. 6: is recovery memorisation of the relearning examples? (A5)
A5 = pd.read_csv(f"{a.p3}/ablation_A5_partA_vs_partB.csv"); A5 = A5[A5.condition == "C1-high"]
fig, ax = plt.subplots(figsize=(2.3, 2.15))
ax.plot([-0.2, 0.8], [-0.2, 0.8], color="#8c8c8c", lw=0.7, ls="--", zorder=1)
for m in MK:
    s = A5[A5.method == m]; ax.scatter(s.G_partB, s.G_partA, marker=MK[m], s=20, c=COL[m], edgecolors="black", linewidths=0.5, zorder=3)
ax.text(0.40, 0.29, "equal recovery", fontsize=6.5, rotation=45, color="#555555")
ax.set_xlim(-0.2, 0.8); ax.set_ylim(-0.2, 0.8); ax.set_aspect("equal")
ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8]); ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8])
ax.set_xlabel(r"$G$, held-out authors (part B)"); ax.set_ylabel(r"$G$, relearning authors (part A)")
ax.legend(handles=method_handles(), loc="upper left", fontsize=6.5, handletextpad=0.1, borderaxespad=0.1)
save(fig, "fig6_ablation_A5")

# ------------------------------------------------------------------ Appendix figure: Spearman rho across all conditions (diverging scale)
from matplotlib.colors import LinearSegmentedColormap
PM = pd.read_csv(f"{a.p3}/metric_predictiveness.csv")
conds = ["C1-low", "C1-high", "C2", "C3", "C4"]; order = list(nm)
M = PM.pivot(index="predictor", columns="condition", values="spearman").loc[order, conds]
div = LinearSegmentedColormap.from_list("div", ["#b35806", "#f1a340", "#f2f2f2", "#67a9cf", "#2166ac"])
fig, ax = plt.subplots(figsize=(W1, 2.2))
im = ax.imshow(M.values, cmap=div, vmin=-1, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M.values[i, j]; ax.text(j, i, f"{v:+.2f}".replace("-", "−"), ha="center", va="center", fontsize=6.5, color="white" if abs(v) > 0.7 else "black")
ax.set_xticks(range(5)); ax.set_xticklabels(["C1-low", "C1-high", "C2", "C3", "C4$^{\\dagger}$"])
ax.set_yticks(range(len(order))); ax.set_yticklabels([nm[k] for k in order])
for sp in ax.spines.values(): sp.set_visible(False)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=[-1, 0, 1]); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6.5)
cb.set_label(r"Spearman $\rho$", fontsize=7)
save(fig, "figA1_spearman_all_conditions")

# ------------------------------------------------------------------ Appendix figure: G for every checkpoint under every condition
fig, ax = plt.subplots(figsize=(W1, 2.2)); off = {"GradDiff": -0.27, "NPO": -0.09, "SimNPO": 0.09, "RMU": 0.27}
for i, c in enumerate(conds):
    gg = T3[(T3.metric == ("rouge_l" if c == "C4" else "answer_probability")) & (T3.condition == c)]
    for m in MK:
        s = gg[gg.method == m]
        ax.scatter(i + off[m] + np.linspace(-0.05, 0.05, len(s)), s.G, marker=MK[m], s=12, c=COL[m], edgecolors="black", linewidths=0.4, zorder=3)
ax.axhline(0.5, color="black", lw=0.6, ls="--"); ax.axhline(0, color="#9a9a9a", lw=0.5)
ax.set_xticks(range(5)); ax.set_xticklabels(["C1-low", "C1-high", "C2", "C3", "C4$^{\\dagger}$"]); ax.set_ylim(-0.8, 0.8)
ax.set_ylabel(r"Recovery $G$"); ax.yaxis.grid(True, **GRID); ax.set_axisbelow(True)
ax.legend(handles=method_handles(), ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6.5, handletextpad=0.1, columnspacing=0.8)
save(fig, "figA2_recovery_all_conditions")
print("figures written to", a.out)
