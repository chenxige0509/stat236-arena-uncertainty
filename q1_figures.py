"""Figures for docs/q1.md, from the saved Q1 replicates (results/q1.npz, results/q1_analysis_per_model.csv).

    python q1_figures.py
Writes docs/figures/q1_*.png.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)

# Palette: first two categorical slots of the reference palette (validated all-pairs, light mode)
SCORE, SPREAD = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False,
})

sim = np.load("results/q1.npz", allow_pickle=True)
theta, r_true = sim["theta"], sim["r_true"]
ci, rs = sim["ci_cover"].astype(bool), sim["rs_cover"].astype(bool)
best, worst = sim["best"], sim["worst"]
R, M = ci.shape
an = pd.read_csv("results/q1_analysis_per_model.csv")


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 1. coverage: per model vs all at once
vals = {"Score intervals": [ci.mean(), ci.all(1).mean()], "Rank spreads": [rs.mean(), rs.all(1).mean()]}
ses = {"Score intervals": [ci.mean(1).std(ddof=1) / np.sqrt(R), 0.0],
       "Rank spreads": [rs.mean(1).std(ddof=1) / np.sqrt(R), rs.all(1).std(ddof=1) / np.sqrt(R)]}
fig, ax = plt.subplots(figsize=(6.4, 3.6))
x = np.arange(2)
w = 0.34
for k, (lab, col) in enumerate([("Score intervals", SCORE), ("Rank spreads", SPREAD)]):
    xs = x + (k - 0.5) * (w + 0.02)
    ax.bar(xs, vals[lab], width=w, color=col, label=lab, zorder=2)
    ax.errorbar(xs, vals[lab], yerr=1.96 * np.array(ses[lab]), fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
    for xi, v in zip(xs, vals[lab]):
        ax.text(xi, v + 0.03, f"{v:.3f}", ha="center", va="bottom", color=INK, fontsize=9)
ax.hlines(0.95, -0.5, 1.5, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
ax.text(1.53, 0.95, "nominal 95%", va="center", ha="left", color=INK2, fontsize=9)
ax.set_xticks(x, ["Per model", f"All {M} models at once"])
ax.set_ylim(0, 1.12)
ax.set_xlim(-0.5, 1.95)
ax.set_ylabel("Coverage")
ax.grid(axis="x", visible=False)
ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(0.0, 1.0), borderaxespad=0.2)
ax.set_title(f"Coverage of Arena's intervals ({R} simulated datasets)", pad=28)
save(fig, "q1_coverage.png")

# ---------------------------------------------------------------- 2. misses per dataset
miss_ci, miss_rs = (~ci).sum(1), (~rs).sum(1)
top = max(miss_ci.max(), miss_rs.max())
bins = np.arange(-0.5, top + 1.5)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, m, col, lab in [(axes[0], miss_ci, SCORE, "Score intervals"), (axes[1], miss_rs, SPREAD, "Rank spreads")]:
    ax.hist(m, bins=bins, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=2)
    ax.set_title(f"{lab}\nmean {m.mean():.2f} misses; zero misses in {np.mean(m == 0):.1%} of datasets", fontsize=10)
    ax.set_xlabel(f"Models (of {M}) whose interval misses the truth")
    ax.set_xticks(np.arange(0, top + 1, 5))
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel(f"Simulated datasets (of {R})")
fig.tight_layout()
save(fig, "q1_misses.png")

# ---------------------------------------------------------------- 3. rank spreads by true rank
order = np.argsort(r_true)
mb, mw = best.mean(0)[order], worst.mean(0)[order]
cov = rs.mean(0)[order]
tr = r_true[order]
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.vlines(tr, mb, mw, color=SPREAD, lw=2.2, zorder=2, label="Mean rank spread [best, worst]")
ax.scatter(tr, tr, s=12, color=INK, zorder=3, label="True rank", linewidths=0)
low = cov < 0.995
ax.scatter(tr[low], mw[low] + 2.5, marker="v", s=18, color=INK2, zorder=3, linewidths=0,
           label="Coverage below 99.5%")
ax.set_xlabel("True rank")
ax.set_ylabel("Rank")
ax.set_xlim(0, M + 1)
ax.set_ylim(M + 6, -3)
ax.legend(loc="lower left")
ax.set_title("Rank spreads, averaged over simulated datasets")
save(fig, "q1_spreads.png")

# ---------------------------------------------------------------- 4. what drives spread width
fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
panels = [("neighbours_within_10pts", "Models within 10 rating points", False),
          ("se", "SE of the score (rating points)", False),
          ("true_rank", "True rank", False)]
for ax, (col, lab, logx) in zip(axes, panels):
    ax.scatter(an[col], an.spread_width, s=16, color=SPREAD, alpha=0.85, linewidths=0.6, edgecolors=SURFACE, zorder=2)
    rho = an[[col, "spread_width"]].corr("spearman").iloc[0, 1]
    ax.set_title(f"Spearman ρ = {rho:.2f}", fontsize=10, fontweight="normal", color=INK2)
    ax.set_xlabel(lab)
    if logx:
        ax.set_xscale("log")
axes[0].set_ylabel("Mean rank spread width")
fig.suptitle("Rank spread width vs. crowding, data and strength (one dot per model)", x=0.01, ha="left",
             fontweight="bold", fontsize=11)
fig.tight_layout()
save(fig, "q1_width_drivers.png")

# ---------------------------------------------------------------- 5. by tier
tiers = pd.cut(an.true_rank, [0, 10, 30, 60, 100, M], labels=["1–10", "11–30", "31–60", "61–100", "101+"])
g = an.groupby(tiers, observed=True)[["spread_width", "neighbours_within_10pts", "se"]].mean()
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), sharex=True)
for ax, (col, lab, colr) in zip(axes, [("spread_width", "Mean rank spread width", SPREAD),
                                        ("neighbours_within_10pts", "Models within 10 points", INK2),
                                        ("se", "Mean SE (rating points)", INK2)]):
    ax.bar(range(len(g)), g[col], color=colr, width=0.7, zorder=2)
    for i, v in enumerate(g[col]):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_title(lab, fontsize=10, fontweight="normal", color=INK2)
    ax.set_xticks(range(len(g)), g.index)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, g[col].max() * 1.15)
axes[1].set_xlabel("True rank tier")
fig.suptitle("By tier: spreads are widest where models are crowded, not where they are weakest", x=0.01, ha="left",
             fontweight="bold", fontsize=11)
fig.tight_layout()
save(fig, "q1_tiers.png")
print("wrote", sorted(f for f in os.listdir(OUT) if f.startswith("q1_")))
