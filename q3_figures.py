"""Figures for Q3, from the saved results (results/q3_pairs.csv, results/q3_sim.csv). No new simulation.

    python q3_figures.py
Writes docs/figures/q3_power.png, q3_identification.png, q3_real_z.png, q3_specialties.png.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.stats import norm
from categories import CATEGORIES

OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)

# Same surfaces, ink and rcParams as q1_figures.py. Categorical slots 1-4 of the reference palette, in fixed order
# (validated: adjacent CVD dE >= 9.1). Slots 3-4 are below 3:1 on the surface, so every line also has its own marker
# and the legend names it. Diverging: blue <-> red with a gray midpoint.
BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
SURFACE, INK, INK2, MUTED, GRID, AXIS, MID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False,
})
TESTS = [("hc", "HC", BLUE, "o"), ("hc_plus", "HC+", ORANGE, "s"),
         ("max_abs_z", "min-P", AQUA, "^"), ("chi2", "χ²", YELLOW, "D")]
LABEL = {"coding": "Coding", "math": "Math", "instruction_following": "Instr. following", "hard_prompts": "Hard prompts",
         "chinese": "Chinese", "russian": "Russian", "multi_turn": "Multi-turn", "long_query": "Long query"}


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=200, bbox_inches="tight")
    plt.close(fig)


sim = pd.read_csv("results/q3_sim.csv")
size_row = sim[sim.get("cell", pd.Series(dtype=str)).eq("size check")].iloc[0]
grid = sim[sim.theta.notna()].copy()
thetas, rs = sorted(grid.theta.unique()), sorted(grid.r.unique())
N = 958

# ---------------------------------------------------------------- 1. power of the global tests
fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True, sharey=True)
for ax, th in zip(axes.flat, thetas):
    g = grid[grid.theta == th].sort_values("r")
    ax.axhline(0.05, color=MUTED, lw=1, ls=(0, (3, 3)))
    for key, name, col, mk in TESTS:
        ax.plot(g.r, g[key], color=col, lw=2, marker=mk, ms=6, mec=SURFACE, mew=1.2, label=name, zorder=3)
    s = int(g.s.iloc[0])
    ax.set_title(f"ϑ = {th:g}: {s} specialties", fontsize=10)
    ax.set_xscale("log")
    ax.set_xticks(rs)
    ax.set_xticklabels([f"{r:g}" for r in rs])
    ax.minorticks_off()
    ax.set_ylim(-0.03, 1.03)
for ax in axes[1]:
    ax.set_xlabel("strength r   (μ = √(2 r log N) in z units)")
for ax in axes[:, 0]:
    ax.set_ylabel("power")
axes[0, 0].text(rs[-1], 0.08, "size 0.05", color=MUTED, fontsize=8, ha="right", va="bottom")
fig.legend(handles=[Line2D([], [], color=c, lw=2, marker=m, ms=6, mec=SURFACE, label=n) for _, n, c, m in TESTS],
           loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
fig.suptitle(f"Power of the global tests for planted specialties (N = {N} tested pairs, 200 replicates per point)",
             x=0.01, y=1.08, ha="left", fontsize=11, fontweight="bold")
fig.text(0.01, -0.02, f"Size with no specialties: HC {size_row.hc:.2f}, HC+ {size_row.hc_plus:.2f}, "
         f"min-P {size_row.max_abs_z:.2f}, χ² {size_row.chi2:.2f}.", color=INK2, fontsize=8.5)
fig.tight_layout()
save(fig, "q3_power.png")

# ---------------------------------------------------------------- 2. identification: TPR and false discoveries
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
styles = {thetas[0]: "-", thetas[-1]: (0, (4, 2))}
for th, ls in styles.items():
    g = grid[grid.theta == th].sort_values("mu")
    a1.plot(g.mu, g.bonf_tpr, color=BLUE, lw=2, ls=ls, marker="o", ms=5, mec=SURFACE)
    a1.plot(g.mu, g.bh_tpr, color=ORANGE, lw=2, ls=ls, marker="s", ms=5, mec=SURFACE)
    a2.plot(g.mu, g.bonf_any_false, color=BLUE, lw=2, ls=ls, marker="o", ms=5, mec=SURFACE)
    a2.plot(g.mu, g.bh_fdp, color=ORANGE, lw=2, ls=ls, marker="s", ms=5, mec=SURFACE)
a1.set_title("Share of planted specialties found")
a1.set_ylabel("true positive rate")
a1.set_ylim(-0.03, 1.03)
a2.set_title("Errors against their targets")
a2.set_ylabel("error rate")
a2.axhline(0.10, color=ORANGE, lw=1, ls=(0, (1, 2)))
a2.axhline(0.05, color=BLUE, lw=1, ls=(0, (1, 2)))
x_end = grid.mu.max() + 0.12
a2.text(x_end, 0.103, "BH target\nFDR 0.10", color=INK2, fontsize=8, ha="left", va="bottom")
a2.text(x_end, 0.053, "Bonferroni target\nFWER 0.05", color=INK2, fontsize=8, ha="left", va="bottom")
a2.set_xlim(right=grid.mu.max() + 1.0)
a2.set_ylim(0, 0.2)
for ax in (a1, a2):
    ax.set_xlabel("planted size μ (z units)")
handles = [Line2D([], [], color=BLUE, lw=2, marker="o", mec=SURFACE, label="Bonferroni (5%)"),
           Line2D([], [], color=ORANGE, lw=2, marker="s", mec=SURFACE, label="BH (10%)"),
           Line2D([], [], color=INK2, lw=1.5, ls="-", label=f"ϑ = {thetas[0]:g} (dense)"),
           Line2D([], [], color=INK2, lw=1.5, ls=(0, (4, 2)), label=f"ϑ = {thetas[-1]:g} (sparse)")]
fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06))
fig.text(0.01, -0.04, "Right panel: BH's false discovery proportion and Bonferroni's chance of any false discovery. "
         "Spillovers into overlapping categories of a planted model are not counted as false.", color=INK2, fontsize=8.5)
fig.tight_layout()
save(fig, "q3_identification.png")

# ---------------------------------------------------------------- 3. real board: z* against the null
pairs = pd.read_csv("results/q3_pairs.csv")
z = pairs.z.values
bar = norm.isf(0.05 / (2 * len(z)))
fig, ax = plt.subplots(figsize=(7.5, 3.6))
LIM = 15
bins = np.arange(-LIM, LIM + 1, 1.0)
ax.hist(np.clip(z, -LIM + 0.5, LIM - 0.5), bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=1,
        label=f"real board: {len(z)} pairs")
xs = np.linspace(-LIM, LIM, 600)
ax.plot(xs, len(z) * norm.pdf(xs), color=ORANGE, lw=2, label="expected with no specialties, N(0, 1)")
for x in (-bar, bar):
    ax.axvline(x, color=MUTED, lw=1, ls=(0, (3, 3)))
ax.text(bar + 0.6, ax.get_ylim()[1] * 0.9, f"Bonferroni bar |z| = {bar:.2f}", color=INK2, fontsize=8.5)
out_of_range = pairs[pairs.z.abs() > LIM]
ext = pairs.loc[pairs.z.abs().idxmax()]
ax.text(0.99, 0.62, f"{len(out_of_range)} pairs beyond ±{LIM}\nare counted in the end bins\n"
        f"(most extreme: {ext.model},\n{LABEL[ext.category]}, z = {ext.z:.0f})", transform=ax.transAxes,
        ha="right", va="top", color=INK2, fontsize=8)
ax.set_xlim(-LIM, LIM)
ax.set_xlabel("standardized specialty z*  (negative: weaker in the category than overall)")
ax.set_ylabel("pairs")
ax.set_title(f"Specialties on the 2024 board: {int(pairs.bonferroni.sum())} of {len(z)} pass Bonferroni")
ax.legend(loc="center left", bbox_to_anchor=(0, 0.6))
save(fig, "q3_real_z.png")

# ---------------------------------------------------------------- 4. specialty map for the top models
top = (pairs.drop_duplicates("model").sort_values("overall_rating", ascending=False).model.head(40).tolist())
mat = pairs.pivot(index="model", columns="category", values="specialty_points").reindex(index=top, columns=CATEGORIES)
sig = pairs.pivot(index="model", columns="category", values="bonferroni").reindex(index=top, columns=CATEGORIES)
cmap = LinearSegmentedColormap.from_list("bgr", [BLUE, MID, RED])
cmap.set_bad(SURFACE)
lim = 80
fig, ax = plt.subplots(figsize=(7.2, 11))
im = ax.imshow(mat.values, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
ys, xs_ = np.nonzero(sig.fillna(False).values.astype(bool))
ax.scatter(xs_, ys, s=10, color=INK, marker="o", label="passes Bonferroni")
ax.set_xticks(range(len(CATEGORIES)))
ax.set_xticklabels([LABEL[c] for c in CATEGORIES], rotation=35, ha="right")
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top, fontsize=8)
ax.grid(False)
ax.set_xticks(np.arange(-0.5, len(CATEGORIES)), minor=True)
ax.set_yticks(np.arange(-0.5, len(top)), minor=True)
ax.grid(which="minor", color=SURFACE, linewidth=2)
ax.tick_params(which="minor", length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, extend="both")
cb.set_label("specialty: category rating − overall rating (points, centered)", color=INK2)
cb.outline.set_visible(False)
ax.set_title("Category specialties of the 40 highest-rated models (2024-08 board)")
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.08), fontsize=8.5)
fig.text(0.01, 0.005, "Blank: fewer than 200 battles in the category (not tested). Blue: weaker in the category, "
         "red: stronger.", color=INK2, fontsize=8)
save(fig, "q3_specialties.png")
print("wrote", ", ".join(f"{OUT}/q3_{n}.png" for n in ["power", "identification", "real_z", "specialties"]))
