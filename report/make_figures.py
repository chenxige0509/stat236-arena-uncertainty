"""Figures for the written report (report/report.tex), from the committed results only. No new simulation.

    python report/make_figures.py        (from the repo root)
Writes report/figures/fig{1..4}_*.pdf.

Same palette and surfaces as q1_figures.py / q3_figures.py (reference palette, validated for color-vision deficiency):
Arena blue, Bonferroni orange, Holm aqua; procedures shown only for comparison are in neutral grays and are always
named next to their mark. Text uses STIX Two to match the report's body font. Figures carry no titles: the captions
in report.tex say what each one shows.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm, spearmanr

OUT = "report/figures"
os.makedirs(OUT, exist_ok=True)
R = lambda name, **k: pd.read_csv(f"results/{name}", **k)

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GRAY1, GRAY2, GRAY0 = "#a8a79f", "#6b6a64", "#cfcec8"   # neutral steps for comparison-only procedures
VIOLET, MAGENTA, GOLD = "#6250d6", "#d55181", "#b07d00"   # Q3 tests (kept apart from the Q2 procedure colors)
plt.rcParams.update({
    "font.family": "STIX Two Text", "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.7, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
W = 6.5   # text width of the report, inches


def panel_label(ax, s, x=-0.02):
    ax.text(x, 1.02, s, transform=ax.transAxes, ha="right", va="bottom", fontsize=10, fontweight="bold")


def save(fig, name):
    fig.savefig(f"{OUT}/{name}")
    plt.close(fig)


# ================================================================ Figure 1: Q1
q1 = R("q1_summary.csv", index_col=0)
an = R("q1_analysis_per_model.csv")
fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.2), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.32})
vals = [("score CI coverage, per model (nominal 0.95)", "rank spread coverage, per model"),
        ("score CIs: all cover (simultaneous)", "rank spread coverage, simultaneous (all models)")]
x = np.arange(2)
for k, (lab, col) in enumerate([("Rating intervals", GRAY1), ("Rank spreads", BLUE)]):
    v = np.array([q1.loc[pair[k], "value"] for pair in vals])
    se = np.array([np.nan_to_num(q1.loc[pair[k], "mc_se"]) for pair in vals])
    pos = x + (k - 0.5) * 0.36
    a.bar(pos, v, width=0.33, color=col, label=lab, zorder=2)
    a.errorbar(pos, v, yerr=1.96 * se, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2, zorder=3)
    for p, vv in zip(pos, v):
        a.text(p, vv + 0.025, f"{vv:.3f}", ha="center", va="bottom", fontsize=7.5)
a.axhline(0.95, color=INK2, lw=0.8, ls=(0, (4, 3)))
a.text(0.5, 0.965, "95%", va="bottom", ha="center", fontsize=7.5, color=INK2)
a.set_xticks(x, ["Each model", "All 129 models"])
a.set_xlim(-0.55, 1.75)
a.set_ylim(0, 1.13)
a.set_yticks([0, 0.25, 0.5, 0.75, 1])
a.set_ylabel("Coverage")
a.grid(axis="x", visible=False)
a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, handlelength=1, columnspacing=1)
panel_label(a, "a")

rho = spearmanr(an.neighbours_within_10pts, an.spread_width)[0]
sc = b.scatter(an.neighbours_within_10pts + np.random.default_rng(0).uniform(-0.18, 0.18, len(an)), an.spread_width,
               c=an.se, cmap="Blues", vmin=0, vmax=an.se.max() * 1.05, s=14, edgecolors=INK2, linewidths=0.3, zorder=3)
b.set_xlabel("Models with true rating within 10 points (crowding)")
b.set_ylabel("Mean rank-spread width (ranks)")
b.text(0.03, 0.95, f"Spearman $\\rho$ = {rho:.2f}", transform=b.transAxes, va="top", fontsize=8)
cb = fig.colorbar(sc, ax=b, pad=0.02, fraction=0.05)
cb.set_label("Standard error (rating points)", fontsize=8)
cb.ax.tick_params(labelsize=7)
cb.outline.set_linewidth(0.5)
panel_label(b, "b")
save(fig, "fig1_q1.pdf")

# ================================================================ Figure 2: Q2 cost-coverage frontier + stress tests
q2 = R("q2_summary.csv", index_col=0)
sw = R("q2_sweep.csv")
SIM, WID = "rank interval coverage, simultaneous (all models)", "rank interval width, mean"
fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.45), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.55})
z_need = sw.loc[sw.rs_all_cover >= 0.95, "z"].min()
e = sw[sw.z == z_need].iloc[0]
bars = [("Arena (uncorrected)", q2.loc[WID, "arena"], q2.loc[SIM, "arena"], BLUE),
        (f"Oracle $z$ = {z_need:.1f}*", e.rs_width, e.rs_all_cover, INK),
        ("Holm", q2.loc[WID, "holm"], q2.loc[SIM, "holm"], AQUA),
        ("One-sided Holm", q2.loc[WID, "holm_onesided"], q2.loc[SIM, "holm_onesided"], GRAY2),
        ("Pairwise Bonferroni", q2.loc[WID, "pairwise_bonferroni"], q2.loc[SIM, "pairwise_bonferroni"], GRAY1),
        ("Bonferroni", q2.loc[WID, "bonferroni"], q2.loc[SIM, "bonferroni"], ORANGE),
        (r"$\chi^2$ Scheffé", q2.loc[WID, "chi2_scheffe"], q2.loc[SIM, "chi2_scheffe"], GRAY0),
        (r"$\chi^2$ projection", q2.loc[WID, "chi2"], q2.loc[SIM, "chi2"], GRAY0)]
yy = np.arange(len(bars))[::-1]
for yi, (name, w, c, col) in zip(yy, bars):
    a.barh(yi, w, height=0.62, color=col, zorder=2)
    a.text(w + 0.6, yi, f"{w:.1f}  ({c:.1%})", va="center", fontsize=7.5, color=INK)
a.set_yticks(yy, [b_[0] for b_ in bars], fontsize=8)
a.tick_params(axis="y", length=0)
a.set_xlim(0, 58)
a.set_xticks([0, 10, 20, 30, 40])
a.grid(axis="y", visible=False)
a.set_xlabel("Mean rank-interval width (ranks)")
panel_label(a, "a", x=-0.62)

worlds = [("_tiers20", "20\ntiers"), ("_tiers10", "10\ntiers"), ("_scale0", "all\ntied"),
          ("_tiers20s0.05", "5%\n"), ("_tiers20s0.1", "10%\n"), ("_tiers20s0.2", "20%\n")]
FO = "[declared orderings] P(any false ordering)"
series = [("bonferroni", "Bonferroni", ORANGE), ("pairwise_bonferroni", "Pairwise Bonferroni", GRAY1),
          ("holm_onesided", "One-sided Holm", GRAY2), ("holm", "Holm", AQUA)]
xs, bw = np.arange(len(worlds)), 0.19
for k, (p, name, col) in enumerate(series):
    v = np.array([R(f"q2{w}_summary.csv", index_col=0).loc[FO, p] for w, _ in worlds])
    se = np.array([R(f"q2{w}_summary.csv", index_col=0).loc[FO, f"{p}_mc_se"] for w, _ in worlds])
    pos = xs + (k - 1.5) * (bw + 0.01)
    b.bar(pos, v, width=bw, color=col, label=name, zorder=2)
    b.errorbar(pos, v, yerr=1.96 * se, fmt="none", ecolor=INK2, elinewidth=0.6, capsize=0, zorder=3)
b.axhline(0.05, color=INK2, lw=0.8, ls=(0, (4, 3)))
b.text(-0.45, 0.0505, r"$\alpha$ = 5%", ha="left", va="bottom", fontsize=7.5, color=INK2)
b.axvline(2.5, color=AXIS, lw=0.6)
b.text(1, -0.21, "exact ties", ha="center", fontsize=7.5, color=INK2, va="top", transform=b.get_xaxis_transform())
b.text(4, -0.21, "near ties: 20 tiers,\nwithin-tier spread", ha="center", fontsize=7.5, color=INK2, va="top",
       linespacing=1.0, transform=b.get_xaxis_transform())
b.set_xticks(xs, [n for _, n in worlds], fontsize=7.5)
b.set_ylim(0, 0.068)
b.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
b.set_ylabel("P(any false ordering)")
b.grid(axis="x", visible=False)
b.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=7, handlelength=1, columnspacing=1.0,
         borderaxespad=0.2)
panel_label(b, "b", x=-0.14)
save(fig, "fig2_q2.pdf")

# ================================================================ Figure 3: Q2 real 2024 board, top 20
real = R("q2_real.csv")
top = real.head(20).iloc[::-1]
y = np.arange(len(top))
fig, ax = plt.subplots(figsize=(5.6, 3.7))
for off, (p, name, col) in zip([0.25, 0.0, -0.25], [("arena", "Arena (uncorrected)", BLUE),
                                                     ("bonferroni", "Bonferroni", ORANGE), ("holm", "Holm", AQUA)]):
    ax.hlines(y + off, top[f"{p}_best"] - 0.32, top[f"{p}_worst"] + 0.32, color=col, lw=2.6, capstyle="round",
              label=name, zorder=3)
ax.set_yticks(y, top.model, fontsize=8)
ax.set_xticks([1, 5, 10, 15, 20, 25])
ax.set_xlim(0.2, top[["arena_worst", "bonferroni_worst", "holm_worst"]].values.max() + 0.8)
ax.set_ylim(-0.6, len(top) - 0.4)
ax.set_xlabel("Rank")
ax.grid(axis="y", visible=False)
ax.legend(loc="upper right", ncol=1, handlelength=1.5)
save(fig, "fig3_q2_real.pdf")

# ================================================================ Figure 4: Q3 real z* and power
q3 = R("q3_pairs.csv")
sim = R("q3_sim.csv")
grid = sim[sim.theta.notna()]
N = len(q3)
crit = norm.isf(0.05 / (2 * N))
fig, axes = plt.subplots(1, 3, figsize=(W, 2.2), gridspec_kw={"width_ratios": [1.35, 1, 1], "wspace": 0.28})
a = axes[0]
z = np.clip(q3.z.values, -15, 15)
bins = np.arange(-15, 15.01, 1)
a.hist(z, bins=bins, color=GRAY1, edgecolor="white", linewidth=0.4, zorder=2, label=f"2024 board ({N} pairs)")
t = np.linspace(-15, 15, 600)
a.plot(t, N * norm.pdf(t), color=INK, lw=1.2, zorder=3, label="Null: N(0, 1)")
for s in (-crit, crit):
    a.axvline(s, color=INK2, lw=0.7, ls=(0, (3, 2)))
a.text(crit + 0.4, 215, f"Bonferroni\n$|z^*|$ = {crit:.2f}", fontsize=7, color=INK2, va="top")
a.set_xlim(-15, 15)
a.set_ylim(0, 430)
a.set_xlabel(r"Standardized specialty $z^*$ (clipped at $\pm$15)")
a.set_ylabel("Pairs")
a.legend(loc="upper left", fontsize=7, handlelength=1.0, borderaxespad=0.1)
panel_label(a, "a")
TESTS = [("hc", "HC", VIOLET, "o", "-"), ("hc_plus", r"HC$^+$", VIOLET, "s", (0, (3, 1.5))),
         ("max_abs_z", r"min-$P$", MAGENTA, "^", "-"), ("chi2", r"$\chi^2$", GOLD, "D", "-")]
for ax, th, lab in [(axes[1], 0.3, "b"), (axes[2], 0.5, "c")]:
    g = grid[grid.theta == th].sort_values("r")
    ax.axhline(0.05, color=MUTED, lw=0.7, ls=(0, (3, 3)))
    for key, name, col, mk, ls in TESTS:
        se = g[f"{key}_mc_se"].values
        ax.fill_between(g.r, g[key] - 1.96 * se, g[key] + 1.96 * se, color=col, alpha=0.12, lw=0, zorder=2)
        ax.plot(g.r, g[key], color=col, lw=1.3, ls=ls, marker=mk, ms=4, mec="white", mew=0.6, label=name, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks(sorted(g.r.unique()), [f"{v:g}" for v in sorted(g.r.unique())])
    ax.minorticks_off()
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel(r"Signal strength $r$")
    ax.set_title(rf"$\vartheta$ = {th:g}: {int(g.s.iloc[0])} specialties", fontsize=8, loc="left", pad=3)
    panel_label(ax, lab, x=-0.06)
axes[1].set_ylabel("Power")
axes[2].tick_params(labelleft=False)
axes[1].legend(loc="lower right", fontsize=7, handlelength=2.0, bbox_to_anchor=(1.0, 0.1))
save(fig, "fig4_q3.pdf")
# ================================================================ Figure A1: Q1 mean rank spreads by true rank
pm = R("q2_per_model.csv").sort_values("true_rank")
fig, ax = plt.subplots(figsize=(W, 2.9))
ax.vlines(pm.true_rank, pm.arena_mean_best - 0.5, pm.arena_mean_worst + 0.5, color=BLUE, lw=2.2, zorder=2,
          label="Mean rank spread [best, worst]")
ax.plot(pm.true_rank, pm.true_rank, "o", ms=2.2, color=INK, zorder=3, label="True rank")
low = pm[pm.arena_rs_coverage < 0.995]
ax.plot(low.true_rank, low.arena_mean_worst + 3, "v", ms=3.5, color=INK2, zorder=3,
        label="Coverage below 99.5% (3 models)" if len(low) == 3 else f"Coverage below 99.5% ({len(low)} models)")
ax.invert_yaxis()
ax.set_xlim(0, len(pm) + 1)
ax.set_xlabel("True rank")
ax.set_ylabel("Rank")
ax.legend(loc="lower left", fontsize=7.5)
save(fig, "figA1_q1_spreads.pdf")

# ================================================================ Figure A2: Q3 specialty map, top 40 models
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
CATS = ["coding", "math", "instruction_following", "hard_prompts", "chinese", "russian", "multi_turn", "long_query"]
CAT_LAB = ["Coding", "Math", "Instr.\nfollowing", "Hard\nprompts", "Chinese", "Russian", "Multi-\nturn", "Long\nquery"]
top40 = q3.groupby("model").overall_rating.first().sort_values(ascending=False).index[:40]
P = q3.pivot(index="model", columns="category", values="specialty_points").reindex(index=top40, columns=CATS)
S = q3.pivot(index="model", columns="category", values="bonferroni").reindex(index=top40, columns=CATS)
cmap = LinearSegmentedColormap.from_list("div", [BLUE, "#f0efec", "#e34948"])
fig, ax = plt.subplots(figsize=(W, 7.6))
im = ax.imshow(P.values, cmap=cmap, norm=TwoSlopeNorm(0, -80, 80), aspect="auto")
yy, xx = np.nonzero(S.fillna(False).values.astype(bool))
ax.plot(xx, yy, "o", ms=3, color=INK)
for (yi, xi) in zip(*np.nonzero(P.isna().values)):
    ax.add_patch(plt.Rectangle((xi - 0.5, yi - 0.5), 1, 1, fill=False, hatch="////", ec=AXIS, lw=0))
ax.set_xticks(range(len(CATS)), CAT_LAB, fontsize=8)
ax.xaxis.tick_top()
ax.set_yticks(range(len(top40)), top40, fontsize=7.5)
ax.tick_params(length=0)
ax.grid(False)
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, extend="both")
cb.set_label("Category rating minus overall rating (points, centered)", fontsize=8)
cb.ax.tick_params(labelsize=7)
cb.outline.set_linewidth(0.5)
save(fig, "figA2_q3_specialties.pdf")
print("wrote", sorted(os.listdir(OUT)))
