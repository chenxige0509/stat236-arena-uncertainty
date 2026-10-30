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
GRAY1, GRAY2 = "#a8a79f", "#6b6a64"          # two neutral steps for comparison-only procedures
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
fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.45), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.32})
vals = [("score CI coverage, per model (nominal 0.95)", "rank spread coverage, per model"),
        ("score CIs: all cover (simultaneous)", "rank spread coverage, simultaneous (all models)")]
x = np.arange(2)
for k, (lab, col) in enumerate([("Score intervals", BLUE), ("Rank spreads", ORANGE)]):
    v = np.array([q1.loc[pair[k], "value"] for pair in vals])
    se = np.array([np.nan_to_num(q1.loc[pair[k], "mc_se"]) for pair in vals])
    pos = x + (k - 0.5) * 0.36
    a.bar(pos, v, width=0.33, color=col, label=lab, zorder=2)
    a.errorbar(pos, v, yerr=1.96 * se, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2, zorder=3)
    for p, vv in zip(pos, v):
        a.text(p, vv + 0.025, f"{vv:.3f}", ha="center", va="bottom", fontsize=7.5)
a.axhline(0.95, color=INK2, lw=0.8, ls=(0, (4, 3)))
a.text(-0.53, 0.965, "95%", va="bottom", ha="left", fontsize=7.5, color=INK2)
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
fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.75), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.55})
z_need = sw.loc[sw.rs_all_cover >= 0.95, "z"].min()
e = sw[sw.z == z_need].iloc[0]
bars = [("Arena (uncorrected)", q2.loc[WID, "arena"], q2.loc[SIM, "arena"], BLUE),
        (f"Oracle $z$ = {z_need:.1f}*", e.rs_width, e.rs_all_cover, INK),
        ("Holm", q2.loc[WID, "holm"], q2.loc[SIM, "holm"], AQUA),
        ("Holm (one-sided)", q2.loc[WID, "holm_onesided"], q2.loc[SIM, "holm_onesided"], GRAY2),
        ("Pairwise Bonferroni", q2.loc[WID, "pairwise_bonferroni"], q2.loc[SIM, "pairwise_bonferroni"], GRAY2),
        ("Bonferroni", q2.loc[WID, "bonferroni"], q2.loc[SIM, "bonferroni"], ORANGE),
        (r"$\chi^2$ Scheffé", q2.loc[WID, "chi2_scheffe"], q2.loc[SIM, "chi2_scheffe"], GRAY1),
        (r"$\chi^2$ projection", q2.loc[WID, "chi2"], q2.loc[SIM, "chi2"], GRAY1)]
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
          ("holm_onesided", "Holm (one-sided)", GRAY2), ("holm", "Holm", AQUA)]
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
b.text(1, 0.068, "exact ties", ha="center", fontsize=7.5, color=INK2, va="bottom")
b.text(4, 0.068, "near ties: 20 tiers,\nwithin-tier spread", ha="center", fontsize=7.5, color=INK2, va="bottom",
       linespacing=1.0)
b.set_xticks(xs, [n for _, n in worlds], fontsize=7.5)
b.set_ylim(0, 0.078)
b.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
b.set_ylabel("P(any false ordering)")
b.grid(axis="x", visible=False)
b.legend(loc="upper right", bbox_to_anchor=(1.0, 0.88), fontsize=7, handlelength=1, frameon=True, framealpha=1,
         edgecolor="none", facecolor="white")
panel_label(b, "b")
save(fig, "fig2_q2.pdf")

# ================================================================ Figure 3: Q2 real 2024 board, top 20
real = R("q2_real.csv")
top = real.head(20).iloc[::-1]
y = np.arange(len(top))
fig, ax = plt.subplots(figsize=(W, 3.45))
for off, (p, name, col) in zip([0.25, 0.0, -0.25], [("arena", "Arena (uncorrected)", BLUE),
                                                     ("bonferroni", "Bonferroni", ORANGE), ("holm", "Holm", AQUA)]):
    ax.hlines(y + off, top[f"{p}_best"] - 0.32, top[f"{p}_worst"] + 0.32, color=col, lw=2.6, capstyle="round",
              label=name, zorder=3)
ax.set_yticks(y, top.model, fontsize=7.5)
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
fig, axes = plt.subplots(1, 3, figsize=(W, 2.35), gridspec_kw={"width_ratios": [1.35, 1, 1], "wspace": 0.28})
a = axes[0]
z = np.clip(q3.z.values, -15, 15)
bins = np.arange(-15, 15.01, 1)
a.hist(z, bins=bins, color=BLUE, edgecolor="white", linewidth=0.4, zorder=2, label=f"Real ({N} pairs)")
t = np.linspace(-15, 15, 600)
a.plot(t, N * norm.pdf(t), color=ORANGE, lw=1.4, zorder=3, label="Null: N(0, 1)")
for s in (-crit, crit):
    a.axvline(s, color=INK2, lw=0.7, ls=(0, (3, 2)))
a.text(crit + 0.4, 215, f"Bonferroni\n$|z^*|$ = {crit:.2f}", fontsize=7, color=INK2, va="top")
a.set_xlim(-15, 15)
a.set_ylim(0, 400)
a.set_xlabel(r"Standardized specialty $z^*$ (clipped at $\pm$15)")
a.set_ylabel("Pairs")
a.legend(loc="upper right", fontsize=7, handlelength=1.0, borderaxespad=0.2)
panel_label(a, "a")
TESTS = [("hc", "HC", BLUE, "o"), ("hc_plus", "HC+", ORANGE, "s"), ("max_abs_z", "min-P", AQUA, "^"),
         ("chi2", r"$\chi^2$", YELLOW, "D")]
for ax, th, lab, tx in [(axes[1], 0.3, "b", (0.96, 0.06, "right", "bottom")), (axes[2], 0.7, "c", (0.04, 0.97, "left", "top"))]:
    g = grid[grid.theta == th].sort_values("r")
    ax.axhline(0.05, color=MUTED, lw=0.7, ls=(0, (3, 3)))
    for key, name, col, mk in TESTS:
        ax.plot(g.r, g[key], color=col, lw=1.3, marker=mk, ms=4, mec="white", mew=0.6, label=name, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks(sorted(g.r.unique()), [f"{v:g}" for v in sorted(g.r.unique())])
    ax.minorticks_off()
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel(r"Signal strength $r$")
    ax.text(tx[0], tx[1], rf"$\vartheta$ = {th:g}: {int(g.s.iloc[0])} specialties", transform=ax.transAxes,
            fontsize=7.5, ha=tx[2], va=tx[3])
    panel_label(ax, lab)
axes[1].set_ylabel("Power")
axes[2].tick_params(labelleft=False)
axes[2].legend(loc="lower right", fontsize=7, handlelength=1.4)
save(fig, "fig4_q3.pdf")
print("wrote", sorted(os.listdir(OUT)))
