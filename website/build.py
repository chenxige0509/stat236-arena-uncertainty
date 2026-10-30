"""Build the project webpage: website/template.html + data from results/ -> docs/index.html (a single static file).

    python website/build.py        (from the repo root)

docs/index.html is self-contained apart from Google Fonts, so it can be served by GitHub Pages ("main" branch,
"/docs" folder) or opened directly. Every number on the page comes from the committed results; nothing is simulated.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import norm

R = lambda name, **k: pd.read_csv(f"results/{name}", **k)
PROCS = ["arena", "bonferroni", "holm", "holm_onesided", "pairwise_bonferroni", "chi2_scheffe", "chi2"]

# ---------------------------------------------------------------- real 2024 board (Q2 applied to the real fit)
real = R("q2_real.csv").sort_values("rating", ascending=False)
board = [{"m": r.model, "r": round(r.rating, 1), "ci": round(1.96 * r.se, 1), "n": int(r.battles),
          **{p: [int(getattr(r, f"{p}_best")), int(getattr(r, f"{p}_worst"))] for p in PROCS}}
         for r in real.itertuples()]

# ---------------------------------------------------------------- Q1: per-dataset simultaneous coverage
q1 = np.load("results/q1.npz", allow_pickle=True)
ci_all = q1["ci_cover"].astype(bool).all(1)
rs_all = q1["rs_cover"].astype(bool).all(1)
q1s = R("q1_summary.csv", index_col=0)["value"]

# ---------------------------------------------------------------- Q2: summary, sweep, stress worlds
s = R("q2_summary.csv", index_col=0)
SIM, WID = "rank interval coverage, simultaneous (all models)", "rank interval width, mean"
POW = "[declared orderings] power: share of true orderings declared"
FALSE = "[declared orderings] P(any false ordering)"
summary = {p: {"cover": round(s.loc[SIM, p], 3), "width": round(s.loc[WID, p], 3), "power": round(s.loc[POW, p], 3),
               "anyFalse": round(s.loc[FALSE, p], 3),
               "realWidth": round(float((real[f"{p}_worst"] - real[f"{p}_best"] + 1).mean()), 1)} for p in PROCS}
sw = R("q2_sweep.csv")
sweep = {"z": sw.z.round(3).tolist(), "cover": sw.rs_all_cover.round(3).tolist(),
         "width": sw.rs_width.round(3).tolist(), "ciCover": sw.ci_all_cover.round(3).tolist()}
WORLDS = [("", "2024 strengths"), ("_scale0.25", "Strengths × 0.25"), ("_tiers20s0.05", "20 tiers, near ties"),
          ("_tiers20", "20 tiers, exact ties"), ("_tiers10", "10 tiers, exact ties"), ("_scale0", "All models tied")]
stress = [{"world": name, **{p: round(R(f"q2{w}_summary.csv", index_col=0).loc[SIM, p], 3)
                             for p in ["arena", "bonferroni", "holm"]}} for w, name in WORLDS]

# ---------------------------------------------------------------- Q3: specialties
q3 = R("q3_pairs.csv")
N = len(q3)
crit = float(norm.isf(0.05 / (2 * N)))
top_models = (q3.groupby("model").overall_rating.first().sort_values(ascending=False).index[:40]).tolist()
CATS = ["coding", "math", "instruction_following", "hard_prompts", "chinese", "russian", "multi_turn", "long_query"]
cells = [{"m": r.model, "c": CATS.index(r.category), "pts": round(r.specialty_points, 1), "z": round(r.z, 2),
          "n": int(r.battles)} for r in q3[q3.model.isin(top_models)].itertuples()]
g = R("q3_global.csv").set_index("statistic")

data = {
    "board": board, "summary": summary, "sweep": sweep, "stress": stress,
    "q1": {"ciAll": "".join("1" if x else "0" for x in ci_all), "rsAll": "".join("1" if x else "0" for x in rs_all),
           "ciEach": round(q1s["score CI coverage, per model (nominal 0.95)"], 3),
           "rsEach": round(q1s["rank spread coverage, per model"], 4),
           "missPerRep": round(q1s["score CIs missing per replicate"], 2)},
    "q3": {"models": top_models, "cats": ["Coding", "Math", "Instruction following", "Hard prompts", "Chinese",
                                          "Russian", "Multi-turn", "Long query"],
           "cells": cells, "allZ": q3.z.abs().round(3).tolist(), "crit": round(crit, 3), "N": N,
           "bh": int(q3.bh.sum()), "hcPlus": round(g.loc["hc_plus", "observed"], 0),
           "hcPlus95": round(g.loc["hc_plus", "null_95"], 1)},
}
tpl = open("website/template.html").read()
assert "/*__DATA__*/null" in tpl
out = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
open("docs/index.html", "w").write(out)
print(f"docs/index.html: {len(out) / 1024:.0f} KB; {len(board)} models, {len(cells)} heatmap cells, crit {crit:.3f}")
