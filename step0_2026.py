"""Step 0 for the current site: are the 2026 leaderboard's intervals still per-model, and is its rank spread computed
from them?  Step 0 answered this for the 2024-08 snapshot only.

Inputs:
  - lmarena-ai/leaderboard-dataset on Hugging Face (downloaded to data/leaderboard_dataset/): every published board
    with rating, rating_lower, rating_upper, variance and vote_count per model. Subsets `text` (no style control)
    and `text_style_control`.
  - results/site_text_overall_20261008.csv: the rank spread, score, +- and votes the site displayed for the top 59
    models of the Text Overall board dated 2026-10-08, read from arena.ai/leaderboard/text on 2026-10-12. The site
    shows only the current board (2026-10-27 by 2026-10-28), so this record is the only site view we can check.
Checks:
  (1) which subset the site shows: compare the site's scores and +- with both subsets on 2026-10-08;
  (2) interval type: half-width / sqrt(variance) on every model of every 2026 text board, against the multiplier a
      Bonferroni or chi^2 correction would need for that board's number of models;
  (3) rank spread: recompute [best, worst] (ranking.rank_spread, the blog formula) from the 2026-10-08 intervals of all
      414 models and compare with the site's record.

What the results establish (current run):
  - (1) The site's default Text view is the style-controlled board: on 2026-10-08 all 59 recorded scores match
    `text_style_control` to rounding (max 0.50), all 59 +- equal its rounded half-widths and all vote counts match.
    The plain `text` subset differs by up to 54 points.
  - (2) Every interval on every 2026 text board is rating +- 1.9600 sqrt(variance): 4,773 boards, 1.5M model rows,
    smallest and largest ratio both 1.9600. These are per-model 95% intervals. A Bonferroni or chi^2 correction would
    need 2.24-3.85 or 2.0-21.6, depending on the board's number of models.
  - (3) The blog's formula applied to those intervals reproduces the site's rank spread exactly: best and worst rank
    equal for 59 of 59 models.
So on the current site, as in 2024, the rank spread is computed from uncorrected per-model intervals. This settles
the question left open in the proposal (Sec. 1.4) and in Step 0, whose rank-spread code is not public.
Not established: anything about boards other than Text; only one site view (2026-10-08, top 59) was recorded.

    python step0_2026.py
Writes results/step0_2026_intervals.csv (one row per board) and results/step0_2026_rank_spread.csv.
"""
import os
import urllib.request
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
from ranking import rank_spread

DIR = "data/leaderboard_dataset"
URL = "https://huggingface.co/api/datasets/lmarena-ai/leaderboard-dataset/parquet/{sub}/full/0.parquet"
SITE_DATE = "2026-10-08"
os.makedirs(DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)


def load(sub):
    path = os.path.join(DIR, f"{sub}_full.parquet")
    if not os.path.exists(path):
        urllib.request.urlretrieve(URL.format(sub=sub), path)
    d = pd.read_parquet(path)
    d["half_width"] = (d.rating_upper - d.rating_lower) / 2
    return d


boards = {sub: load(sub) for sub in ["text", "text_style_control"]}
site = pd.read_csv("results/site_text_overall_20261008.csv")

# (1) which subset is the site's default view
print(f"(1) site record: top {len(site)} models of Text Overall, board dated {SITE_DATE}")
for sub, d in boards.items():
    o = d[(d.category == "overall") & (d.leaderboard_publish_date == SITE_DATE)].set_index("model_name")
    j = site.set_index("model").join(o[["rating", "half_width", "vote_count"]], how="left", rsuffix="_board")
    print(f"    {sub:20s} matched {int(j.rating.notna().sum())}/{len(site)}; "
          f"max |site score - rating| {np.abs(j.score - j.rating).max():.2f}; "
          f"site +- = rounded half-width: {int((np.round(j.half_width_board) == j.half_width).sum())}/{len(site)}; "
          f"votes equal: {int((j.vote_count == j.votes).sum())}/{len(site)}")

# (2) interval type on every 2026 text board
rows = []
for sub, d in boards.items():
    d26 = d[d.leaderboard_publish_date.str.startswith("2026")]
    for (date, cat), g in d26.groupby(["leaderboard_publish_date", "category"]):
        ok = g.variance > 0
        r = (g.half_width[ok] / np.sqrt(g.variance[ok]))
        M = len(g)
        rows.append({"subset": sub, "date": date, "category": cat, "models": M,
                     "ratio_median": r.median(), "ratio_min": r.min(), "ratio_max": r.max(),
                     "bonferroni_multiplier": norm.ppf(1 - 0.05 / (2 * M)),
                     "chi2_multiplier": np.sqrt(chi2.ppf(0.95, M - 1))})
iv = pd.DataFrame(rows)
iv.to_csv("results/step0_2026_intervals.csv", index=False)
print(f"\n(2) half-width / sqrt(variance) over {len(iv):,} boards (both subsets, {iv.date.nunique()} dates in 2026, "
      f"{iv.category.nunique()} categories), {int(iv.models.sum()):,} model rows:")
print(f"    median of board medians {iv.ratio_median.median():.4f}; smallest {iv.ratio_min.min():.4f}, "
      f"largest {iv.ratio_max.max():.4f}  (marginal 95%: {norm.ppf(0.975):.4f})")
print(f"    a correction would need: Bonferroni {iv.bonferroni_multiplier.min():.2f}-{iv.bonferroni_multiplier.max():.2f}, "
      f"chi^2 {iv.chi2_multiplier.min():.1f}-{iv.chi2_multiplier.max():.1f}")

# (3) rank spread on the site's board
sc = boards["text_style_control"]
o = sc[(sc.category == "overall") & (sc.leaderboard_publish_date == SITE_DATE)].reset_index(drop=True)
best, worst = rank_spread(o.rating_lower, o.rating_upper)
o["our_best"], o["our_worst"] = best, worst
cmp = site.merge(o[["model_name", "rating", "half_width", "our_best", "our_worst"]],
                 left_on="model", right_on="model_name", how="left")
cmp["best_equal"], cmp["worst_equal"] = cmp.best == cmp.our_best, cmp.worst == cmp.our_worst
cmp.drop(columns="model_name").to_csv("results/step0_2026_rank_spread.csv", index=False)
print(f"\n(3) rank spread recomputed from the {len(o)} published intervals of {SITE_DATE} (style control), "
      f"vs the site's top {len(site)}:")
print(f"    best rank equal: {int(cmp.best_equal.sum())}/{len(cmp)}; worst rank equal: {int(cmp.worst_equal.sum())}/{len(cmp)}")
bad = cmp[~(cmp.best_equal & cmp.worst_equal)]
if len(bad):
    print(bad[["rank", "model", "best", "our_best", "worst", "our_worst"]].to_string(index=False))
print(f"    mean width on the site's top {len(site)}: {(cmp.worst - cmp.best + 1).mean():.1f}; "
      f"over all {len(o)} models: {(worst - best + 1).mean():.1f}")
