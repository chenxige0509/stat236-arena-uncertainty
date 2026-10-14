"""Step 0: confirm we reproduce Arena's 2024 pipeline before Q1.

Inputs:
  - clean_battle_20240814_public.json (via prepare_data.py): battle records only, with no ratings. We fit BT on it.
  - elo_results_20240813.pkl (text / full): the official output (ratings, bootstrap intervals, ranks, the 100
    bootstrap fits), the closest snapshot to the dump. This is the answer key.
  The board was computed from LMSYS's internal battle log; the JSON is its public release, which lacks some recent
  battles (e.g. hidden pre-release models), so recent models cannot match exactly. See (1).
Checks:
  (1) data:      the same battles (anony, dedup-sampled, up to the snapshot time) give the same battle counts;
  (2) ratings:   arena-rank's BT fit reproduces the published ratings;
  (3) intervals: the published intervals are marginal (per-model bootstrap quantiles, half-width ~ 1.96 SD),
                 not multiplicity-corrected (Bonferroni / chi^2 multipliers are far larger);
  (4) ranks:     the published rank is R_m = 1 + #{m': L_m' > U_m} on those marginal intervals.

What the results establish (numbers from the current run; see README "Step 0"):
  - (1) + (2): our data filter and arena-rank's BradleyTerry reproduce the published leaderboard, so the
    "Arena method" simulated in Q1 is the method behind the site, not our reading of it.
  - (3): the 2024 site showed per-model 95% intervals with no multiplicity correction, although Chiang et al. (2024)
    say the ranking needs simultaneous intervals.
  - (4): the published rank is the paper's R_m (the best end of today's rank spread), computed from those
    uncorrected intervals. This is the practice Q1 evaluates.
  - (3, last line): today's sandwich intervals have about the same width as the 2024 bootstrap ones, so Q1's
    conclusions about arena-rank also apply to the 2024 board.
Not established: anything about the current (2026) site, which this snapshot cannot speak to; the worst end of the
rank spread, which the 2024 site did not show.

    python step0.py
Writes results/step0_comparison.csv.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
from arena_rank.utils.data_utils import PairDataset
from arena_data import load_battles, load_official
from ranking import fit_arena, rank_spread

ANCHOR, ANCHOR_RATING = "mixtral-8x7b-instruct-v0.1", 1114.0   # the 2024 leaderboard pins this model's rating
RENAME = {"chatgpt-4o-latest": "chatgpt-4o-latest-2024-08-08"}   # dump name -> leaderboard name
Z = norm.ppf(0.975)

os.makedirs("results", exist_ok=True)
off, cutoff, boot = load_official()
M_off = len(off)
print(f"official snapshot: {M_off} models, last updated {pd.to_datetime(cutoff, unit='s')} UTC")

# (1) data
# Apply the official filter (anonymous battles, dedup_tag.sampled, tstamp <= snapshot time) and compare battle
# counts per model with the published num_battles.
# Result: 1,657,517 battles; 30 models match exactly and 91 of 128 within +-10 battles. Models released in the last
# few weeks (llama-3.1, gpt-4o-mini, ...) have 1-2k fewer battles even without the time cutoff, so the public dump
# simply lacks some recent battles.
# Shows: our filter is the one the leaderboard used. Without dedup, no model matches within +-10 battles (about
# 244k battles off in total), so the dedup step is confirmed by the data even though we found no line of code
# applying it.
b = load_battles(cutoff=cutoff)
for c in ["model_a", "model_b"]:
    b[c] = b[c].replace(RENAME)
nb = pd.concat([b.model_a, b.model_b]).value_counts()
cmp = off.set_index("model").join(nb.rename("our_battles"), how="left")
d = cmp.our_battles - cmp.num_battles
print(f"\n(1) {len(b):,} battles, {len(nb)} models (official {M_off}); not on the board: {sorted(set(nb.index) - set(off.model))}")
print(f"    battle count diff (ours - official): exact {int((d == 0).sum())}, |diff| <= 10: {int((d.abs() <= 10).sum())}, "
      f"median relative diff {np.median(d / cmp.num_battles):+.2%}")
print("    largest gaps (recent models; the public dump lacks some recent battles):")
print(cmp.assign(diff=d).sort_values("diff").head(5)[["num_battles", "our_battles", "diff"]].to_string())

# (2) ratings
# Fit arena-rank's BradleyTerry (ties = 0.5) on these battles. arena-rank centers ratings at mean 1000 while the
# 2024 board pinned mixtral at 1114, so shift ours to the same anchor before comparing.
# Result: median |diff| 0.06 points, correlation 0.99999; among models whose battle counts match, max |diff| 0.3.
# The largest gap (about 3 points, llama-3.1-8b) is a model with missing battles from (1).
# Shows: arena-rank's estimator gives the same point estimates as the 2024 FastChat code (both are the BT MLE with
# ties as half a win), so the estimator in Q1 is the one behind the site. Remaining gaps come from the data, not
# the method.
ds = PairDataset.from_pandas(b[["model_a", "model_b", "winner"]])
fit, V = fit_arena(ds, full_cov=True)
fit = fit.set_index("competitors")
shift = ANCHOR_RATING - fit.loc[ANCHOR, "ratings"]          # arena-rank centers at mean 1000; 2024 pinned the anchor
cmp["our_rating"] = fit.ratings + shift
k = list(fit.index).index(ANCHOR)
dv = np.diag(V) + V[k, k] - 2 * V[:, k]                       # Var(rating_m - rating_anchor): matches the anchored bootstrap
cmp["our_se_vs_anchor"] = pd.Series(np.sqrt(np.maximum(dv, 0)), index=fit.index)
cmp["our_se"] = np.sqrt(fit.variances)
err = cmp.our_rating - cmp.rating
print(f"\n(2) rating diff (ours - official), anchored at {ANCHOR} = {ANCHOR_RATING:g}: "
      f"median |diff| {err.abs().median():.2f}, max |diff| {err.abs().max():.2f} ({err.abs().idxmax()}), "
      f"corr {np.corrcoef(cmp.our_rating, cmp.rating)[0, 1]:.5f}")
old = d.abs() <= 10
print(f"    models whose battle counts match (|diff| <= 10, n={old.sum()}): max |rating diff| {err[old].abs().max():.2f}")

# (3) intervals
# Two independent pieces of evidence that the published intervals are marginal (per-model), not corrected:
#  (a) Construction. Each model's 2.5% / 97.5% quantile over the 100 bootstrap fits stored in the pickle equals the
#      published rating_q025 / rating_q975 exactly (difference 0). A per-column 2.5%/97.5% quantile is by definition
#      a per-model 95% interval; a Bonferroni or simultaneous interval would need far more extreme quantiles.
#  (b) Width. Half-width / bootstrap SD has median 1.90 [1.63, 2.26]. It is below 1.96 only because 100 draws are
#      too few to reach the true tails: a normal distribution sampled 100 times gives 1.88 [1.70, 2.09] (benchmark
#      below). A correction would need 3.55 (Bonferroni) or 12.42 (chi^2 set) for M = 128.
# Shows: the 2024 site displayed uncorrected intervals, which settles the open question in Proposal Sec. 1.4 for this
# snapshot. Last line: our sandwich SE / official bootstrap SD has median 0.98, so arena-rank's sandwich intervals
# (simulated in Q1) and the 2024 bootstrap intervals have about the same width.
bq = boot.quantile([0.025, 0.975]).T.reindex(cmp.index)
q_err = max(np.abs(bq[0.025] - cmp.rating_q025).max(), np.abs(bq[0.975] - cmp.rating_q975).max())
assert q_err < 1e-8
sd = np.sqrt(cmp.variance)
ratio = ((cmp.rating_q975 - cmp.rating_q025) / 2 / sd).drop(ANCHOR)   # anchor has a zero-width interval
# What that ratio looks like for an exactly normal bootstrap distribution estimated from the same number of draws:
# sample quantiles of B = 100 draws sit inside +-1.96 SD, so a marginal interval gives ~1.88, not 1.96.
draws = np.random.default_rng(0).standard_normal((20000, len(boot)))
qd = np.quantile(draws, [0.025, 0.975], axis=1)
bench = (qd[1] - qd[0]) / 2 / draws.std(1, ddof=1)
mult = {"marginal 95%": Z, "Bonferroni": norm.ppf(1 - 0.05 / (2 * M_off)), "chi^2 simultaneous": np.sqrt(chi2.ppf(0.95, M_off - 1))}
print(f"\n(3) published intervals = per-model 2.5%/97.5% quantiles of the {len(boot)} bootstrap fits in the pickle: "
      f"max |difference| {q_err:.1e} (every bootstrap pins the anchor)")
print(f"    half-width / bootstrap SD: median {ratio.median():.2f}  [{ratio.min():.2f}, {ratio.max():.2f}]")
print(f"    same ratio for a normal distribution and {len(boot)} draws (marginal benchmark): median {np.median(bench):.2f}, "
      f"middle 95% [{np.quantile(bench, 0.025):.2f}, {np.quantile(bench, 0.975):.2f}]")
print("    multiplier each correction would need: " + ", ".join(f"{k} {v:.2f}" for k, v in mult.items()))
se_ratio = (cmp.our_se_vs_anchor / sd).drop(ANCHOR)
print(f"    our sandwich SE (of rating - anchor) / official bootstrap SD: median {se_ratio.median():.2f} "
      f"[{se_ratio.quantile(.05):.2f}, {se_ratio.quantile(.95):.2f}] (5-95%)")

# (4) ranks
# Recompute R_m = 1 + #{m': L_m' > U_m} (Chiang et al., 2024; the best end of today's rank spread) from the published
# intervals and compare with the published final_ranking.
# Result: not equal, but off by a model-specific constant: +5 for 112 models, 0-4 for the top 16.
# Explanation: the board was ranked while it still listed 5 models that were removed before publishing
# (pre-release anonymous models). Then gap_m = #{hidden h: L_h > U_m}, which must be >= 0 and can only grow as U_m
# falls. Both hold, so the data are consistent with this explanation (we cannot see the hidden models to confirm it).
# Shows: the published rank is the paper's R_m computed from the uncorrected intervals of (3). Chiang et al. say this
# rank is valid only with simultaneous intervals; the site used marginal ones. That is the practice Q1 evaluates.
best, worst = rank_spread(cmp.rating_q025, cmp.rating_q975)
cmp["R_recomputed"], cmp["worst_recomputed"] = best, worst
gap = cmp.final_ranking - cmp.R_recomputed
print(f"\n(4) published final_ranking - 1 - #{{L_m' > U_m}} on the published table: {gap.value_counts().sort_index().to_dict()}")
# If the board was ranked on a superset that also had h hidden models, gap_m = #{hidden: L_h > U_m}, which is >= 0 and
# non-increasing in U_m.
by_u = gap[cmp.rating_q975.sort_values(ascending=False).index]
print(f"    gap >= 0: {bool((gap >= 0).all())}; non-decreasing as U_m falls: {bool(by_u.is_monotonic_increasing)} "
      f"-> consistent with ranks computed with {gap.max()} unlisted (hidden) models present")

# Our own fit, ranked the same way with marginal sandwich intervals.
# Result: mean rank spread width 8.5 on the published (bootstrap) intervals vs 7.9 on ours (sandwich).
# Shows: swapping the 2024 bootstrap for today's sandwich barely changes the rank spreads.
L, U = fit.rating_lower + shift, fit.rating_upper + shift
ob, ow = rank_spread(L, U)
fit["our_best"], fit["our_worst"] = ob, ow
cmp = cmp.join(fit[["our_best", "our_worst"]])
print(f"    rank spreads on the published intervals: mean width {(worst - best + 1).mean():.1f}; on ours (sandwich): "
      f"{(cmp.our_worst - cmp.our_best + 1).mean():.1f}")

cols = ["rating", "our_rating", "rating_q025", "rating_q975", "variance", "our_se_vs_anchor", "num_battles", "our_battles",
        "final_ranking", "R_recomputed", "worst_recomputed", "our_best", "our_worst"]
out = cmp[cols].sort_values("rating", ascending=False)
out.to_csv("results/step0_comparison.csv")
print("\n" + out.head(15).round(1).to_string())
