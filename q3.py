"""Q3 Step 2: are the category "specialties" on the real 2024 board real?  (Proposal Sec. 3.3)

For every (model, category) pair with at least MIN_BATTLES battles in the category, z_mc compares the model's score
in the category with what its overall Bradley-Terry strength predicts (specialty.py). Then, in this order:
  Step 1  is there any real specialty?  global tests: Higher Criticism, min-P, chi^2
  Step 2  if so, which ones?            Bonferroni (FWER 5%) and Benjamini-Hochberg (FDR 10%)

Calibration. beta_hat is fit on all battles, including the category's, so z_mc is shrunk towards 0 and the z_mc are
correlated. We therefore simulate the null "no specialties" with the Q1 machinery: true strengths = the real overall
fit, the real battles (same pairs, same categories, same counts), outcomes redrawn from BT with the observed tie
rate, and beta_hat refit on every draw. Each z_mc is standardized against its null mean and SD (z*), and each global
statistic gets its p-value from its own null distribution, which accounts for the dependence.

Specialty size, for reading the results: the category board's rating minus the overall rating, both fitted by
arena-rank (the difference Q3 Step 0 checks against the board), then centered within the category by subtracting its
battle-weighted mean over the models. On the board both ratings are anchored at mixtral = 1114, so the raw difference
is relative to mixtral; centering makes it relative to the category's models, like z (sum of O - E is 0 in each
category).

What the results establish (numbers from the current run, 500 null replicates):
  - Calibration matters: under the null the raw z_mc have SD 0.70-1.05 (median 0.92), not 1.
  - Step 1: every global test rejects at the smallest p the simulation can give (1/501). HC+ = 330 against a null
    95% point of 3.1; chi^2 = 21,210 against 1,028; max |z*| = 40 against 4.0. Plain HC is ~1e62 because p_(1)
    underflows to ~0, which is why HC+ (p > 1/N only) is the usable version here.
  - Step 2: Bonferroni finds 275 of the 958 tested pairs, BH (10%) 525; naive p < 0.05 gives 521, where about 48
    would be expected by chance. 94 of the 129 models have at least one Bonferroni specialty, in every category
    (17-40% of pairs). Their size (centered category rating - overall rating) has median |24.8| points.
  - So on the 2024 board the category effects are neither rare nor weak: they are common and often large (the
    strongest are language effects, e.g. llama-3-70b-instruct is 73 points weaker in Chinese, z = -40). That is
    the regime where every global test works; HC's advantage is for rare and weak effects, which the simulation in
    Q3 Step 3 examines.
Not established: that each specialty is a property of the model. The null is exact Bradley-Terry with fixed
strengths, so anything else that varies with the category (e.g. strengths drifting over time while the category mix
changes, voters who differ between languages) also shows up as a specialty.

    python q3.py [--reps 500] [--seed 0]
Writes results/q3_pairs.csv (one row per tested pair), results/q3_global.csv, results/q3_null.npz.
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
from arena_rank.utils.data_utils import PairDataset
from arena_data import ANCHOR, ANCHOR_RATING
from categories import CATEGORIES, MIN_BATTLES, category_mask
from ranking import fit_arena
from simulate import World, fixed_shape_dataset
from specialty import real_world, specialty_z, two_sided_p, global_stats, bonferroni, benjamini_hochberg

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=500, help="null replicates for the calibration")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
os.makedirs("results", exist_ok=True)
rng = np.random.default_rng(args.seed)

# ---------------------------------------------------------------- real data
W = real_world()
b, names, pairs, n_pair, G, beta_hat, tie = W.battles, W.names, W.pairs, W.n_pair, W.G, W.beta_hat, W.tie
nb, mi, ci, N = W.nb, W.mi, W.ci, W.N
tested = nb >= MIN_BATTLES
O, E, V, z = specialty_z(G, G.cells, beta_hat)
print(f"{len(b):,} battles, {len(names)} models, {len(CATEGORIES)} categories, {len(G.n):,} (pair, pattern) groups")
print(f"tested pairs (>= {MIN_BATTLES} battles): {N} of {tested.size}; per category: "
      + ", ".join(f"{c} {int(tested[:, k].sum())}" for k, c in enumerate(CATEGORIES)))

# ---------------------------------------------------------------- null calibration
world = World(pairs, n_pair, beta_hat, tie)
null_z = np.zeros((args.reps, N))
t0 = time.time()
for r in range(args.reps):
    cells = G.draw(world.probs, rng)
    beta_r = fit_arena(fixed_shape_dataset(pairs, G.pair_cells(cells))).beta.values
    null_z[r] = specialty_z(G, cells, beta_r)[3][mi, ci]
    if (r + 1) % 100 == 0:
        print(f"  null {r + 1}/{args.reps}, {time.time() - t0:.0f}s", flush=True)
mu0, sd0 = null_z.mean(0), null_z.std(0, ddof=1)
print(f"null z: mean of means {mu0.mean():+.3f}, SD median {np.median(sd0):.3f} [{sd0.min():.3f}, {sd0.max():.3f}] "
      f"(an uncalibrated z would assume 0 and 1)")

zs = (z[mi, ci] - mu0) / sd0
null_zs = (null_z - mu0) / sd0
p = two_sided_p(zs)

# ---------------------------------------------------------------- Step 1: global tests
obs = global_stats(zs)
null_g = pd.DataFrame([global_stats(row) for row in null_zs])
small_is_evidence = {"min_p"}
rows = []
for k, v in obs.items():
    nul = null_g[k].values
    extreme = (nul <= v) if k in small_is_evidence else (nul >= v)
    rows.append({"statistic": k, "observed": v, "null_median": float(np.median(nul)),
                 "null_95": float(np.quantile(nul, 0.05 if k in small_is_evidence else 0.95)),
                 "p_bootstrap": (1 + extreme.sum()) / (args.reps + 1)})
glob = pd.DataFrame(rows)
print("\nStep 1: is there any specialty? (p-values from the null simulation)")
print(glob.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

# ---------------------------------------------------------------- Step 2: identification
bonf, bh = bonferroni(p), benjamini_hochberg(p)


def board_rating(sub):
    f = fit_arena(PairDataset.from_pandas(sub[["model_a", "model_b", "winner"]])).set_index("competitors")
    return f.ratings + (ANCHOR_RATING - f.loc[ANCHOR, "ratings"])


overall = board_rating(b)
spec = np.full(tested.shape, np.nan)
for k, c in enumerate(CATEGORIES):
    d = (board_rating(b[category_mask(b, c)]) - overall).reindex(names).values
    w = np.where(np.isfinite(d), nb[:, k], 0)
    spec[:, k] = d - np.nansum(w * d) / w.sum()

out = pd.DataFrame({
    "category": np.array(CATEGORIES)[ci], "model": names[mi], "battles": nb[mi, ci].astype(int),
    "overall_rating": overall.reindex(names).values[mi], "specialty_points": spec[mi, ci],
    "observed_score": O[mi, ci], "expected_score": E[mi, ci], "z_raw": z[mi, ci],
    "null_mean": mu0, "null_sd": sd0, "z": zs, "p": p, "bonferroni": bonf, "bh": bh,
}).sort_values("p")
out.to_csv("results/q3_pairs.csv", index=False)
glob.to_csv("results/q3_global.csv", index=False)
np.savez_compressed("results/q3_null.npz", null_z=null_z, model=names[mi], category=np.array(CATEGORIES)[ci])

print(f"\nStep 2: which ones?  Bonferroni (alpha 0.05, bar p < {0.05 / N:.1e}): {int(bonf.sum())};  "
      f"BH (q 0.10): {int(bh.sum())};  naive p < 0.05: {int((p < 0.05).sum())} (about {0.05 * N:.0f} by chance)")
print("by category (Bonferroni / BH):", ", ".join(
    f"{c} {int(out[(out.category == c)].bonferroni.sum())}/{int(out[(out.category == c)].bh.sum())}" for c in CATEGORIES))
print("\nstrongest specialties:")
print(out.head(20)[["category", "model", "battles", "overall_rating", "specialty_points", "z", "p", "bonferroni", "bh"]]
      .to_string(index=False, float_format=lambda x: f"{x:.3g}"))
