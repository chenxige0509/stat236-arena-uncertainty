"""Q3 Step 3: which test finds rare and weak specialties?  (Proposal Sec. 3.3, "Validation by simulation")

Step 2 found the 2024 specialties common and large, a regime where every global test works. This step plants
specialties of known number and size into the same simulated Arena as Step 2's null, and measures how often each
procedure finds them.

World: the real battles (same pairs, same category patterns, same counts), true overall strengths = the real
overall fit, ties at the observed rate (exactly Step 2's null), plus s planted specialties. A planted pair (m, c)
gets delta_mc added to model m's strength in every battle of category c; a battle in several categories gets the sum.

Grid, following the Rare/Weak model of Lecture 3 with N = number of tested pairs:
  rarity    s = round(N^(1 - theta)) planted pairs, chosen at random among the tested pairs, random signs;
  strength  delta_mc is set so that, before beta_hat is refit, the planted pair's z has mean mu = sqrt(2 r log N):
            delta_mc = mu sqrt(Var_mc) / I_mc, with I_mc = sum p(1 - p) and Var_mc = sum [p(1 - p) - t/4] over the
            pair's battles (t = tie probability). Refitting beta_hat absorbs part of it, so the realized mean |z*|
            is reported too.
Each replicate: draw outcomes, refit beta_hat, compute z*, standardized with Step 2's null mean and SD
(results/q3_null.npz), then
  global tests   reject when the statistic passes its null 95% point: HC, HC+, min-P (max |z*|), chi^2. The
                 rejection rate is the power. The 95% points come from fresh null draws, not from Step 2's: those
                 draws also set the standardization, and reusing them makes HC+ and chi^2 about twice too liberal
                 (size 0.10 instead of 0.05). A second set of fresh null draws checks the size of every test;
  identification Bonferroni (5%) and BH (10%). Categories overlap (many coding battles are also hard prompts), so a
                 specialty planted in (m, c) also changes m's true score in other categories. A discovery is therefore
                   right      if it is a planted pair with the right sign;
                   spillover  if it is not planted but its model has a planted specialty (a real, induced effect);
                   false      if its model has no planted specialty, or it is a planted pair with the wrong sign.
                 Reported: true positive rate (right / planted), false discovery proportion (false / discoveries),
                 spillover share, and P(any false discovery).

What the results establish (numbers from the current run: 200 replicates per cell, 2 x 400 fresh null draws):
  - Size: HC 0.060, HC+ 0.050, min-P 0.052, chi^2 0.083 (MC SE 0.011, plus the error of the estimated 95% point).
  - Dense effects (theta <= 0.4, s >= 61): chi^2 and HC+ are the most powerful and about equal; min-P is far behind
    (theta 0.3, r 0.2: chi^2 1.00, HC+ 1.00, min-P 0.57).
  - theta 0.5-0.6: chi^2 and HC+ still about equal; plain HC catches up with them.
  - Sparsest effects (theta 0.7, s = 8): chi^2 falls behind and plain HC is best (r 0.8: HC 0.84, min-P 0.74,
    HC+ 0.72, chi^2 0.62). This is the ordering Lecture 3 predicts, but on a narrow range: with N = 958 the
    asymptotic separation between HC and chi^2 only shows at the sparse end of the grid.
  - Identification: Bonferroni finds about 53% of the planted specialties at mu = 4.5 (27 points) and 16-19% at
    mu = 3.3; BH about 76-93% and 32-74%. BH's false discovery proportion stays near its 10% target (0.09-0.12,
    MC SE 0.01-0.02) and Bonferroni's P(any false discovery) near 5% (0.02-0.09). Separately, 1-23% of discoveries
    are spillovers into other categories of a planted model, which category overlap makes real effects.
So the HC advantage of the theory exists here but is modest; on the real 2024 board (Step 2) the effects are dense
and large, where chi^2 and HC+ both work and HC's advantage does not matter.

    python q3_sim.py [--reps 200] [--seed 1]
Needs results/q3_null.npz (python q3.py). Writes results/q3_sim.csv (one row per grid cell, with Monte Carlo SEs).
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
from categories import CATEGORIES
from ranking import fit_arena, ALPHA
from simulate import fixed_shape_dataset
from specialty import real_world, specialty_z, two_sided_p, global_stats, bonferroni, benjamini_hochberg

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=200, help="replicates per grid cell")
ap.add_argument("--null-reps", type=int, default=400, help="fresh null draws for the critical values (and as many for the size check)")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--thetas", default="0.2,0.3,0.4,0.5,0.6,0.7")
ap.add_argument("--rs", default="0.1,0.2,0.4,0.8,1.5")
args = ap.parse_args()
os.makedirs("results", exist_ok=True)
rng = np.random.default_rng(args.seed)

W = real_world()
G, beta, tie, mi, ci, N = W.G, W.beta_hat, W.tie, W.mi, W.ci, W.N
null = np.load("results/q3_null.npz", allow_pickle=True)
assert (null["model"] == W.names[mi]).all() and (null["category"] == np.array(CATEGORIES)[ci]).all()
mu0, sd0 = null["null_z"].mean(0), null["null_z"].std(0, ddof=1)
TESTS = ["hc", "hc_plus", "max_abs_z", "chi2"]
i_g, j_g = G.pairs[G.gp, 0], G.pairs[G.gp, 1]
M, C = G.M, len(CATEGORIES)


def group_probs(delta):
    """(G, 3) win / tie / loss probabilities of the lower-index model, with planted specialties delta (M, C)."""
    d_i = (G.pattern * delta[i_g]).sum(1)
    d_j = (G.pattern * delta[j_g]).sum(1)
    p = 1 / (1 + np.exp(-((beta[i_g] + d_i) - (beta[j_g] + d_j))))
    t = np.minimum(tie, 2 * np.minimum(p, 1 - p))                # as simulate.World
    pr = np.clip(np.column_stack([p - t / 2, t, 1 - p - t / 2]), 0, 1)
    return pr / pr.sum(1, keepdims=True), p, t


# delta (logit scale) that gives a planted pair's z mean 1 before the refit: sqrt(Var) / I
_, p0, t0 = group_probs(np.zeros((M, C)))
info, var = np.zeros((M, C)), np.zeros((M, C))
for c in range(C):
    m = G.pattern[:, c]
    for side in (i_g, j_g):
        info[:, c] += np.bincount(side[m], G.n[m] * p0[m] * (1 - p0[m]), M)
        var[:, c] += np.bincount(side[m], G.n[m] * (p0[m] * (1 - p0[m]) - t0[m] / 4), M)
unit = (np.sqrt(var) / info)[mi, ci]
print(f"delta for mu = 1, in rating points: median {np.median(unit) * ALPHA:.1f} "
      f"[{np.min(unit) * ALPHA:.1f}, {np.max(unit) * ALPHA:.1f}]")


def null_draw():
    cells = rng.multinomial(G.n, group_probs(np.zeros((M, C)))[0]).astype(float)
    b_r = fit_arena(fixed_shape_dataset(W.pairs, G.pair_cells(cells))).beta.values
    return global_stats((specialty_z(G, cells, b_r)[3][mi, ci] - mu0) / sd0)


t_start = time.time()
null_crit = pd.DataFrame([null_draw() for _ in range(args.null_reps)])
crit = {k: float(np.quantile(null_crit[k], 0.95)) for k in TESTS}
null_size = pd.DataFrame([null_draw() for _ in range(args.null_reps)])
size = {k: float((null_size[k] > crit[k]).mean()) for k in TESTS}
print(f"critical values from {args.null_reps} fresh null draws: " + ", ".join(f"{k} {v:.3g}" for k, v in crit.items()))
print(f"size on {args.null_reps} more null draws (nominal 0.05, MC SE {np.sqrt(0.05 * 0.95 / args.null_reps):.3f}): "
      + ", ".join(f"{k} {v:.3f}" for k, v in size.items()) + f"  ({time.time() - t_start:.0f}s)")


def mc(x):
    x = np.asarray(x, dtype=float)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


thetas = [float(x) for x in args.thetas.split(",")]
rs = [float(x) for x in args.rs.split(",")]
cells = [(None, 0.0)] + [(th, r) for th in thetas for r in rs]     # first cell: no specialties (FDR / FWER at s = 0)
rows = [{"theta": None, "r": 0.0, "s": 0, "mu": 0.0, "delta_points_median": 0.0, "cell": "size check",
         **{k: size[k] for k in TESTS}, **{f"{k}_mc_se": np.sqrt(size[k] * (1 - size[k]) / args.null_reps) for k in TESTS}}]
for k, (th, r) in enumerate(cells):
    s = 0 if th is None else max(1, int(round(N ** (1 - th))))
    mu = np.sqrt(2 * r * np.log(N)) if s else 0.0
    rec = {key: [] for key in TESTS + [f"{a}_{b}" for a in ("bonf", "bh")
                                       for b in ("n", "tpr", "fdp", "spill", "any_false")] + ["planted_abs_z"]}
    for _ in range(args.reps):
        delta = np.zeros((M, C))
        planted = rng.choice(N, s, replace=False) if s else np.array([], dtype=int)
        sign = rng.choice([-1.0, 1.0], s)
        delta[mi[planted], ci[planted]] = sign * mu * unit[planted]
        cells_r = rng.multinomial(G.n, group_probs(delta)[0]).astype(float)   # probs are per group, not per pair
        beta_r = fit_arena(fixed_shape_dataset(W.pairs, G.pair_cells(cells_r))).beta.values
        zs = (specialty_z(G, cells_r, beta_r)[3][mi, ci] - mu0) / sd0
        g = global_stats(zs)
        for key in TESTS:
            rec[key].append(g[key] > crit[key])
        truth = np.zeros(N)
        truth[planted] = sign
        is_planted = truth != 0
        model_planted = np.isin(mi, mi[planted])                # the pair's model has some planted specialty
        p = two_sided_p(zs)
        for name, found in [("bonf", bonferroni(p)), ("bh", benjamini_hochberg(p))]:
            right = found & is_planted & (np.sign(zs) == truth)
            spill = found & ~is_planted & model_planted
            false = found & ~right & ~spill
            n_found = max(found.sum(), 1)
            rec[f"{name}_n"].append(found.sum())
            rec[f"{name}_tpr"].append(right.sum() / s if s else np.nan)
            rec[f"{name}_fdp"].append(false.sum() / n_found)
            rec[f"{name}_spill"].append(spill.sum() / n_found)
            rec[f"{name}_any_false"].append(false.any())
        rec["planted_abs_z"].append(np.abs(zs[planted]).mean() if s else np.nan)
    row = {"theta": th, "r": r, "s": s, "mu": mu,
           "delta_points_median": float(np.median(mu * unit) * ALPHA) if s else 0.0}
    for key, v in rec.items():
        row[key], row[f"{key}_mc_se"] = mc(v) if not np.isnan(v).all() else (np.nan, np.nan)
    rows.append(row)
    print(f"[{k + 1}/{len(cells)}] theta {th} r {r}: s={s} mu={mu:.2f} ({row['delta_points_median']:.1f} pts), "
          f"realized |z*| {row['planted_abs_z']:.2f} | power HC {row['hc']:.2f} HC+ {row['hc_plus']:.2f} "
          f"min-P {row['max_abs_z']:.2f} chi2 {row['chi2']:.2f} | Bonf TPR {row['bonf_tpr']:.2f} "
          f"BH TPR {row['bh_tpr']:.2f} FDP {row['bh_fdp']:.2f} spill {row['bh_spill']:.2f} | {time.time() - t_start:.0f}s", flush=True)

out = pd.DataFrame(rows)
out.to_csv("results/q3_sim.csv", index=False)
print("\npower of the global tests (rows: theta = rarity, columns: r = strength):")
for key in TESTS:
    print(f"\n{key}")
    print(out.dropna(subset=["theta"]).pivot(index="theta", columns="r", values=key).round(2).to_string())
