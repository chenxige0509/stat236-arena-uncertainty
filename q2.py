"""Q2: do multiplicity-corrected procedures reach 95% coverage, and at what cost?  (semi-synthetic; Proposal Sec. 3.2)

World:      the same as q1.py (same truth, seed and draws), so the "arena" rows reproduce results/q1_summary.csv.
Procedures: run side by side on every simulated dataset. Each maps one fit to a set of declared orderings
            above[i, j] = "i ranks above j" (plus score intervals [L, U] if it has them). The rank interval follows as
                best_m = 1 + #{j declared above m},   worst_m = M - #{j declared below m},
            which for score intervals with above = (L_i > U_j) is exactly Arena's rank spread.
  arena       marginal 95% sandwich CIs (the method Q1 evaluates)
  bonferroni  z_{1 - alpha/(2M)} in place of z_{1 - alpha/2}
  chi2        Chiang et al.'s chi^2 simultaneous set projected onto each model: z = sqrt(chi^2_{M-1, 1-alpha})
  chi2_scheffe  the same set, i above j when beta_i > beta_j on the whole set (uses the full covariance)
  holm, bh: stubs below, to be filled in; then pass them with --procedures
Sweep:      rank spreads from beta_hat +- z SE over a grid of z, to find the smallest z at which simultaneous coverage
            reaches 95%. This is an oracle benchmark (it needs the truth), not a procedure: it measures how much of
            Bonferroni's widening the rank spreads actually need.
Real data:  every procedure applied to the real fit, next to Arena's rank spreads.

    python q2.py --reps 1000 [--procedures arena,bonferroni] [--tie-rate 0] [--seed 0]
Writes results/q2{tag}_summary.csv, q2{tag}_per_model.csv, q2{tag}_sweep.csv, q2{tag}.npz, q2_real.csv.
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
from arena_rank.utils.data_utils import PairDataset
from arena_data import load_battles, load_official
from ranking import fit_arena, rank_spread, point_rank
from simulate import World, pair_counts

LEVEL = 0.05


# ---------------------------------------------------------------- procedures
# Each takes one fit (est: the DataFrame from fit_arena; V: the full rating-scale covariance, or None) and returns
# {"above": M x M bool array, above[i, j] = "i declared above j"}, plus "L" and "U" if it has score intervals.
def from_intervals(L, U):
    """Score intervals -> declared orderings: i above j when the intervals don't overlap (Arena's rule)."""
    return {"above": L[:, None] > U[None, :], "L": L, "U": U}


def arena(est, V):
    """Arena's method (Q1): arena-rank's marginal 95% sandwich CIs."""
    return from_intervals(est.rating_lower.values, est.rating_upper.values)


def bonferroni(est, V):
    """Bonferroni: each of the M CIs at level 1 - alpha/M. P(some CI misses) <= M * alpha/M = alpha whatever the
    dependence between the estimates, so all CIs hold at once w.p. >= 95%, and when they all hold every rank
    interval holds too (L_j > U_m forces theta_j > theta_m, and theta_j > theta_m forces U_j > L_m)."""
    z = norm.ppf(1 - LEVEL / (2 * len(est)))
    r, se = est.ratings.values, np.sqrt(est.variances.values)
    return from_intervals(r - z * se, r + z * se)


def chi2_set(est, V):
    """chi^2 simultaneous set of Chiang et al. (2024), {beta: (beta_hat - beta)' V^-1 (beta_hat - beta) <= chi^2_{M-1, 1-alpha}},
    projected onto each model. The projection of that ellipsoid onto coordinate m is beta_hat_m +- sqrt(c V_mm), so this
    is Arena's rule with z = sqrt(chi^2_{M-1, 1-alpha}) (12.47 at M = 129). M - 1 degrees of freedom because
    sum(beta) = 0 leaves V with rank M - 1. All M projections hold whenever the true beta is in the set."""
    c = chi2.ppf(1 - LEVEL, len(est) - 1)
    r, se = est.ratings.values, np.sqrt(np.diag(V))
    return from_intervals(r - np.sqrt(c) * se, r + np.sqrt(c) * se)


def chi2_scheffe(est, V):
    """The same chi^2 set read through pairwise contrasts (Scheffe): i above j when beta_i > beta_j everywhere on the
    ellipsoid, i.e. beta_hat_i - beta_hat_j > sqrt(c) SE(beta_hat_i - beta_hat_j), with Var = V_ii + V_jj - 2 V_ij.
    Valid for the same reason as chi2_set (every contrast holds when beta is in the set), and it declares every
    ordering chi2_set does, since SE(diff) <= SE_i + SE_j. Unlike the overlap rule it uses the covariances.
    Keeps chi2_set's projected score intervals for the score-coverage rows."""
    c = chi2.ppf(1 - LEVEL, len(est) - 1)
    r, v = est.ratings.values, np.diag(V)
    var_diff = np.maximum(v[:, None] + v[None, :] - 2 * V, 0)
    return chi2_set(est, V) | {"above": r[:, None] - r[None, :] > np.sqrt(c * var_diff)}


def pairwise(est, V, correction):
    """Tests of H0: beta_i = beta_j for every pair, with Var(beta_i - beta_j) from V and errors controlled by Holm (FWER)
    or BH (FDR). TODO (Q2): return {"above": H0 rejected and beta_hat_i > beta_hat_j}."""
    raise NotImplementedError


PROCEDURES = {"arena": arena, "bonferroni": bonferroni, "chi2": chi2_set, "chi2_scheffe": chi2_scheffe,
              "holm": lambda est, V: pairwise(est, V, "holm"), "bh": lambda est, V: pairwise(est, V, "bh")}
NEEDS_COV = {"chi2", "chi2_scheffe", "holm", "bh"}


def ranks_from_orderings(above):
    """Rank interval implied by declared orderings. With above = (L_i > U_j) this equals ranking.rank_spread(L, U)."""
    return 1 + above.sum(0), len(above) - above.sum(1)


ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=1000)
ap.add_argument("--procedures", default="arena,bonferroni", help=f"comma-separated, from {list(PROCEDURES)}")
ap.add_argument("--tie-rate", type=float, default=None, help="default: observed tie share; 0 = binary outcomes")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
procs = args.procedures.split(",")
full_cov = bool(NEEDS_COV & set(procs))
os.makedirs("results", exist_ok=True)
rng = np.random.default_rng(args.seed)

# ---------------------------------------------------------------- the true world (as in q1.py)
_, cutoff, _ = load_official()
battles = load_battles(cutoff=cutoff)
real_ds = PairDataset.from_pandas(battles[["model_a", "model_b", "winner"]])
fit = fit_arena(real_ds, full_cov=full_cov)
real, real_V = fit if full_cov else (fit, None)
names = np.array(real_ds.competitors)
tie = battles.winner.str.startswith("tie").mean() if args.tie_rate is None else args.tie_rate
pairs, n = pair_counts(battles, names)
world = World(pairs, n, real.beta.values, tie)
theta, M = world.rating, world.M
r_true = point_rank(theta)
truly_above = theta[:, None] > theta[None, :]
n_ordered = truly_above.sum()
z_bonf = norm.ppf(1 - LEVEL / (2 * M))
Z_GRID = np.unique(np.r_[np.round(np.arange(1.80, 4.001, 0.05), 2), norm.ppf(1 - LEVEL / 2), z_bonf])
print(f"world: M={M} models, {n.sum():,} battles/replicate, tie rate {tie:.3f}; procedures {procs}; "
      f"Bonferroni z = {z_bonf:.3f}")

# ---------------------------------------------------------------- replicates
R = args.reps
store = {p: {"ci_cover": np.full((R, M), np.nan), "rs_cover": np.zeros((R, M), bool),
             "best": np.zeros((R, M), np.int16), "worst": np.zeros((R, M), np.int16)} for p in procs}
orders = {p: [] for p in procs}
est_rating, est_se = np.zeros((R, M)), np.zeros((R, M))
sweep = np.zeros((R, len(Z_GRID), 6))   # all CIs cover, all spreads cover, spread coverage, width, power, any false
t0 = time.time()
for r in range(R):
    fit = fit_arena(world.draw(rng), full_cov=full_cov)
    est, V = fit if full_cov else (fit, None)
    for p in procs:
        out = PROCEDURES[p](est, V)
        best, worst = ranks_from_orderings(out["above"])
        s = store[p]
        if "L" in out:
            s["ci_cover"][r] = (out["L"] <= theta) & (theta <= out["U"])
        s["rs_cover"][r] = (best <= r_true) & (r_true <= worst)
        s["best"][r], s["worst"][r] = best, worst
        row = {}
        # "orderings": the procedure's own declarations; "rs": rank intervals that don't overlap (the site's tie rule)
        for rule, sep in [("orderings", out["above"]), ("rs", worst[:, None] < best[None, :])]:
            row[f"{rule}_declared"], row[f"{rule}_false"] = sep.sum(), (sep & ~truly_above).sum()
        orders[p].append(row)

    rating, se = est.ratings.values, np.sqrt(est.variances.values)
    est_rating[r], est_se[r] = rating, se
    for k, z in enumerate(Z_GRID):
        L, U = rating - z * se, rating + z * se
        b, w = rank_spread(L, U)
        cov = (b <= r_true) & (r_true <= w)
        sep = L[:, None] > U[None, :]
        sweep[r, k] = [((L <= theta) & (theta <= U)).all(), cov.all(), cov.mean(), (w - b + 1).mean(),
                       (sep & truly_above).sum() / n_ordered, (sep & ~truly_above).any()]
    if (r + 1) % 100 == 0:
        print(f"  {r + 1}/{R} replicates, {time.time() - t0:.0f}s", flush=True)

tag = "" if args.tie_rate is None else f"_tie{args.tie_rate:g}"
np.savez_compressed(f"results/q2{tag}.npz", names=names, theta=theta, r_true=r_true, z_grid=Z_GRID,
                    est_rating=est_rating, est_se=est_se, sweep=sweep,
                    **{f"{p}_{k}": v for p in procs for k, v in store[p].items()})

# Same seed, same draws: Arena's rank spreads must equal those saved by q1.py.
q1_path = f"results/q1{tag}.npz"
if "arena" in procs and args.seed == 0 and os.path.exists(q1_path):
    q1 = np.load(q1_path, allow_pickle=True)
    k = min(R, len(q1["best"]))
    same = all((store["arena"][c][:k] == q1[c][:k]).all() for c in ["best", "worst"])
    print(f"arena rank spreads identical to the first {k} replicates in {q1_path}: {same}")


# ---------------------------------------------------------------- summary: estimate and Monte Carlo SE
def mc(x):
    x = np.asarray(x, dtype=float)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


def summarize(s, od):
    rs, width = s["rs_cover"], s["worst"] - s["best"] + 1
    out = {}
    if not np.isnan(s["ci_cover"]).any():
        ci = s["ci_cover"].astype(bool)
        out |= {"score CI coverage, per model": mc(ci.mean(1)), "score CIs: all cover (simultaneous)": mc(ci.all(1))}
    worst_model = rs.mean(0).min()
    out |= {
        "rank interval coverage, per model": mc(rs.mean(1)),
        "rank interval coverage, worst model": (worst_model, np.sqrt(worst_model * (1 - worst_model) / R)),
        "rank interval coverage, simultaneous (all models)": mc(rs.all(1)),
        "rank interval width, mean": mc(width.mean(1)),
    }
    for rule, label in [("orderings", "declared orderings"), ("rs", "rank intervals don't overlap")]:
        dec, false = od[f"{rule}_declared"], od[f"{rule}_false"]
        out |= {
            f"[{label}] orderings per replicate": mc(dec),
            f"[{label}] false orderings per replicate": mc(false),
            f"[{label}] P(any false ordering)": mc(false > 0),
            f"[{label}] power: share of true orderings declared": mc((dec - false) / n_ordered),
        }
    return out


summ = {p: summarize(store[p], pd.DataFrame(orders[p])) for p in procs}
table = pd.concat({p: pd.DataFrame(v, index=["value", "mc_se"]).T for p, v in summ.items()}, axis=1)
table.columns = [f"{p}_{c}" if c == "mc_se" else p for p, c in table.columns]
table.to_csv(f"results/q2{tag}_summary.csv")
print(f"\nsummary ({R} replicates; Monte Carlo SEs in results/q2{tag}_summary.csv):")
print(table[procs].to_string(float_format=lambda x: f"{x:.4g}"))

per_model = pd.DataFrame({"model": names, "true_rating": theta, "true_rank": r_true,
                          "battles": world.battles_per_model(), "se_mean": est_se.mean(0)})
for p in procs:
    s = store[p]
    per_model[f"{p}_rs_coverage"] = s["rs_cover"].mean(0)
    per_model[f"{p}_mean_best"], per_model[f"{p}_mean_worst"] = s["best"].mean(0), s["worst"].mean(0)
    per_model[f"{p}_mean_width"] = (s["worst"] - s["best"] + 1).mean(0)
per_model = per_model.sort_values("true_rank")
per_model.to_csv(f"results/q2{tag}_per_model.csv", index=False)

# ---------------------------------------------------------------- multiplier sweep
sw = pd.DataFrame({"z": Z_GRID, "alpha_each": 2 * norm.sf(Z_GRID)})
sw["bonferroni_M_equivalent"] = LEVEL / sw.alpha_each   # number of intervals a Bonferroni z this large would cover
for k, col in enumerate(["ci_all_cover", "rs_all_cover", "rs_cover_per_model", "rs_width", "power", "p_any_false"]):
    sw[col] = sweep[:, :, k].mean(0)
sw["rs_all_cover_mc_se"] = sweep[:, :, 1].std(0, ddof=1) / np.sqrt(R)
sw.to_csv(f"results/q2{tag}_sweep.csv", index=False)


def smallest_z(col):
    ok = sw[col] >= 1 - LEVEL
    return sw.z[ok].min() if ok.any() else np.nan


print(f"\nmultiplier sweep: smallest z on the grid with simultaneous coverage >= 95%: score CIs {smallest_z('ci_all_cover'):.2f}, "
      f"rank spreads {smallest_z('rs_all_cover'):.2f} (Arena uses 1.96, Bonferroni {z_bonf:.2f})")
shown = np.isclose(sw.z * 10, np.round(sw.z * 10)) | np.isclose(sw.z, z_bonf) | np.isclose(sw.z, norm.ppf(1 - LEVEL / 2))
print(sw[shown].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# ---------------------------------------------------------------- real data
res = pd.DataFrame({"model": names, "rating": real.ratings.values, "se": np.sqrt(real.variances.values),
                    "battles": world.battles_per_model()})
for p in procs:
    out = PROCEDURES[p](real, real_V)
    res[f"{p}_best"], res[f"{p}_worst"] = ranks_from_orderings(out["above"])
    res[f"{p}_orderings"] = out["above"].sum(1)
if "arena" in procs:
    b, w = rank_spread(real.rating_lower, real.rating_upper)
    assert (res.arena_best.values == b).all() and (res.arena_worst.values == w).all()
res = res.sort_values("rating", ascending=False)
res.to_csv("results/q2_real.csv", index=False)
print("\nreal data: " + "; ".join(f"{p}: mean rank interval width {(res[f'{p}_worst'] - res[f'{p}_best'] + 1).mean():.1f}, "
                                  f"{res[f'{p}_orderings'].sum():,} declared orderings" for p in procs))
print(res.head(20).round(1).to_string(index=False))
