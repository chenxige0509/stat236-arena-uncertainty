"""Q1: how often does Arena's rank spread contain the true rank?  (semi-synthetic; Proposal Sec. 3.2)

Truth:   arena-rank fit on the real battles behind the 2024-08-12 leaderboard (same data as Step 0), beta*.
World:   the real comparison graph (which pairs met, how often), outcomes redrawn from BT(beta*) with ties.
Method:  Arena's current pipeline: arena-rank marginal 95% sandwich CIs -> rank spread [best, worst] (blog formula).
Records, per replicate and model: does the score CI cover the true rating, does the rank spread cover the true rank,
the spread's endpoints, and which orderings "i above j" are declared (and wrong) under the site's tie rule.

    python q1.py --reps 1000 [--tie-rate 0] [--seed 0]
Writes results/q1{tag}_summary.csv, results/q1{tag}_per_model.csv, results/q1{tag}.npz.
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
from arena_rank.utils.data_utils import PairDataset
from arena_data import load_battles, load_official
from ranking import fit_arena, rank_spread, point_rank
from simulate import World, pair_counts

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=1000)
ap.add_argument("--tie-rate", type=float, default=None, help="default: observed tie share; 0 = binary outcomes")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
os.makedirs("results", exist_ok=True)
rng = np.random.default_rng(args.seed)

# ---------------------------------------------------------------- the true world
_, cutoff, _ = load_official()
battles = load_battles(cutoff=cutoff)
real_ds = PairDataset.from_pandas(battles[["model_a", "model_b", "winner"]])
real = fit_arena(real_ds)
names = np.array(real_ds.competitors)
tie = battles.winner.str.startswith("tie").mean() if args.tie_rate is None else args.tie_rate
pairs, n = pair_counts(battles, names)
world = World(pairs, n, real.beta.values, tie)
theta, M = world.rating, world.M
r_true = point_rank(theta)
truly_above = theta[:, None] > theta[None, :]                 # [i, j]: i truly ranks above j
density = len(pairs) / (M * (M - 1) / 2)
print(f"world: M={M} models, {n.sum():,} battles/replicate, {density:.1%} of model pairs observed, tie rate {tie:.3f}")

# ---------------------------------------------------------------- replicates
keys = ["ci_cover", "rs_cover", "best", "worst", "est_rank", "se"]
store = {k: np.zeros((args.reps, M)) for k in keys}
orders = []
t0 = time.time()
for r in range(args.reps):
    est = fit_arena(world.draw(rng))
    L, U = est.rating_lower.values, est.rating_upper.values
    best, worst = rank_spread(L, U)
    store["ci_cover"][r] = (L <= theta) & (theta <= U)
    store["rs_cover"][r] = (best <= r_true) & (r_true <= worst)
    store["best"][r], store["worst"][r] = best, worst
    store["est_rank"][r] = point_rank(est.ratings.values)
    store["se"][r] = np.sqrt(est.variances.values)
    row = {}
    # "i above j" when the rank spreads don't overlap (site's tie rule) or the score CIs don't overlap
    for rule, sep in [("rs", worst[:, None] < best[None, :]), ("ci", L[:, None] > U[None, :])]:
        row[f"{rule}_declared"], row[f"{rule}_false"] = sep.sum(), (sep & ~truly_above).sum()
    orders.append(row)
    if (r + 1) % 100 == 0:
        print(f"  {r + 1}/{args.reps} replicates, {time.time() - t0:.0f}s", flush=True)

tag = "" if args.tie_rate is None else f"_tie{args.tie_rate:g}"
real_best, real_worst = rank_spread(real.rating_lower, real.rating_upper)
np.savez_compressed(f"results/q1{tag}.npz", names=names, theta=theta, r_true=r_true,
                    real_best=real_best, real_worst=real_worst, **store)


# ---------------------------------------------------------------- summary: estimate and Monte Carlo SE
def mc(x):
    x = np.asarray(x, dtype=float)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


ci, rs = store["ci_cover"].astype(bool), store["rs_cover"].astype(bool)
width = store["worst"] - store["best"] + 1
too_low = store["best"] > r_true[None, :]            # spread lies entirely below the truth (model looks worse)
od = pd.DataFrame(orders)
n_ordered = truly_above.sum()
summary = {
    "models": (M, np.nan),
    "replicates": (args.reps, np.nan),
    "battles per replicate": (n.sum(), np.nan),
    "share of model pairs observed": (density, np.nan),
    "tie rate": (tie, np.nan),
    "score CI coverage, per model (nominal 0.95)": mc(ci.mean(1)),
    "score CIs: all cover (simultaneous)": mc(ci.all(1)),
    "score CIs missing per replicate": mc((~ci).sum(1)),
    "rank spread coverage, per model": mc(rs.mean(1)),
    "rank spread coverage, worst model": (rs.mean(0).min(), np.sqrt(rs.mean(0).min() * (1 - rs.mean(0).min()) / args.reps)),
    "rank spread coverage, simultaneous (all models)": mc(rs.all(1)),
    "rank spreads missing per replicate": mc((~rs).sum(1)),
    "share of misses with spread below truth": ((~rs & too_low).sum() / max((~rs).sum(), 1), np.nan),
    "rank spread width, mean (simulated)": mc(width.mean(1)),
    "rank spread width, mean (real data fit)": ((real_worst - real_best + 1).mean(), np.nan),
    "point-estimate rank exact, share of models": mc((store["est_rank"] == r_true).mean(1)),
}
for rule, label in [("rs", "rank spreads don't overlap"), ("ci", "score CIs don't overlap")]:
    dec, false = od[f"{rule}_declared"], od[f"{rule}_false"]
    summary |= {
        f"[{label}] orderings declared per replicate": mc(dec),
        f"[{label}] false orderings per replicate": mc(false),
        f"[{label}] P(any false ordering)": mc(false > 0),
        f"[{label}] power: share of true orderings declared": mc((dec - false) / n_ordered),
    }
s = pd.DataFrame(summary, index=["value", "mc_se"]).T
print(s.to_string(float_format=lambda x: f"{x:.4g}"))
s.to_csv(f"results/q1{tag}_summary.csv")

per_model = pd.DataFrame({
    "model": names, "true_rating": theta, "true_rank": r_true, "battles": world.battles_per_model(),
    "se_mean": store["se"].mean(0), "ci_coverage": ci.mean(0), "rs_coverage": rs.mean(0),
    "miss_below": (~rs & too_low).mean(0), "miss_above": (~rs & ~too_low).mean(0),
    "mean_best": store["best"].mean(0), "mean_worst": store["worst"].mean(0), "mean_width": width.mean(0),
    "real_best": real_best, "real_worst": real_worst,
}).sort_values("true_rank")
per_model.to_csv(f"results/q1{tag}_per_model.csv", index=False)
print("\nlowest rank-spread coverage:")
print(per_model.nsmallest(8, "rs_coverage").round(3).to_string(index=False))
