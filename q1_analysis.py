"""Q1 follow-up (exploratory): which models have stable ranks, and why?

Uses the replicates saved by q1.py (results/q1.npz); no new simulation. For each model:
  stability (outcomes): SD of the estimated point rank over replicates, P(estimated rank == true rank),
                        mean rank spread width;
  candidate drivers:    true rating / rank (frontier vs weak models), battles and SE (amount of data),
                        number of distinct opponents (graph sparsity), crowding (how many models are truly
                        within 2 SE, and the gap to the nearest neighbour in SE units), and era (median date
                        of the model's battles).
Reports Spearman correlations, the same outcomes by rank tier, and a regression of each outcome on all drivers
at once, to separate drivers that are correlated with each other (e.g. old models have fewer battles).

    python q1_analysis.py
Writes results/q1_analysis_per_model.csv.
"""
import numpy as np
import pandas as pd
from arena_data import load_battles, load_official
from simulate import pair_counts

sim = np.load("results/q1.npz", allow_pickle=True)
names, theta, r_true = sim["names"], sim["theta"], sim["r_true"]
est_rank, best, worst, se = sim["est_rank"], sim["best"], sim["worst"], sim["se"]
M = len(names)

# ---------------------------------------------------------------- per-model drivers
_, cutoff, _ = load_official()
battles = load_battles(cutoff=cutoff)
pairs, n = pair_counts(battles, names)
opponents = np.bincount(pairs.ravel(), minlength=M)
n_battles = np.bincount(pairs.ravel(), weights=np.repeat(n, 2), minlength=M)
long = pd.concat([battles[["model_a", "tstamp"]].rename(columns={"model_a": "model"}),
                  battles[["model_b", "tstamp"]].rename(columns={"model_b": "model"})])
era = pd.to_datetime(long.groupby("model").tstamp.median().reindex(names).values, unit="s")

se_mean = se.mean(0)
gap = np.abs(theta[:, None] - theta[None, :])
pair_se = np.sqrt(se_mean[:, None] ** 2 + se_mean[None, :] ** 2)
z_gap = gap / pair_se                                   # true gap in units of the SE of the difference
np.fill_diagonal(z_gap, np.inf)

df = pd.DataFrame({
    "model": names, "true_rating": theta, "true_rank": r_true,
    "battles": n_battles, "se": se_mean, "opponents": opponents,
    "neighbours_within_10pts": (gap < 10).sum(1) - 1,   # crowding in rating points only (no SE involved)
    "neighbours_within_2se": (z_gap < 2).sum(1),        # models whose true gap is < 2 SE of the difference
    "nearest_gap_in_se": z_gap.min(1),
    "era": era,
    # outcomes
    "rank_sd": est_rank.std(0, ddof=1),
    "p_rank_exact": (est_rank == r_true[None, :]).mean(0),
    "spread_width": (worst - best + 1).mean(0),
}).sort_values("true_rank")
df.to_csv("results/q1_analysis_per_model.csv", index=False)

drivers = ["true_rank", "battles", "se", "opponents", "neighbours_within_10pts", "neighbours_within_2se",
           "nearest_gap_in_se", "era"]
outcomes = ["rank_sd", "p_rank_exact", "spread_width"]
num = df.assign(era=df.era.astype("int64"))

print(f"{M} models, {est_rank.shape[0]} replicates\n")
print("Spearman correlation, drivers (rows) x outcomes (columns):")
print(num[drivers + outcomes].corr("spearman").loc[drivers, outcomes].round(2).to_string())

print("\nCorrelation among drivers (Spearman):")
print(num[drivers].corr("spearman").round(2).to_string())

tiers = pd.cut(df.true_rank, [0, 10, 30, 60, 100, M], labels=["1-10", "11-30", "31-60", "61-100", f"101-{M}"])
print("\nBy true-rank tier (means):")
print(df.groupby(tiers, observed=True)[["battles", "se", "opponents", "neighbours_within_10pts", "neighbours_within_2se", "rank_sd",
                                        "p_rank_exact", "spread_width"]].mean().round(2).to_string())

# Regression on standardized drivers: which ones matter once the others are held fixed?
# battles is dropped: log(se) is almost exactly -0.5 log(battles) (Spearman -1.00), so both cannot be in one model.
# Two specifications: crowding in rating points (raw spacing of true strengths, independent of the data) and
# crowding in SE units (what the data can resolve, which mixes spacing with SE).
def ols(X, yv):
    beta, *_ = np.linalg.lstsq(X.values, yv, rcond=None)
    resid = yv - X.values @ beta
    s2 = resid @ resid / (len(yv) - X.shape[1])
    se_b = np.sqrt(np.diag(s2 * np.linalg.inv(X.values.T @ X.values)))
    return beta, se_b, 1 - resid.var() / yv.var()


for crowd in [["neighbours_within_10pts"], ["neighbours_within_2se", "nearest_gap_in_se"]]:
    X = num[["true_rank", "se", "opponents", "era"] + crowd].copy()
    X["se"] = np.log(X.se)
    if "nearest_gap_in_se" in X:
        X["nearest_gap_in_se"] = np.log(X.nearest_gap_in_se)
    X = (X - X.mean()) / X.std()
    X.insert(0, "const", 1.0)
    print(f"\nOLS on standardized drivers, crowding = {crowd}; coefficient (t-stat):")
    rows = {}
    for y in outcomes:
        beta, se_b, r2 = ols(X, num[y].values)
        rows[y] = [f"{b:+.2f} ({b / s:+.1f})" for b, s in zip(beta, se_b)] + [f"{r2:.2f}"]
    print(pd.DataFrame(rows, index=list(X.columns) + ["R^2"]).to_string())
