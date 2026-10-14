"""Arena's estimator and rank rules, shared by Step 0 / Q1 / Q2.

The estimator is always the official arena-rank BradleyTerry (MLE, sandwich SEs). Ratings are on the Elo-like
scale rating = ALPHA * beta + 1000, with sum(beta) = 0.
"""
import math
import numpy as np
import pandas as pd
import jax.numpy as jnp
from arena_rank.models.bradley_terry import BradleyTerry
from arena_rank.utils.data_utils import PairDataset

ALPHA = 400 / math.log(10)


def fit_arena(ds, significance_level=0.05, full_cov=False):
    """arena-rank's leaderboard computation on a PairDataset.
    Returns a DataFrame (competitors, ratings, rating_lower, rating_upper, variances, beta) and, if requested,
    the full sandwich covariance on the rating scale (arena-rank computes it but only returns the diagonal)."""
    bt = BradleyTerry(n_competitors=ds.n_competitors)
    res = bt.compute_ratings_and_cis(ds, significance_level=significance_level)
    out = pd.DataFrame({k: np.asarray(v) for k, v in res.items()})
    out["beta"] = np.asarray(bt.params["ratings"])
    if not full_cov:
        return out
    # same regularization rule as compute_ratings_and_cis
    total = float(jnp.sum(ds.counts))
    reg = bt.hessian_reg * (total if bool(jnp.allclose(ds.weights, 1.0)) else 1.0)
    H, G = bt.compute_hessian_and_covariance(bt.params["ratings"], ds.pairs, ds.outcomes, ds.counts,
                                             ds.opt_weights, reg, bt.n_competitors)
    Hi = np.linalg.inv(np.asarray(H))
    return out, ALPHA ** 2 * (Hi @ np.asarray(G) @ Hi)


def rank_spread(L, U):
    """Arena blog (2025): best = 1 + #{m': L_m' > U_m}, worst = 1 + #{m' != m: U_m' > L_m}.
    `best` is also the 2024 leaderboard's rank and the R_m of Chiang et al. (2024)."""
    L, U = np.asarray(L, dtype=float), np.asarray(U, dtype=float)
    best = 1 + (L[None, :] > U[:, None]).sum(1)
    worst = (U[None, :] > L[:, None]).sum(1)   # counts m itself (U_m > L_m), which supplies the "+1"
    return best, worst


def point_rank(x):
    """1 + #{models with a strictly higher value}: the true rank when x = true ratings."""
    x = np.asarray(x, dtype=float)
    return 1 + (x[None, :] > x[:, None]).sum(1)
