"""Pairwise-difference tests for Q2: test every pair of models directly instead of comparing score intervals.

For each pair i < j the statistic is z_ij = (beta_hat_i - beta_hat_j) / SE(beta_hat_i - beta_hat_j), with
Var(beta_hat_i - beta_hat_j) = V_ii + V_jj - 2 V_ij from the full sandwich covariance V. A procedure declares
"i above j" when it rejects the pair and z_ij > 0. Everything is done on |z| with critical values from the normal
quantile function, so p-values never underflow (thousands of pairs have |z| > 40).

Error that matters. In the simulation every true score is distinct, so no H0: beta_i = beta_j is true and the only
possible mistake is a wrong direction ("i above j" when beta_i < beta_j). If no ordering is wrong, every rank
interval holds (best_m <= true rank <= worst_m for all m), so P(any false ordering) <= alpha implies simultaneous rank
coverage >= 1 - alpha. The procedures differ in how they guarantee that:

  pairwise_bonferroni  two-sided test of each pair at alpha / K, K = M(M-1)/2. P(any false ordering) <= alpha under
                       any dependence, ties allowed (a false ordering needs |z| beyond the bar with the wrong sign).
  holm                 Holm (1979) step-down on the K two-sided p-values; the k-th smallest is compared with
                       alpha / (K - k + 1). This is the textbook version the proposal names. Holm controls the chance
                       of rejecting a true H0 (exactly tied pairs), but not wrong directions for pairs that are close
                       and not tied. That is proven only for independent statistics under conditions normal ones meet
                       (Shaffer 1980); pairwise z_ij that share a model are never independent, and for dependent
                       statistics it is open (Finner 1999; Guo and Romano). q2.py --tiers T --tier-spread f builds
                       worlds with many such near-ties to check it empirically.
  holm_onesided        Holm on the M(M-1) one-sided hypotheses H_ij: beta_i <= beta_j ("i is not above j"). Rejecting
                       H_ij is exactly declaring i above j, so this controls P(any false ordering) <= alpha under any
                       dependence and with ties, by Holm's own theorem.

All guarantees are asymptotic: they treat z_ij as normal with the plug-in sandwich covariance. The k-th largest |z| is compared with the
                       one-sided alpha / (2K - k + 1): it starts at the pairwise Bonferroni threshold and relaxes only
                       half as fast as the two-sided Holm.

Every procedure here rejects exactly the pairs with |z| at or above some cutoff, so its declared orderings are
transitive (`is_transitive`): if i > j and j > k are declared, then z_ik >= min(z_ij, z_jk) because
SE(beta_i - beta_k) <= SE(beta_i - beta_j) + SE(beta_j - beta_k). So the rank intervals come from a proper partial
order, and taking the transitive closure would add nothing.
"""
import numpy as np
from scipy.stats import norm


def pairwise_z(ratings, V):
    """M x M antisymmetric matrix of z_ij = (r_i - r_j) / SE(r_i - r_j); 0 on the diagonal."""
    r, v = np.asarray(ratings, dtype=float), np.diag(V)
    var = np.maximum(v[:, None] + v[None, :] - 2 * V, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (r[:, None] - r[None, :]) / np.sqrt(var)
    np.fill_diagonal(z, 0.0)
    return z


def critical_values(K, level, method):
    """Critical |z| for the k-th largest |z| (k = 1..K), i.e. the step-down sequence written on the z scale.
    A pair's two-sided p-value is <= a  <=>  |z| >= norm.isf(a / 2)."""
    k = np.arange(1, K + 1)
    if method == "pairwise_bonferroni":
        return np.full(K, norm.isf(level / (2 * K)))
    if method == "holm":
        return norm.isf(level / (2 * (K - k + 1)))
    if method == "holm_onesided":
        return norm.isf(level / (2 * K - k + 1))
    raise ValueError(method)


def step_down(abs_z, crit):
    """Number of rejections of a step-down procedure: sort |z| in decreasing order and reject while
    |z|_(k) >= crit_k; stop at the first failure. (Single-step procedures are the constant-crit case.)"""
    s = np.sort(abs_z)[::-1]
    fail = s < crit
    return int(np.argmax(fail)) if fail.any() else len(s)


def pairwise_test(ratings, V, method, level=0.05):
    """Declared orderings of one pairwise procedure.
    Returns (above, z_cut): above[i, j] = "i declared above j", and z_cut = the critical |z| in force where the
    procedure stopped (the bar the largest non-rejected |z| failed; the last critical value if every pair is rejected).
    Holm's critical values only fall as k grows, so tied |z| are never split between rejected and not rejected."""
    z = pairwise_z(ratings, V)
    M = len(z)
    iu = np.triu_indices(M, 1)
    abs_z = np.abs(z[iu])
    assert np.isfinite(abs_z).all(), "zero variance for some pair difference"
    crit = critical_values(len(abs_z), level, method)
    n_rej = step_down(abs_z, crit)
    z_cut = crit[min(n_rej, len(crit) - 1)]
    reject = np.zeros(len(abs_z), bool)
    reject[np.argsort(-abs_z, kind="stable")[:n_rej]] = True
    R = np.zeros((M, M), bool)
    R[iu] = reject
    R |= R.T
    return R & (z > 0), z_cut


def is_transitive(above):
    """True if i > j and j > k declared always implies i > k declared."""
    a = above.astype(np.int32)
    return not ((a @ a > 0) & ~above).any()
