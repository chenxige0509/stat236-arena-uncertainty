"""Checks for pairwise.py (the Q2 pairwise-difference procedures). Plain asserts, as in test_simulate.py.
  - a hand example whose answer is worked out below;
  - each procedure agrees with statsmodels' Holm / Bonferroni run on ordinary p-values, on random correlated data;
  - the procedures are nested (pairwise Bonferroni <= one-sided Holm <= two-sided Holm) and their declared
    orderings are antisymmetric and transitive;
  - on the real fit, pairwise_z reproduces the pair statistic that q2.py's chi2_scheffe uses.
    python test_pairwise.py            (the last check needs data/; skip it with --no-data)
"""
import sys
import numpy as np
from scipy.stats import norm, chi2
from statsmodels.stats.multitest import multipletests
from pairwise import pairwise_z, critical_values, step_down, pairwise_test, is_transitive

LEVEL = 0.05
METHODS = ["pairwise_bonferroni", "holm_onesided", "holm"]

# ---------------------------------------------------------------- hand example
# Three independent models with SE 1 (so SE of a difference is sqrt 2) and ratings 0, 4, 10:
# |z| = 10/sqrt2 = 7.07 (C vs A), 6/sqrt2 = 4.24 (C vs B), 4/sqrt2 = 2.83 (B vs A); K = 3.
# Bonferroni bar: z_{1 - 0.05/6} = 2.39 -> all three rejected. Holm: same first bar, then 2.24, then 1.96 -> all three.
# With ratings 0, 3, 10 the B-A gap is 3/sqrt2 = 2.12: Bonferroni (2.39) and one-sided Holm (last bar
# z_{1 - 0.05/4} = 2.24) miss it, two-sided Holm (last bar 1.96) declares it.
V = np.eye(3)
for r, expect in [([0, 4, 10], {"pairwise_bonferroni": 3, "holm_onesided": 3, "holm": 3}),
                  ([0, 3, 10], {"pairwise_bonferroni": 2, "holm_onesided": 2, "holm": 3})]:
    for m in METHODS:
        above, _ = pairwise_test(r, V, m, LEVEL)
        assert above.sum() == expect[m], (r, m, above.sum())
        assert above[2, 0] and above[2, 1] and not above[0].any()   # C above A and B, A above nobody
assert np.isclose(critical_values(3, LEVEL, "holm")[-1], norm.isf(0.025))
assert np.isclose(critical_values(3, LEVEL, "holm_onesided")[-1], norm.isf(0.05 / 4))

# ---------------------------------------------------------------- against statsmodels, on random correlated data
rng = np.random.default_rng(0)
for trial in range(200):
    M = rng.integers(3, 40)
    A = rng.normal(size=(M, M + 5)) * rng.uniform(0.2, 3, size=(M, 1))
    V = A @ A.T / (M + 5) + np.diag(rng.uniform(0.1, 2, M))
    r = rng.normal(scale=rng.uniform(0.5, 6), size=M)
    z = pairwise_z(r, V)
    iu = np.triu_indices(M, 1)
    K = len(iu[0])
    p2 = 2 * norm.sf(np.abs(z[iu]))
    ours = {m: pairwise_test(r, V, m, LEVEL)[0] for m in METHODS}

    # two-sided procedures: same rejected pairs as statsmodels, direction = sign of z
    for m, sm in [("pairwise_bonferroni", "bonferroni"), ("holm", "holm")]:
        rej = multipletests(p2, alpha=LEVEL, method=sm)[0]
        assert ((ours[m] | ours[m].T)[iu] == rej).all(), (trial, m)
    # one-sided Holm: statsmodels on all M(M-1) one-sided p-values, H_ij: r_i <= r_j rejected <=> i declared above j
    off = ~np.eye(M, dtype=bool)
    rej1 = np.zeros((M, M), bool)
    rej1[off] = multipletests(norm.sf(z[off]), alpha=LEVEL, method="holm")[0]
    assert (ours["holm_onesided"] == rej1).all(), trial

    for m in METHODS:
        a = ours[m]
        assert not (a & a.T).any() and not a.diagonal().any() and is_transitive(a), (trial, m)
        assert (a <= (z > 0)).all()
    assert (ours["pairwise_bonferroni"] <= ours["holm_onesided"]).all()
    assert (ours["holm_onesided"] <= ours["holm"]).all()
print("hand example and 200 random cases: agree with statsmodels, nested, antisymmetric, transitive")

# step_down edge cases: nothing / everything rejected
assert step_down(np.array([0.1, 0.2]), np.array([3.0, 2.0])) == 0
assert step_down(np.array([9.0, 8.0]), np.array([3.0, 2.0])) == 2
assert step_down(np.array([9.0, 1.0, 8.0]), np.array([3.0, 2.5, 2.0])) == 2

# ---------------------------------------------------------------- real fit (needs data/)
if "--no-data" not in sys.argv:
    from arena_rank.utils.data_utils import PairDataset
    from arena_data import load_battles, load_official
    from ranking import fit_arena

    _, cutoff, _ = load_official()
    b = load_battles(cutoff=cutoff)
    est, Vr = fit_arena(PairDataset.from_pandas(b[["model_a", "model_b", "winner"]]), full_cov=True)
    r = est.ratings.values
    z = pairwise_z(r, Vr)
    # chi2_scheffe in q2.py: i above j when r_i - r_j > sqrt(c) * SE(diff)
    c = chi2.ppf(1 - LEVEL, len(r) - 1)
    var = np.maximum(np.diag(Vr)[:, None] + np.diag(Vr)[None, :] - 2 * Vr, 0)
    assert ((z > np.sqrt(c)) == (r[:, None] - r[None, :] > np.sqrt(c * var))).all()
    # the diagonal of V is arena-rank's own variances
    assert np.allclose(np.diag(Vr), est.variances.values, rtol=1e-8)
    for m in METHODS:
        a, cut = pairwise_test(r, Vr, m, LEVEL)
        assert is_transitive(a)
        print(f"real fit, {m}: {a.sum():,} of {len(r) * (len(r) - 1) // 2:,} pairs ordered, stopping |z| bar {cut:.2f}")
print("all checks passed")
