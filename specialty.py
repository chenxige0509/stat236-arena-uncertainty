"""Q3 building blocks: category-specific scores, the specialty z-statistics, and the multiple-testing procedures.

Data layout. Battles are aggregated into groups g = (model pair, category pattern), where the pattern is the 8-bit
set of categories the battle belongs to (categories overlap). Each group has counts of win / tie / loss from the
side of the pair's lower-index model, exactly like simulate.py's cells. Summing a pair's groups gives simulate.py's
pair-level cells, so the overall BT fit and every category score come from the same counts, for real and simulated
data alike.

Specialty statistic. For model m and category c, with p = sigmoid(beta_hat_m - beta_hat_opponent) from the OVERALL fit
and y in {1, 0.5, 0} the battle's score for m:
    O_mc = sum y,   E_mc = sum p,   V_mc = sum (y - p)^2,   z_mc = (O_mc - E_mc) / sqrt(V_mc)
over m's battles in c. z > 0 means m does better in c than its overall strength predicts. Because beta_hat is fit on
all battles, including those in c, z_mc is shrunk towards 0 and correlated across pairs; q3.py calibrates each z_mc
by a parametric bootstrap under "no specialties" (z* = standardized against its null mean and SD).

Tests on the N tested pairs (two-sided p-values p = 2 Phi(-|z*|)):
  global (is there any specialty?)  min-P = N * min p, chi^2 = sum z*^2, Higher Criticism (Donoho & Jin 2004)
  identification (which ones?)      Bonferroni (p < alpha / N), Benjamini-Hochberg at level q
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from categories import CATEGORIES, category_mask


# ---------------------------------------------------------------- data
class Groups:
    """Battles aggregated into (pair, category pattern) groups on a fixed comparison graph.
    pairs: (P, 2) lower/higher model index; gp: group -> pair index; pattern: (G, C) bool; n: battles per group."""

    def __init__(self, battles, competitors, pairs):
        idx = {c: k for k, c in enumerate(competitors)}
        a, b = battles.model_a.map(idx).values, battles.model_b.map(idx).values
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        key = pd.Series(np.arange(len(pairs)), index=pd.MultiIndex.from_arrays([pairs[:, 0], pairs[:, 1]]))
        pair_of_battle = key.loc[list(zip(lo, hi))].values
        bits = np.zeros(len(battles), dtype=np.int64)
        for k, name in enumerate(CATEGORIES):
            bits |= category_mask(battles, name).values.astype(np.int64) << k
        y = battles.winner.map({"model_a": 1.0, "model_b": 0.0}).fillna(0.5).values
        y = np.where(a < b, y, 1 - y)                            # score of the lower-index model
        outcome = np.select([y == 1.0, y == 0.5], [0, 1], 2)     # 0 win, 1 tie, 2 loss (lower-index side)

        g_key = pair_of_battle * (1 << len(CATEGORIES)) + bits
        uniq, g_of_battle = np.unique(g_key, return_inverse=True)
        self.pairs = np.asarray(pairs)
        self.M = int(self.pairs.max()) + 1
        self.gp = (uniq >> len(CATEGORIES)).astype(np.int64)
        gbits = uniq & ((1 << len(CATEGORIES)) - 1)
        self.pattern = np.column_stack([(gbits >> k) & 1 for k in range(len(CATEGORIES))]).astype(bool)
        self.cells = np.zeros((len(uniq), 3))                    # observed win / tie / loss per group
        np.add.at(self.cells, (g_of_battle, outcome), 1)
        self.n = self.cells.sum(1).astype(np.int64)

    def pair_cells(self, cells):
        """Group cells -> pair-level cells in simulate.py's layout [wins; ties; losses] (length 3P)."""
        out = np.zeros((len(self.pairs), 3))
        np.add.at(out, self.gp, cells)
        return out.T.reshape(-1)

    def battles(self):
        """(M, C) battles per model and category."""
        i, j = self.pairs[self.gp, 0], self.pairs[self.gp, 1]
        return np.column_stack([np.bincount(i[m], self.n[m], self.M) + np.bincount(j[m], self.n[m], self.M)
                                for m in self.pattern.T])

    def draw(self, probs, rng):
        """Null outcomes: each group's battles redrawn from its pair's (win, tie, loss) probabilities."""
        return rng.multinomial(self.n, probs[self.gp]).astype(float)


def specialty_z(groups, cells, beta):
    """(M, C) arrays O, E, V and z for the given group cells and overall strengths beta (natural BT scale)."""
    i, j = groups.pairs[groups.gp, 0], groups.pairs[groups.gp, 1]
    p = 1 / (1 + np.exp(-(beta[i] - beta[j])))                  # P(lower-index model wins), per group
    w, t, l = cells.T
    s_i, s_j = w + 0.5 * t, l + 0.5 * t
    n = w + t + l
    v = w * (1 - p) ** 2 + t * (0.5 - p) ** 2 + l * p ** 2       # sum (y - p)^2, the same for both sides
    M, C = groups.M, groups.pattern.shape[1]
    O, E, V = np.zeros((M, C)), np.zeros((M, C)), np.zeros((M, C))
    for c in range(C):
        m = groups.pattern[:, c]
        O[:, c] = np.bincount(i[m], s_i[m], M) + np.bincount(j[m], s_j[m], M)
        E[:, c] = np.bincount(i[m], n[m] * p[m], M) + np.bincount(j[m], n[m] * (1 - p[m]), M)
        V[:, c] = np.bincount(i[m], v[m], M) + np.bincount(j[m], v[m], M)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (O - E) / np.sqrt(V)
    return O, E, V, z


# ---------------------------------------------------------------- tests
def two_sided_p(z):
    return 2 * norm.sf(np.abs(z))


def higher_criticism(p, alpha0=0.5, plus=False):
    """Donoho & Jin (2004): HC = max over k <= alpha0 N of sqrt(N) (k/N - p_(k)) / sqrt(p_(k) (1 - p_(k))).
    plus=True is HC+, which only uses p_(k) > 1/N (Lecture 2). Returns (HC, k at the max)."""
    p = np.sort(np.asarray(p, dtype=float))
    N = len(p)
    k = np.arange(1, N + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        hc = np.sqrt(N) * (k / N - p) / np.sqrt(p * (1 - p))
    ok = k <= alpha0 * N
    if plus:
        ok &= p > 1 / N
    hc = np.where(ok & np.isfinite(hc), hc, -np.inf)
    return float(hc.max()), int(hc.argmax()) + 1


def global_stats(zs):
    """The three global statistics on standardized specialties z* (1-D array over the tested pairs)."""
    p = two_sided_p(zs)
    return {"min_p": float(len(p) * p.min()),       # Bonferroni-adjusted smallest p (small = evidence)
            "max_abs_z": float(np.abs(zs).max()),   # the same test on the z scale (large = evidence)
            "chi2": float((zs ** 2).sum()),
            "hc": higher_criticism(p)[0],
            "hc_plus": higher_criticism(p, plus=True)[0]}


def bonferroni(p, alpha=0.05):
    return np.asarray(p) < alpha / len(p)


def benjamini_hochberg(p, q=0.10):
    """BH step-up: reject the k smallest p-values, k = max{k : p_(k) <= q k / N}."""
    p = np.asarray(p, dtype=float)
    N = len(p)
    order = np.argsort(p)
    ok = p[order] <= q * np.arange(1, N + 1) / N
    k = int(np.nonzero(ok)[0].max()) + 1 if ok.any() else 0
    out = np.zeros(N, bool)
    out[order[:k]] = True
    return out
