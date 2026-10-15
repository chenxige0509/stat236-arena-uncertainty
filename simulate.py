"""Semi-synthetic Arena worlds (Proposal Sec. 3.2): keep the real matchups, redraw the outcomes from BT(beta*).

Each battle outcome is coded as arena-rank codes it: y = 1 (i wins), 0.5 (tie / both bad), 0 (j wins).
With p = sigmoid(beta_i - beta_j) and tie probability t, we draw
    P(y=1) = p - t/2,   P(y=0.5) = t,   P(y=0) = 1 - p - t/2,
so E[y] = p exactly: beta* is precisely the target of arena-rank's BT likelihood, ties included. t is the observed
tie share, capped at 2 min(p, 1-p) so the probabilities stay valid. tie_rate=0 gives plain binary BT outcomes.

Datasets are built directly in arena-rank's aggregated PairDataset format with a fixed shape (every observed pair x
{win, tie, loss}), so JAX compiles the fit once. Empty cells get a negligible count (TINY), which leaves the fit and
the sandwich unchanged but avoids dividing by zero in arena-rank's per-cell gradient variance.
"""
import numpy as np
import pandas as pd
import jax.numpy as jnp
from arena_rank.utils.data_utils import PairDataset
from ranking import ALPHA

TINY = 1e-12


def pair_counts(battles, competitors):
    """Observed comparison graph: unique pairs (i < j, as indices into `competitors`) and their battle counts."""
    idx = {c: k for k, c in enumerate(competitors)}
    a, b = battles.model_a.map(idx).values, battles.model_b.map(idx).values
    pr = pd.DataFrame({"i": np.minimum(a, b), "j": np.maximum(a, b)}).value_counts().sort_index().reset_index()
    return pr[["i", "j"]].values.astype(np.int32), pr["count"].values


def fixed_shape_dataset(pairs, cell_counts, weights=None):
    """PairDataset with rows [pairs x win; pairs x tie; pairs x loss] and counts `cell_counts` (length 3P)."""
    M = int(pairs.max()) + 1
    P = len(pairs)
    c = np.maximum(np.asarray(cell_counts, dtype=float), TINY)
    w = np.ones(3 * P) if weights is None else np.asarray(weights, dtype=float)
    return PairDataset(competitors=list(range(M)), pairs=jnp.array(np.tile(pairs, (3, 1))),
                       outcomes=jnp.array(np.repeat([1.0, 0.5, 0.0], P)),
                       counts=jnp.array(c), weights=jnp.array(w), opt_weights=jnp.array(w * c))


def observed_cell_counts(battles, competitors, pairs):
    """Real battles aggregated into the fixed-shape cells (used to check that the format reproduces from_pandas)."""
    idx = {c: k for k, c in enumerate(competitors)}
    a, b = battles.model_a.map(idx).values, battles.model_b.map(idx).values
    y = battles.winner.map({"model_a": 1.0, "model_b": 0.0}).fillna(0.5).values
    y = np.where(a < b, y, 1 - y)                                  # orient every battle from the lower index's side
    key = pd.Series(np.arange(len(pairs)), index=pd.MultiIndex.from_arrays([pairs[:, 0], pairs[:, 1]]))
    k = key.loc[list(zip(np.minimum(a, b), np.maximum(a, b)))].values
    cell = np.select([y == 1.0, y == 0.5], [k, len(pairs) + k], 2 * len(pairs) + k)
    return np.bincount(cell, minlength=3 * len(pairs)).astype(float)


class World:
    """True strengths beta* on a fixed comparison graph (pairs, n). Only the outcomes are random."""

    def __init__(self, pairs, n, beta, tie_rate):
        self.pairs = np.asarray(pairs, dtype=np.int32)
        self.n = np.asarray(n, dtype=np.int64)
        self.beta = np.asarray(beta, dtype=float) - np.mean(beta)   # arena-rank's fit has sum(beta) = 0
        self.M = len(self.beta)
        p = 1 / (1 + np.exp(-(self.beta[self.pairs[:, 0]] - self.beta[self.pairs[:, 1]])))
        t = np.minimum(tie_rate, 2 * np.minimum(p, 1 - p))
        self.probs = np.clip(np.column_stack([p - t / 2, t, 1 - p - t / 2]), 0, 1)
        self.probs /= self.probs.sum(1, keepdims=True)

    @property
    def rating(self):
        """True ratings on the leaderboard scale."""
        return ALPHA * self.beta + 1000

    def battles_per_model(self):
        return np.bincount(self.pairs.ravel(), weights=np.repeat(self.n, 2), minlength=self.M)

    def draw(self, rng):
        """One simulated Arena dataset, as an arena-rank PairDataset."""
        cells = rng.multinomial(self.n, self.probs)      # (P, 3): wins, ties, losses for the lower-index model
        return fixed_shape_dataset(self.pairs, cells.T.reshape(-1))
