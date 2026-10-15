"""Sanity checks: run before Q1.
  - rank_spread on a hand example;
  - the fixed-shape dataset used in simulations reproduces arena-rank's own from_pandas fit on the real battles;
  - simulated datasets are unbiased for beta* on average (the truth is the estimator's target).
    python test_simulate.py
"""
import numpy as np
import pandas as pd
from arena_rank.utils.data_utils import PairDataset
from arena_data import load_battles, load_official
from ranking import fit_arena, rank_spread
from simulate import World, pair_counts, fixed_shape_dataset, observed_cell_counts

# A = [10, 12], B = [11, 13], C = [0, 1]: A and B overlap (ranks 1-2), C is 3rd
best, worst = rank_spread([10, 11, 0], [12, 13, 1])
assert list(best) == [1, 1, 3] and list(worst) == [2, 2, 3], (best, worst)

_, cutoff, _ = load_official()
b = load_battles(cutoff=cutoff)
ref_ds = PairDataset.from_pandas(b[["model_a", "model_b", "winner"]])
ref = fit_arena(ref_ds)

pairs, n = pair_counts(b, ref_ds.competitors)
mine = fit_arena(fixed_shape_dataset(pairs, observed_cell_counts(b, ref_ds.competitors, pairs)))
d_r, d_v = np.abs(mine.ratings - ref.ratings).max(), np.abs(mine.variances / ref.variances - 1).max()
print(f"fixed-shape vs from_pandas on real data: max |rating diff| {d_r:.1e}, max relative variance diff {d_v:.1e}")
assert d_r < 1e-6 and d_v < 1e-6

world = World(pairs, n, ref.beta.values, tie_rate=b.winner.str.startswith("tie").mean())
rng = np.random.default_rng(1)
est = np.array([fit_arena(world.draw(rng)).ratings.values for _ in range(20)])
bias_z = (est.mean(0) - world.rating) / (est.std(0, ddof=1) / np.sqrt(len(est)))
print(f"simulated fits (20): mean |bias| / MC SE = {np.abs(bias_z).mean():.2f} (about 0.8 if unbiased)")
assert np.abs(bias_z).mean() < 1.5
print("all checks passed")
