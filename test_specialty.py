"""Regression checks for HC's extreme-tail arithmetic. Run with python test_specialty.py."""
import numpy as np
from scipy.stats import norm
from specialty import global_stats, higher_criticism


# Ordinary p-values agree with the defining formula, including its maximizing index.
p = np.array([0.0001, 0.02, 0.18, 0.2, 0.6, 0.9])
k = np.arange(1, len(p) + 1)
direct = np.sqrt(len(p)) * (k / len(p) - p) / np.sqrt(p * (1 - p))
for plus in (False, True):
    eligible = (k <= len(p) / 2) & ((p > 1 / len(p)) if plus else True)
    expected = np.where(eligible, direct, -np.inf)
    for values, is_log in [(p, False), (np.log(p), True)]:
        value, index = higher_criticism(values, plus=plus, log_p=is_log)
        np.testing.assert_allclose(value, expected.max(), rtol=1e-12)
        assert index == expected.argmax() + 1

# Strengthening a single extreme signal must not erase it when its p-value underflows.
z = np.random.default_rng(0).normal(size=958)
z[0] = 30
before = global_stats(z)
z[0] = 40
after = global_stats(z)
assert 2 * norm.sf(40) == 0
assert np.isfinite(after["hc"]) and after["hc"] > before["hc"] > 1e90
np.testing.assert_allclose(after["hc_plus"], before["hc_plus"], rtol=1e-12)

# The real board's extreme z has a finite HC term even though p itself is unrepresentable.
z = np.zeros(958)
z[0] = 40.254542478676605
np.testing.assert_allclose(global_stats(z)["hc"], 1.979902434806666e175, rtol=1e-12)
assert np.isposinf(higher_criticism([0.0, 0.5, 0.8, 0.9])[0])
print("HC formula, extreme-tail monotonicity, HC+ exclusion and real-board magnitude: passed")
