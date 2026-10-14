# How Certain Is the Chatbot Arena Leaderboard?

STAT 236 (Fall 2026) final project — Harry Hu, Chloe Yu, Ender Yang, Chenxi Ge

Chatbot Arena ranks LLMs with a Bradley–Terry model and reports a **rank spread** for each model, built from per-model 95% confidence intervals without multiplicity correction. We study how reliable these rank spreads are, and whether differences across category leaderboards are real.

## Research questions

- **Q1 (Diagnosis):** What is the actual coverage of Arena's rank spread, i.e., how often does it contain the true rank?
- **Q2 (Repair):** Do multiplicity-corrected procedures (Bonferroni, χ² simultaneous set, pairwise tests with Holm/BH) reach 95% coverage, and how much wider are they?
- **Q3 (Category effects):** Are category "specialties" real? First test whether any exist (Higher Criticism vs. min-P and χ²), then identify which ones (Bonferroni, BH).

## Approach

Semi-synthetic simulation: fit Bradley–Terry on real Arena battles and treat the fit as the truth, keep the real matchups, regenerate outcomes, then compare each procedure's rank intervals against the true ranks over many replications.

## Data and code

- [`lmarena-ai/arena-human-preference-140k`](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k) — battle-level votes with category tags
- [`lmarena-ai/leaderboard-dataset`](https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset) — historical leaderboard snapshots with scores and CIs
- [`lmarena/arena-rank`](https://github.com/lmarena/arena-rank) — Arena's official ranking code

## References

- Chiang et al. (2024). *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference.* ICML. [arXiv:2403.04132](https://arxiv.org/abs/2403.04132)
- LMArena (2025). [Arena's Ranking Method](https://arena.ai/blog/ranking-method)
- Donoho & Jin (2004). Higher criticism for detecting sparse heterogeneous mixtures. *Annals of Statistics.*
