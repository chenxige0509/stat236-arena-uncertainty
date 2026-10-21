# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

STAT 236 final project: how reliable are Chatbot Arena's rank spreads, which are built from per-model 95% Bradley–Terry intervals with no multiplicity correction? There are three research questions (see README and `docs/Proposal-Draft.pdf`). Step 0 and Q1 are done. Q2 (multiplicity-corrected procedures) is in progress: `q2.py` has Arena's method, Bonferroni, the χ² simultaneous set (projected per model, `chi2`, and read through Scheffé pairwise contrasts, `chi2_scheffe`) and a multiplier sweep; the pairwise Holm/BH procedures are stubs. The χ² procedures have not been run yet. Q3 (category effects) is not implemented yet. The data differ from the proposal: the proposal names `arena-human-preference-140k` and `leaderboard-dataset`, but the code uses the 2024-08-14 public dump and the 2024-08-13 official snapshot, because that pair lets Step 0 check the pipeline against an answer key computed from the same battles. The method is a semi-synthetic simulation: fit BT on the real battles, treat that fit as the truth, keep the real matchups, redraw the outcomes, rerun Arena's method, and check its intervals against the true ranks.

## Commands

All scripts use paths relative to the repo root (`data/`, `results/`, `docs/figures/`), so run them from there.

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python prepare_data.py --src ~/Downloads/clean_battle_20240814_public.json   # one-time: 2.1 GB JSON -> data/battles_20240814.parquet
python step0.py              # reproduce the 2024-08-12 official leaderboard -> results/step0_comparison.csv
python test_simulate.py      # simulator sanity checks (plain asserts, no test framework)
python q1.py --reps 1000     # ~3 min; --tie-rate 0 for binary outcomes, --seed N
python q1_analysis.py        # reads results/q1.npz
python q1_figures.py         # reads results/q1.npz + q1_analysis_per_model.csv -> docs/figures/q1_*.png
python q2.py --reps 1000     # ~5 min; --procedures arena,bonferroni[,chi2,chi2_scheffe,holm,bh]
jupyter nbconvert --to notebook --execute --inplace notebook/q2_bonferroni.ipynb   # Bonferroni report; reads results/q2_* only
```

Jupyter is not in `requirements.txt`. Install it separately with `pip install jupyter`.

The pinned numpy and pandas need Python ≥ 3.11 (`/opt/homebrew/bin/python3.12` on this machine). The system `python3` is 3.9.

`data/` is gitignored. Every script except `prepare_data.py` needs the parquet file, and the first call to `load_official()` downloads the official pickle into `data/official/`. `results/` and `docs/figures/` are committed, so rerunning a script changes tracked files.

## Architecture

Three shared modules and one top-level script per analysis step:

- `arena_data.py`: `load_battles()` applies the official 2024 filter: anonymous battles, `dedup_sampled`, and `tstamp <= cutoff`. Step 0 shows the dedup filter is needed for the battle counts to match. `load_official()` returns `(table, last_updated_tstamp, bootstrap_df)` for the 2024-08-13 snapshot. Each analysis uses that snapshot's timestamp as the battle cutoff (`_, cutoff, _ = load_official()`). The pickle is loaded with `_SafeUnpickler`, an allow-list that permits only pandas and numpy objects. Do not replace it with `pickle.load`.
- `ranking.py`: this module defines "Arena's method". `fit_arena()` wraps arena-rank's `BradleyTerry` (MLE with sandwich SEs). With `full_cov=True` it also returns the full covariance matrix, which arena-rank computes but does not expose. `rank_spread(L, U)` implements the 2025 blog formula. `best` is also the 2024 leaderboard's rank, i.e. Chiang et al.'s R_m. `point_rank()` gives true or estimated point ranks. Ratings are on the scale `ALPHA * beta + 1000` with `sum(beta) = 0`. The 2024 board pinned `mixtral-8x7b-instruct-v0.1` at 1114 instead, and Step 0 shifts our fit to that anchor before comparing.
- `simulate.py`: `World` holds the true betas on the real comparison graph (`pair_counts`). `draw()` samples multinomial win/tie/loss counts for each pair instead of simulating individual battles. With tie rate t, P(win) = p − t/2, which keeps E[y] = p, so beta* is exactly arena-rank's target. Datasets are built by `fixed_shape_dataset` with a fixed shape (every pair × {win, tie, loss}), so JAX compiles the fit only once. Empty cells get the count `TINY` to avoid division by zero in the sandwich. Keep this fixed-shape format: if the shapes change, every replicate recompiles. `test_simulate.py` checks that this format reproduces `PairDataset.from_pandas` to about 1e-9.

The pipeline: `step0.py` validates the method against the official answer key. `q1.py` runs the replicates and saves the raw per-replicate arrays (`results/q1{tag}.npz`), a summary with Monte Carlo SEs, and per-model CSVs. `q1_analysis.py` and `q1_figures.py` only post-process the saved npz and run no new simulation. A `--tie-rate` run writes files with a `_tie…` suffix, which the downstream scripts do not read.

`q2.py` uses the same world, seed and draw sequence as `q1.py`. It checks that its `arena` rank spreads are identical to the ones saved in `results/q1.npz`. Keep that check passing: draw from `rng` only through `world.draw` and in the same order. Each Q2 procedure is a function `(est, V) -> {"above": M×M bool, optional "L", "U"}`. `above[i, j]` means "i is declared above j". Rank intervals come from `ranks_from_orderings`, which for interval methods reproduces `rank_spread` exactly. To add a procedure, register it in `PROCEDURES`, and also add it to `NEEDS_COV` if it needs the full covariance.

## Things that look like bugs but aren't

- The fit has 129 models but the official board lists 128. The public dump contains one of the board's hidden pre-release models.
- Recent models have 1–2k fewer battles than the official `num_battles` because the public dump omits some recent battles. Published `final_ranking` is mostly 5 higher than R_m recomputed from the published intervals. The data are consistent with the board having been ranked while 5 hidden models were still listed.
- In `rank_spread`, `worst` has no explicit `+1` because the count includes the model itself (U_m > L_m).

## Conventions

- Each script's module docstring states its inputs, outputs, and what the results establish. `step0.py` also has `# Result:` / `# Shows:` comments with the actual numbers. If a rerun changes the numbers, update those comments, the README tables, and `docs/q1.md` to match.
- Simulation summaries report a Monte Carlo SE next to each estimate (`mc()` in `q1.py`).
- Figures use the palette and rcParams defined at the top of `q1_figures.py`. Reuse them for new figures so all figures look consistent.
- Write-ups (README, `docs/q1.md`) use plain, direct prose and state their limitations, such as the idealized simulation and that only one snapshot is checked.
