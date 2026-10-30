# How Certain Is the Chatbot Arena Leaderboard?

STAT 236 (Fall 2026) final project — Harry Hu, Chloe Yu, Ender Yang, Chenxi Ge

Chatbot Arena ranks LLMs with a Bradley–Terry model and reports a **rank spread** for each model, built from per-model 95% confidence intervals without multiplicity correction. We study how reliable these rank spreads are, and whether differences across category leaderboards are real.

## Research questions

- **Q1 (Diagnosis):** What is the actual coverage of Arena's rank spread, i.e., how often does it contain the true rank?
- **Q2 (Repair):** Do multiplicity-corrected procedures (Bonferroni, χ² simultaneous set, pairwise tests with Holm) reach 95% coverage, and how much wider are they?
- **Q3 (Category effects):** Are category "specialties" real? First test whether any exist (Higher Criticism vs. min-P and χ²), then identify which ones (Bonferroni, BH).

## Approach

Semi-synthetic simulation: fit Bradley–Terry on real Arena battles and treat the fit as the truth, keep the real matchups, regenerate outcomes, then compare each procedure's rank intervals against the true ranks over many replications.

## Step 0: reproducing Arena's published leaderboard

Before simulating Arena's method we check that it *is* Arena's method. The two inputs play different roles:

- **Input:** `clean_battle_20240814_public.json` contains only battle records (who played whom, who won, when, prompt tags). It has no ratings, intervals or ranks. We compute those ourselves by fitting arena-rank's Bradley–Terry model.
- **Answer key:** `elo_results_20240813.pkl` is what the official pipeline (LMSYS's FastChat code) produced: ratings, bootstrap intervals, ranks and the 100 bootstrap fits for each leaderboard (we use text / overall: 128 models, last updated 2024-08-12 09:20 PDT).

The two come from the same source but are not the same data. The official board was computed from LMSYS's internal battle log, while the JSON is the public release of that log, which is missing some battles. For example, `llama-3.1-70b-instruct` has 14,277 anonymous battles in the JSON even before deduplication or the time cutoff, but 14,622 on the board, and only one of the board's hidden pre-release models appears in the JSON. The missing battles are concentrated in the last few weeks. Older models match exactly, and the remaining rating gaps are largest for exactly the models missing battles (see (1) and (2) below).

`step0.py` refits the JSON and runs four checks against the answer key; the full model-by-model comparison is in `results/step0_comparison.csv`.

| Check | How | Result |
|---|---|---|
| (1) Same battles | Filter as the official pipeline does (anonymous battles, `dedup_tag.sampled`, up to the snapshot time) and compare battle counts per model | 1,657,517 battles. Older models match exactly or within ±10 battles (91 of 128); models released in the last few weeks have 1–2k fewer battles because the public dump omits some recent battles |
| (2) Same ratings | Fit arena-rank's `BradleyTerry` (ties = 0.5), then shift so `mixtral-8x7b-instruct-v0.1` = 1114, the 2024 anchor | Median \|diff\| 0.06 points; max 0.3 among models whose battle counts match |
| (3) Intervals are marginal | (a) Recompute each model's 2.5%/97.5% quantile from the 100 bootstrap fits stored in the pickle; (b) half-width ÷ bootstrap SD, against the multiplier each correction would need | (a) Identical to the published intervals (difference 0). (b) Median 1.90; a normal distribution sampled 100 times gives 1.88, versus 3.55 for Bonferroni and 12.42 for the χ² simultaneous set (M = 128). So the published intervals are per-model 95% intervals with no multiplicity correction |
| (4) Rank formula | Recompute R_m = 1 + #{m′ : L_m′ > U_m} (Chiang et al., 2024; the rank spread's best end) from the published intervals | Matches up to a constant: published ranks are mostly 5 higher. The gap is ≥ 0 and grows as U_m falls, consistent with ranks computed while 5 unreleased models were still on the board |

Also reported: our sandwich SE ÷ the official bootstrap SD has median 0.98, so the current arena-rank intervals (sandwich, used in Q1) and the 2024 intervals (bootstrap) have about the same width.

**Limitations.** The dedup filter is justified by the matching counts, not by a cited line of code. The 2024 site showed only the best end of the rank spread, so the worst end cannot be checked here; the 2026 check below covers both ends.

### The current (2026) site

`step0_2026.py` repeats the interval and rank checks for today's site, using the published boards in [`lmarena-ai/leaderboard-dataset`](https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset) (rating, interval, variance and vote count per model) and a record of what the site displayed for the top 59 models of the Text Overall board dated 2026-10-08 (`results/site_text_overall_20261008.csv`, read from arena.ai on 2026-10-12).

| Check | Result |
|---|---|
| Which board the site shows | The style-controlled one: all 59 scores match `text_style_control` to rounding (max 0.50), all ± and vote counts match. The plain `text` board differs by up to 54 points |
| Intervals are marginal | Half-width ÷ √variance is exactly 1.9600 on every 2026 Text board: 4,773 boards, 1.5M model rows, no exception. A correction would need 2.24–3.85 (Bonferroni) or 2.0–21.6 (χ²), depending on the board's size |
| Rank spread formula | The blog's formula on the published intervals reproduces the site's rank spread for 59 of 59 models, both ends |

So the current site, like the 2024 one, computes its rank spread from uncorrected per-model 95% intervals. The rank-spread code is not public, so this is the evidence for it. **Limitations.** Only the Text boards are checked, and only one site view (the top 59 of one board) was recorded; the site shows only its latest board.

## Q1: coverage of Arena's rank spread

Semi-synthetic simulation: the arena-rank fit on the Step 0 battles is the truth, the real matchups (3,457 pairs, 1.66M battles) are kept, outcomes are redrawn 1,000 times, and Arena's method is rerun each time. Full write-up with figures: [`docs/q1.md`](docs/q1.md).

| | Per model | All 129 models at once |
|---|---|---|
| Score intervals | 0.948 | 0.003 |
| Rank spreads | 0.999 | 0.909 ± 0.009 |

Each score interval is valid on its own, but they almost never all hold at once. The overlap rule absorbs most of those misses, so the rank spread is too conservative for a single model (99.9%) and slightly short for the whole table (91%). Rank stability depends mainly on crowding (how many models have nearly the same strength) and data per model, not on strength itself. The simulation is idealized (independent battles, exact BT, fixed strengths), so real coverage is probably lower.

## Q2: multiplicity-corrected procedures

The same 1,000 simulated datasets as Q1, with each correction applied side by side. Write-ups: [`notebook/q2_bonferroni.ipynb`](notebook/q2_bonferroni.ipynb), [`notebook/q2_chi2.ipynb`](notebook/q2_chi2.ipynb) and [`notebook/q2_holm.ipynb`](notebook/q2_holm.ipynb).

| | Arena | Bonferroni | χ² Scheffé | χ² projection | Pairwise Bonferroni | Holm |
|---|---|---|---|---|---|---|
| Bar for "A above B" | gap > 1.96 (SE_A + SE_B) | gap > 3.55 (SE_A + SE_B) | \|z\| > 12.47 | gap > 12.47 (SE_A + SE_B) | \|z\| > 4.52 | \|z\| > 4.52, falling to 3.96 |
| All rank spreads right at once | 0.909 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Mean rank spread width (ranks) | 8.0 | 13.6 | 33.1 | 43.4 | 12.7 | 11.3 |
| True orderings declared | 0.945 | 0.901 | 0.749 | 0.669 | 0.908 | 0.920 |
| Top model's rank spread, real 2024 board | 1–1 | 1–2 | 1–4 | 1–10 | 1–1 | 1–1 |

Here z is a pair's score gap divided by the standard error of that gap, which uses the full covariance.

All the corrections reach the 95% target, but they overshoot. In the multiplier sweep, z ≈ 2.1 already makes all rank spreads right 95% of the time with 6% more width, because Arena's overlap rule supplies most of the protection. Bonferroni widens the rank spreads by 71%. The χ² simultaneous set of Chiang et al. (2024) protects the whole 129-dimensional score vector, which is far more than a ranking needs. Projected onto each model, it gives rank spreads 3.2 times as wide as Bonferroni's. Reading the same set through pairwise differences (Scheffé), which uses the covariances, reduces this to 2.4 times. The z ≈ 2.1 benchmark uses the true ranks, so it is not a procedure that can be applied to real data. Same idealized simulation as Q1.

Testing each of the 8,256 pairs directly, with Holm's step-down correction, gives the narrowest valid rank spreads so far: 18% narrower than Bonferroni's. Testing each gap directly instead of checking whether intervals overlap saves 0.9 ranks. Holm's step-down saves 1.5 more, because thousands of clearly different pairs lower the bar for the rest. The textbook (two-sided) Holm controls the chance of ordering an exactly tied pair. But it is not proven to control wrong directions for pairs that are close and not tied, when the tests are correlated. A one-sided version that is proven narrows the spreads only to 12.4.

Nine stress-test worlds with the same battles and different true strengths put this to the test:

- **Exact ties:** in tiers of exactly tied models, where the error bounds are tight, Holm gets all rank spreads right in at least 96.6% of datasets. Arena falls to 7–24%, and to 0.2% when all models are tied.
- **Near ties:** in tiers of nearly tied models, which is the unproven case, Holm gets all rank spreads right in at least 99.4%.

On the real battles, Holm does not support 223 of the 7,808 orderings that Arena's method declares. It separates no pair that Arena shows as tied, whether a tie means overlapping score intervals or overlapping rank spreads. All guarantees are asymptotic (normal approximation).

## Q3: are category specialties real?

The 2024 snapshot also holds the official category boards. Q3 asks whether a model's rating in a category (say 3rd in Coding but 7th overall) differs from its overall rating by more than chance. With 129 models and 8 categories there are about 1,000 such comparisons, so about 50 would look significant at p < 0.05 even if no model had any specialty.

**Categories (Step 0).** `categories.py` defines coding, math, instruction following, hard prompts, Chinese, Russian, multi-turn and long query from the dump's prompt tags. `q3_step0.py` checks each definition against the official board of the same name: 97–100 of 128 models are within ±10 battles, as for the overall board, and the category-minus-overall rating has correlation ≥ 0.999 with the official one for the pairs we test. Below 200 battles, a few old models disagree by up to 408 points. The official values look like a fit that did not converge (three Russian models share the identical rating 1137.95, although they scored only 19–31% of their battles). Q3 only tests pairs with at least 200 battles in the category: 958 of 1,032.

**Statistic (Step 2).** For model m in category c, z compares m's score in c (win 1, tie 0.5) with what its overall Bradley–Terry strength predicts, divided by the standard deviation. The overall strength is fit on all battles, including those in c, so z is shrunk towards 0 and correlated across pairs. `q3.py` therefore simulates the null "no specialties" 500 times (real battles, outcomes redrawn from the overall fit, refit each time), standardizes each z against its null mean and SD, and takes the p-values of the global statistics from their null distributions. Under the null the raw z have SD 0.70–1.05, not 1.

| Step 1: is there any specialty? | Observed | Null 95% point | p |
|---|---|---|---|
| HC+ | 330 | 3.1 | < 0.002 |
| χ² | 21,210 | 1,028 | < 0.002 |
| min-P (max \|z\|) | 40 | 4.0 | < 0.002 |

| Step 2: which ones? | Specialties found (of 958) |
|---|---|
| Bonferroni (5%) | 275 |
| BH (10%) | 525 |
| Naive p < 0.05 | 521 (about 48 expected by chance) |

Under this null the 2024 specialties are neither rare nor weak. 94 of the 129 models have at least one Bonferroni specialty, and every category has some (17–40% of its pairs), with median size 25 points (category rating minus overall rating, centered within the category). The largest are language effects: `llama-3-70b-instruct` is 73 points weaker in Chinese (z = −40), and Qwen models are about 60 points stronger. Plain HC is about 10^62: the smallest p-value underflows to 0 and is dropped, and HC is then driven by the second smallest (2.8 × 10^−127), since it grows like 1/√p. HC+, which only uses p > 1/N, is the usable version.

The count of specialties depends on the null's noise level. The null redraws battles independently, so it ignores correlated votes (same user or session, time clusters) and misfit of the Bradley–Terry model. On the real board the standardized z have a robust SD of 3.3, against 1 under the null. If the true noise were 1.5, 2 or 3 times the null's, Bonferroni would find 149, 76 or 26 specialties instead of 275. That some specialties exist is robust; how many is not.

**Power (Step 3).** `q3_sim.py` plants s specialties of size μ (in z units) into the same simulated Arena, following the Rare/Weak model of Lecture 3: s = N^(1−ϑ), μ = √(2r log N), N = 958. Critical values come from fresh null draws; size is HC 0.06, HC+ 0.05, min-P 0.05, χ² 0.08.

| Power | HC | HC+ | min-P | χ² |
|---|---|---|---|---|
| Dense: ϑ = 0.3 (122 specialties), r = 0.2 (≈ 10 points) | 1.00 | 1.00 | 0.57 | 1.00 |
| Middle: ϑ = 0.5 (31), r = 0.4 (≈ 14 points) | 0.88 | 0.94 | 0.56 | 0.95 |
| Sparse: ϑ = 0.7 (8), r = 0.8 (≈ 20 points) | 0.84 | 0.72 | 0.74 | 0.62 |

χ² and HC+ are best for dense effects, where min-P is far behind. For the sparsest effects χ² falls behind and plain HC is best. This is the ordering Lecture 3 predicts, but the gap is modest: with N = 958 it only shows at the sparse end of the grid. For identification, BH finds 76–93% of 27-point specialties and Bonferroni about 53%. BH's false discovery proportion stays at or below about 12% (0.01–0.12 across the grid), often far below 10%, partly because discoveries in a planted model's other categories count as spillovers rather than as false. Bonferroni's chance of any false discovery is a little above 5% (6.5% with no specialties, up to 9% on the grid), probably because each null SD is estimated from only 500 draws. Categories overlap (many coding battles are also hard prompts), so a planted specialty also raises the model in overlapping categories. Those discoveries (1–23%) are counted as spillovers, not as false.

Figures (`q3_figures.py`): power of the four global tests on the whole grid (`docs/figures/q3_power.png`), Bonferroni and BH against their targets (`q3_identification.png`), the real z* against N(0, 1) (`q3_real_z.png`), and the specialty map of the 40 highest-rated models (`q3_specialties.png`).

**Limitations.** The null is exact Bradley–Terry with fixed strengths, so anything else that varies with the category also shows up as a specialty, such as strengths drifting over time while the category mix changes, or voters who differ between languages. Only the 2024-08 snapshot is used, and the simulation is as idealized as in Q1.

## Data and code

- `clean_battle_20240814_public.json` — the public LMSYS battle dump released with Chiang et al. (2024)'s notebook (1.8M battles, 2023-04 to 2024-08-14; [download](https://storage.googleapis.com/arena_external_data/public/clean_battle_20240814_public.json)). Battle records only: no ratings. A public version of the log behind the official board, missing some recent battles (see Step 0). Not in the repo; pass its path to `prepare_data.py`.
- Official leaderboard snapshot `elo_results_20240813.pkl` (ratings, intervals and ranks computed by LMSYS, for the overall board and 18 category boards; our answer key in Step 0 and Q3 Step 0) from the [`lmarena-ai/chatbot-arena-leaderboard`](https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard) space (downloaded automatically by Step 0; loaded with an allow-list unpickler).
- [`lmarena-ai/leaderboard-dataset`](https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset) — every board Arena has published, with rating, interval, variance and vote count per model; the `text` and `text_style_control` subsets are used by `step0_2026.py` (downloaded automatically to `data/leaderboard_dataset/`, about 115 MB).
- [`lmarena/arena-rank`](https://github.com/lmarena/arena-rank) — Arena's official ranking code; our "Arena method" is exactly its `BradleyTerry` fit.

## Running

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python prepare_data.py --src ~/Downloads/clean_battle_20240814_public.json   # -> data/battles_20240814.parquet
python step0.py              # Step 0: reproduce the 2024-08-12 official leaderboard
python step0_2026.py         # Step 0 for the 2026 site: intervals and rank spread (downloads leaderboard-dataset)
python test_simulate.py      # simulator sanity checks
python q1.py --reps 1000     # Q1: coverage of Arena's rank spread (semi-synthetic)
python q1_analysis.py        # Q1 follow-up: which models have unstable ranks
python q1_figures.py         # figures for docs/q1.md
python test_pairwise.py      # Q2 pairwise procedures: checks against statsmodels
python q2.py --reps 1000 --procedures arena,bonferroni,chi2,chi2_scheffe,pairwise_bonferroni,holm_onesided,holm   # Q2
for a in "--beta-scale 0.5" "--beta-scale 0.25" "--beta-scale 0.1" "--tiers 20" "--tiers 10" \
         "--tiers 20 --tier-spread 0.05" "--tiers 20 --tier-spread 0.1" "--tiers 20 --tier-spread 0.2" "--beta-scale 0"; do
  python q2.py --reps 1000 $a --procedures arena,bonferroni,chi2,chi2_scheffe,pairwise_bonferroni,holm_onesided,holm
done                         # Q2 stress-test worlds (about 4 min each)
pip install jupyter && jupyter nbconvert --to notebook --execute --inplace notebook/q2_*.ipynb   # Q2 write-ups
python q3_step0.py           # Q3 Step 0: category definitions vs the official 2024 category boards
python q3.py --reps 500      # Q3 Step 2: specialty tests on the real board (about 2 min)
python q3_sim.py --reps 200  # Q3 Step 3: power for planted specialties (about 35 min; needs results/q3_null.npz)
python q3_figures.py         # Q3 figures in docs/figures/
```

Results go to `results/`.

## References

- Chiang et al. (2024). *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference.* ICML. [arXiv:2403.04132](https://arxiv.org/abs/2403.04132)
- LMArena (2025). [Arena's Ranking Method](https://arena.ai/blog/ranking-method)
- Holm (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics.*
- Shaffer (1980). Control of directional errors with stagewise multiple test procedures. *Annals of Statistics.*
- Donoho & Jin (2004). Higher criticism for detecting sparse heterogeneous mixtures. *Annals of Statistics.*
- Benjamini & Hochberg (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society B.*
