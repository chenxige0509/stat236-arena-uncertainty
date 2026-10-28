"""Q3 Step 0: check our category definitions against the official 2024-08-13 category leaderboards.

Inputs:
  - the battles behind Step 0 (prepare_data.py; anonymous, dedup-sampled, up to the snapshot time);
  - elo_results_20240813.pkl, which holds 19 text category boards next to the overall one. They are the answer key.
For each Q3 category (categories.py):
  (1) battles:   our category subset gives the official num_battles per model;
  (2) ratings:   arena-rank's BradleyTerry on the subset, anchored like the board (mixtral = 1114), gives the
                 official category ratings;
  (3) specialty: category rating minus overall rating, ours vs official. This difference is what Q3 tests, so it
                 is the check that matters most.
As in Step 0, models released in the last few weeks have fewer battles in the public dump, so (2) and (3) are also
reported for the models whose category battle counts match (|diff| <= 10), and for the pairs Q3 actually tests
(at least MIN_BATTLES battles in the category).

What the results establish (numbers from the current run):
  - (1): all eight definitions reproduce the official battle counts as well as the overall board does in Step 0:
    97-100 of 128 models within +-10 battles, median relative difference 0.00%. Alternatives fail clearly, e.g.
    5 of 7 criteria for hard_prompts (+44%), if_score >= 3 (+35%), sum_user_tokens >= 1000 for long_query (-52%).
  - (2)+(3) on the pairs Q3 tests (>= 200 battles; 93-127 models per category): specialty correlation 0.999-1.000,
    max |diff| 1.3-7.1 points, against an official specialty SD of 14-56 points. The few gaps above 3 points are
    recent models missing battles in the public dump (e.g. deepseek-coder-v2-0724, llama-3.1-70b-instruct).
  - Below 200 battles a few old models disagree by up to 408 points (russian: fastchat-t5-3b,
    stablelm-tuned-alpha-7b, chatglm2-6b; long_query: RWKV-4-Raven-14B). The official values look like a fit that
    did not converge: the three russian models have the identical rating 1137.95 and identical bootstrap draws,
    although they scored only 19-31% of their 36-76 battles. Our arena-rank fit puts them at 730-850, in line
    with those results. Q3 excludes such pairs through MIN_BATTLES anyway.
So the categories in Q3 are the official category boards, and the specialties we test are the ones the board shows.

    python q3_step0.py
Writes results/q3_step0_summary.csv (one row per category) and results/q3_step0_comparison.csv (category x model).
"""
import os
import numpy as np
import pandas as pd
from arena_rank.utils.data_utils import PairDataset
from arena_data import load_battles, load_official, ANCHOR, ANCHOR_RATING, RENAME
from categories import CATEGORIES, OFFICIAL_KEY, MIN_BATTLES, category_mask
from ranking import fit_arena

os.makedirs("results", exist_ok=True)


def anchored_fit(battles):
    """arena-rank fit on `battles`, shifted so the anchor model has the board's anchor rating."""
    fit = fit_arena(PairDataset.from_pandas(battles[["model_a", "model_b", "winner"]])).set_index("competitors")
    return fit.ratings + (ANCHOR_RATING - fit.loc[ANCHOR, "ratings"])


full_off, cutoff, _ = load_official(category="full")
full_off = full_off.set_index("model")
b = load_battles(cutoff=cutoff)
for c in ["model_a", "model_b"]:
    b[c] = b[c].replace(RENAME)
full_ours = anchored_fit(b)
print(f"{len(b):,} battles up to {pd.to_datetime(cutoff, unit='s')} UTC; overall fit on {len(full_ours)} models\n")

summary, long = [], []
for name in CATEGORIES:
    off, cat_cutoff, _ = load_official(category=OFFICIAL_KEY[name])
    off = off.set_index("model")
    # Each board has its own last_updated_tstamp (e.g. chinese is 10 minutes before the overall board).
    assert cat_cutoff <= cutoff, (name, cat_cutoff, cutoff)
    sub = b[category_mask(b, name) & (b.tstamp <= cat_cutoff)]

    # (1) battles
    ours_n = pd.concat([sub.model_a, sub.model_b]).value_counts()
    t = off[["num_battles", "rating"]].rename(columns={"num_battles": "official_battles", "rating": "official_rating"})
    t["our_battles"] = ours_n.reindex(t.index).fillna(0).astype(int)
    d = t.our_battles - t.official_battles
    match = d.abs() <= 10

    # (2) ratings
    t["our_rating"] = anchored_fit(sub).reindex(t.index)
    err = t.our_rating - t.official_rating

    # (3) specialty = category rating - overall rating
    t["official_specialty"] = t.official_rating - full_off.rating.reindex(t.index)
    t["our_specialty"] = t.our_rating - full_ours.reindex(t.index)
    s_err = t.our_specialty - t.official_specialty
    m = match & (t.index != ANCHOR)                       # the anchor's specialty is 0 by construction
    q3 = (t.official_battles >= MIN_BATTLES) & (t.index != ANCHOR)   # the pairs Q3 tests

    summary.append({
        "category": name, "official_board": OFFICIAL_KEY[name],
        "battles": len(sub), "share_of_all": len(sub) / len(b), "models": len(t),
        "battles_exact": int((d == 0).sum()), "battles_within_10": int(match.sum()),
        "battles_median_rel_diff": float(np.median(d / t.official_battles)),
        "rating_median_abs_diff": float(err.abs().median()),
        "rating_max_abs_diff_matched": float(err[match].abs().max()),
        "rating_corr": float(np.corrcoef(t.our_rating, t.official_rating)[0, 1]),
        "specialty_median_abs_diff": float(s_err.abs().median()),
        "specialty_max_abs_diff_matched": float(s_err[m].abs().max()),
        "specialty_corr": float(np.corrcoef(t.our_specialty[t.index != ANCHOR],
                                            t.official_specialty[t.index != ANCHOR])[0, 1]),
        "specialty_sd_official": float(t.official_specialty[t.index != ANCHOR].std()),
        f"models_ge_{MIN_BATTLES}": int(q3.sum()),
        f"rating_max_abs_diff_ge_{MIN_BATTLES}": float(err[q3].abs().max()),
        f"specialty_max_abs_diff_ge_{MIN_BATTLES}": float(s_err[q3].abs().max()),
        f"specialty_corr_ge_{MIN_BATTLES}": float(np.corrcoef(t.our_specialty[q3], t.official_specialty[q3])[0, 1]),
    })
    long.append(t.assign(category=name, battles_match=match).reset_index())

s = pd.DataFrame(summary).set_index("category")
pd.set_option("display.width", 200)
print("(1) battle counts per model (ours - official), out of 128 models:")
print(s[["battles", "share_of_all", "battles_exact", "battles_within_10", "battles_median_rel_diff"]]
      .to_string(float_format=lambda x: f"{x:.4f}"))
print("\n(2) category ratings, anchored at mixtral = 1114:")
print(s[["rating_median_abs_diff", "rating_max_abs_diff_matched", "rating_corr"]].round(4).to_string())
print("\n(3) specialty = category rating - overall rating:")
print(s[["specialty_median_abs_diff", "specialty_max_abs_diff_matched", "specialty_corr", "specialty_sd_official"]]
      .round(4).to_string())
print(f"\n(2)+(3) restricted to the pairs Q3 tests (>= {MIN_BATTLES} battles in the category):")
print(s[[f"models_ge_{MIN_BATTLES}", f"rating_max_abs_diff_ge_{MIN_BATTLES}", f"specialty_max_abs_diff_ge_{MIN_BATTLES}",
         f"specialty_corr_ge_{MIN_BATTLES}"]].round(4).to_string())
s.to_csv("results/q3_step0_summary.csv")
pd.concat(long).to_csv("results/q3_step0_comparison.csv", index=False)
