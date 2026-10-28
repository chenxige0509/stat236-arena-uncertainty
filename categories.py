"""Q3 category definitions: which battles belong to each category leaderboard.

Each definition is chosen to reproduce the official 2024-08-13 category board of the same name (q3_step0.py checks
the battle counts and ratings). The fields come from the public dump's `category_tag` and `conv_metadata`, flattened
by prepare_data.py.

  Q3 name                official board  battle is in the category when
  coding                 coding          is_code
  math                   math            category_tag.math_v0.1.math
  instruction_following  if              category_tag.if_v0.1.if (equivalently if_score >= 4)
  hard_prompts           hard_6          at least 6 of the 7 criteria_v0.1 tags are true
  chinese                chinese         language == "Chinese"
  russian                russian         language == "Russian"
  multi_turn             multiturn       turn > 1
  long_query             long_user       sum_user_tokens >= 500

Categories overlap: a battle can be in several of them, or in none.

Q3 only tests a (model, category) pair with at least MIN_BATTLES battles in that category. Below that the category
rating is too noisy to test, and on the 2024 board a few such models have official ratings that look like a
non-converged fit (see q3_step0.py).
"""
import pandas as pd

OFFICIAL_KEY = {
    "coding": "coding",
    "math": "math",
    "instruction_following": "if",
    "hard_prompts": "hard_6",
    "chinese": "chinese",
    "russian": "russian",
    "multi_turn": "multiturn",
    "long_query": "long_user",
}
CATEGORIES = list(OFFICIAL_KEY)
CRITERIA = ["specificity", "domain_knowledge", "complexity", "problem_solving",
            "creativity", "technical_accuracy", "real_world"]
HARD_MIN_CRITERIA = 6
LONG_QUERY_TOKENS = 500
MIN_BATTLES = 200


def _flag(s):
    return s.fillna(False).astype(bool)


def category_mask(battles, name):
    """Boolean Series: which rows of `battles` (from arena_data.load_battles) are in category `name`."""
    b = battles
    if name == "coding":
        return _flag(b.is_code)
    if name == "math":
        return _flag(b.math)
    if name == "instruction_following":
        return _flag(b["if"])
    if name == "hard_prompts":
        n_true = pd.concat([_flag(b[f"criteria_{c}"]) for c in CRITERIA], axis=1).sum(axis=1)
        return n_true >= HARD_MIN_CRITERIA
    if name == "chinese":
        return b.language.astype(str).eq("Chinese")
    if name == "russian":
        return b.language.astype(str).eq("Russian")
    if name == "multi_turn":
        return b.turn.fillna(1) > 1
    if name == "long_query":
        return b.sum_user_tokens.fillna(0) >= LONG_QUERY_TOKENS
    raise ValueError(name)
