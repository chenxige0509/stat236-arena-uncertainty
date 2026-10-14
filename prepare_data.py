"""Convert the public battle dump (clean_battle_20240814_public.json, a single 2.1 GB JSON array)
into a flat parquet file, keeping every battle and every field we may need (Step 0, Q1, Q3).

    python prepare_data.py [--src ~/Downloads/clean_battle_20240814_public.json]

Writes data/battles_20240814.parquet. Filtering (anony / dedup) is done later, in arena_data.py.
"""
import argparse
import os
import ijson
import pandas as pd

CRITERIA = ["specificity", "domain_knowledge", "complexity", "problem_solving",
            "creativity", "technical_accuracy", "real_world"]
META = ["sum_user_tokens", "sum_assistant_a_tokens", "sum_assistant_b_tokens",
        "context_a_tokens", "context_b_tokens", "turns"]


def flatten(b):
    tag = b.get("category_tag") or {}
    crit = tag.get("criteria_v0.1") or {}
    ifv = tag.get("if_v0.1") or {}
    meta = b.get("conv_metadata") or {}
    dedup = b.get("dedup_tag") or {}
    row = {
        "model_a": b["model_a"], "model_b": b["model_b"], "winner": b["winner"],
        "judge": b.get("judge"), "turn": b.get("turn"), "anony": b.get("anony"),
        "language": b.get("language"), "tstamp": b.get("tstamp"),
        "is_code": b.get("is_code"), "is_refusal": b.get("is_refusal"),
        "dedup_high_freq": dedup.get("high_freq"), "dedup_sampled": dedup.get("sampled"),
        "if": ifv.get("if"), "if_score": ifv.get("score"),
        "math": (tag.get("math_v0.1") or {}).get("math"),
    }
    row |= {f"criteria_{c}": crit.get(c) for c in CRITERIA}
    row |= {m: meta.get(m) for m in META}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.expanduser("~/Downloads/clean_battle_20240814_public.json"))
    ap.add_argument("--out", default="data/battles_20240814.parquet")
    args = ap.parse_args()

    with open(args.src, "rb") as f:
        rows = [flatten(b) for b in ijson.items(f, "item", use_float=True)]
    df = pd.DataFrame(rows)
    for c in ["model_a", "model_b", "winner", "language", "judge"]:
        df[c] = df[c].astype("category")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"{len(df):,} battles, {df.shape[1]} columns -> {args.out}")
    print(df.winner.value_counts().to_string())
    print(f"time range: {pd.to_datetime(df.tstamp.min(), unit='s')} .. {pd.to_datetime(df.tstamp.max(), unit='s')} UTC")


if __name__ == "__main__":
    main()
