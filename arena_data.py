"""Data loading: the public battle dump (after prepare_data.py) and the official leaderboard snapshot."""
import os
import pickle
import urllib.request
import pandas as pd

BATTLES_PATH = "data/battles_20240814.parquet"
OFFICIAL_DATE = "20240813"   # closest published snapshot to the 20240814 battle dump
OFFICIAL_URL = "https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard/resolve/main/elo_results_{date}.pkl"


def load_battles(path=BATTLES_PATH, cutoff=None, anony_only=True, dedup=True):
    """Battles filtered as in the official 2024 leaderboard notebook: anonymous battles only, and only the
    `dedup_tag.sampled` subset (down-samples prompts that appear very many times).
    `cutoff`: keep battles with tstamp <= cutoff (unix seconds), e.g. a snapshot's last_updated_tstamp."""
    df = pd.read_parquet(path)
    keep = pd.Series(True, index=df.index)
    if anony_only:
        keep &= df.anony.astype(bool)
    if dedup:
        keep &= df.dedup_sampled.astype(bool)
    if cutoff is not None:
        keep &= df.tstamp <= cutoff
    df = df[keep].reset_index(drop=True)
    for c in ["model_a", "model_b", "winner"]:
        df[c] = df[c].astype(str)
    return df


class _Stub:
    """Placeholder for plotly figures stored in the pickle (not needed, not imported)."""
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass


class _SafeUnpickler(pickle.Unpickler):
    """Only pandas / numpy objects are reconstructed; anything else is refused (pickles can run code)."""
    def find_class(self, module, name):
        if module.startswith(("pandas", "numpy")) or (module, name) == ("builtins", "slice"):
            return super().find_class(module, name)
        if module.startswith("plotly"):
            return _Stub
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def load_official(date=OFFICIAL_DATE, category="full", cache_dir="data/official"):
    """Official leaderboard table for one snapshot and category. Returns (table, last_updated_tstamp).
    Table columns: model, rating, variance, rating_q025, rating_q975, num_battles, final_ranking."""
    path = os.path.join(cache_dir, f"elo_results_{date}.pkl")
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        urllib.request.urlretrieve(OFFICIAL_URL.format(date=date), path)
    with open(path, "rb") as f:
        res = _SafeUnpickler(f).load()["text"][category]
    table = res["leaderboard_table_df"].rename_axis("model").reset_index()
    return table, float(res["last_updated_tstamp"]), res["bootstrap_df"]
