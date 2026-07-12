"""Load MovieLens 1M and turn explicit ratings into implicit interactions.

Same idea as the course notebook (neural-collaborative-filtering-ii.ipynb):
- every observed rating >= MIN_POSITIVE_RATING counts as a positive interaction (label 1)
- for each positive, we sample a few movies the user never rated as negatives (label 0)
- for each user, the single latest interaction is held out for evaluation (leave-one-out)
"""
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ML_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MIN_POSITIVE_RATING = 4.0
NEGATIVES_PER_POSITIVE = 4
EVAL_NEGATIVES = 100


def download_movielens(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ml-1m.zip"
    if not zip_path.exists():
        print(f"Downloading MovieLens 1M to {zip_path} ...")
        urllib.request.urlretrieve(ML_1M_URL, zip_path)
    return zip_path


def load_ratings_and_movies(data_dir: Path):
    zip_path = download_movielens(data_dir)
    with zipfile.ZipFile(zip_path) as zf:
        ratings = pd.read_csv(
            zf.open("ml-1m/ratings.dat"),
            sep="::",
            engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"],
            encoding="latin-1",
        )
        movies = pd.read_csv(
            zf.open("ml-1m/movies.dat"),
            sep="::",
            engine="python",
            names=["movie_id", "title", "genres"],
            encoding="latin-1",
        )
    return ratings, movies


def build_interactions(ratings: pd.DataFrame, movies: pd.DataFrame, random_state: int = 42):
    """Keep only positive ratings, map ids to dense indices, and split leave-one-out."""
    positives = ratings.loc[ratings["rating"] >= MIN_POSITIVE_RATING].copy()
    positives = positives.sort_values(["user_id", "timestamp"])

    # drop users with a single interaction: nothing left to hold out for evaluation
    counts = positives.groupby("user_id")["movie_id"].transform("size")
    positives = positives.loc[counts >= 2].copy()

    user_ids = np.sort(positives["user_id"].unique())
    movie_ids = np.sort(positives["movie_id"].unique())
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    movie_to_idx = {m: i for i, m in enumerate(movie_ids)}
    idx_to_user = {i: u for u, i in user_to_idx.items()}
    idx_to_movie = {i: m for m, i in movie_to_idx.items()}

    positives["user_idx"] = positives["user_id"].map(user_to_idx).astype(np.int32)
    positives["item_idx"] = positives["movie_id"].map(movie_to_idx).astype(np.int32)

    # leave-one-out: the last interaction per user goes to the test set
    positives["rank"] = positives.groupby("user_idx").cumcount(ascending=False)
    train_df = positives.loc[positives["rank"] > 0].reset_index(drop=True)
    test_df = positives.loc[positives["rank"] == 0].reset_index(drop=True)

    positive_items_by_user = positives.groupby("user_idx")["item_idx"].apply(set).to_dict()

    movie_lookup = movies.loc[movies["movie_id"].isin(movie_ids)].copy()
    movie_lookup["item_idx"] = movie_lookup["movie_id"].map(movie_to_idx)
    movie_lookup = movie_lookup.set_index("item_idx").sort_index()

    return {
        "train_df": train_df,
        "test_df": test_df,
        "n_users": len(user_ids),
        "n_items": len(movie_ids),
        "user_to_idx": user_to_idx,
        "idx_to_user": idx_to_user,
        "movie_to_idx": movie_to_idx,
        "idx_to_movie": idx_to_movie,
        "positive_items_by_user": positive_items_by_user,
        "movie_lookup": movie_lookup,
    }


def sample_negatives(train_df, positive_items_by_user, n_items, n_negatives=NEGATIVES_PER_POSITIVE, random_state=42):
    """For every positive (user, item), sample a few items the user hasn't interacted with."""
    rng = np.random.default_rng(random_state)
    all_items = np.arange(n_items, dtype=np.int32)

    rows = []
    for user_idx, group in train_df.groupby("user_idx"):
        seen = positive_items_by_user[user_idx]
        candidates = np.setdiff1d(all_items, np.fromiter(seen, dtype=np.int32), assume_unique=False)
        if len(candidates) == 0:
            continue
        sample_size = min(n_negatives, len(candidates)) * len(group)
        sampled = rng.choice(candidates, size=sample_size, replace=True)
        rows.append(pd.DataFrame({"user_idx": user_idx, "item_idx": sampled, "label": 0.0}))

    negatives = pd.concat(rows, ignore_index=True)
    positives = train_df[["user_idx", "item_idx"]].copy()
    positives["label"] = 1.0

    examples = pd.concat([positives, negatives], ignore_index=True)
    return examples.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def sample_eval_candidates(test_df, positive_items_by_user, n_items, n_negatives=EVAL_NEGATIVES, random_state=42):
    """For each test user: the held-out item + a handful of items never seen by that user."""
    rng = np.random.default_rng(random_state)
    all_items = np.arange(n_items, dtype=np.int32)

    candidates_by_user, heldout_by_user = {}, {}
    for user_idx, item_idx in test_df[["user_idx", "item_idx"]].itertuples(index=False):
        seen = positive_items_by_user[user_idx]
        candidates = np.setdiff1d(all_items, np.fromiter(seen, dtype=np.int32), assume_unique=False)
        sample_size = min(n_negatives, len(candidates))
        negatives = rng.choice(candidates, size=sample_size, replace=False)
        candidates_by_user[user_idx] = np.concatenate([[item_idx], negatives]).astype(np.int32)
        heldout_by_user[user_idx] = item_idx

    return candidates_by_user, heldout_by_user
