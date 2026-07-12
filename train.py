"""Train the NCF model on MovieLens 1M and save it for the Streamlit app.

Usage: python train.py
"""
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf

from data import build_interactions, load_ratings_and_movies, sample_eval_candidates, sample_negatives
from model import build_ncf_model

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
RANDOM_STATE = 42
BATCH_SIZE = 256
EPOCHS = 5
TOP_K = 10


def make_tf_dataset(frame, batch_size, shuffle):
    features = {
        "user_idx": frame["user_idx"].to_numpy(dtype=np.int32),
        "item_idx": frame["item_idx"].to_numpy(dtype=np.int32),
    }
    labels = frame["label"].to_numpy(dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(frame), seed=RANDOM_STATE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def evaluate_hit_rate_ndcg(model, candidates_by_user, heldout_by_user, k=TOP_K):
    hits, ndcgs = [], []
    for user_idx, candidate_items in candidates_by_user.items():
        user_array = np.full(len(candidate_items), user_idx, dtype=np.int32)
        scores = model.predict({"user_idx": user_array, "item_idx": candidate_items}, verbose=0).reshape(-1)
        top_k_items = candidate_items[np.argsort(scores)[-k:][::-1]]

        heldout_item = heldout_by_user[user_idx]
        hit = heldout_item in top_k_items
        hits.append(hit)
        if hit:
            rank = int(np.where(top_k_items == heldout_item)[0][0])
            ndcgs.append(1.0 / np.log2(rank + 2))
        else:
            ndcgs.append(0.0)
    return float(np.mean(hits)), float(np.mean(ndcgs))


def main():
    print("Loading MovieLens 1M ...")
    ratings, movies = load_ratings_and_movies(DATA_DIR)

    print("Building implicit interactions and leave-one-out split ...")
    data = build_interactions(ratings, movies, random_state=RANDOM_STATE)
    print(f"users={data['n_users']} items={data['n_items']} train={len(data['train_df'])} test={len(data['test_df'])}")

    print("Sampling negative examples ...")
    training_examples = sample_negatives(data["train_df"], data["positive_items_by_user"], data["n_items"], random_state=RANDOM_STATE)
    train_ds = make_tf_dataset(training_examples, BATCH_SIZE, shuffle=True)

    print("Training NCF model ...")
    model = build_ncf_model(data["n_users"], data["n_items"])
    model.fit(train_ds, epochs=EPOCHS)

    print("Evaluating on held-out interactions (HR@10 / NDCG@10) ...")
    candidates_by_user, heldout_by_user = sample_eval_candidates(
        data["test_df"], data["positive_items_by_user"], data["n_items"], random_state=RANDOM_STATE + 1
    )
    hit_rate, ndcg = evaluate_hit_rate_ndcg(model, candidates_by_user, heldout_by_user, k=TOP_K)
    print(f"HR@{TOP_K}={hit_rate:.3f}  NDCG@{TOP_K}={ndcg:.3f}")

    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_DIR / "ncf_model.keras")
    with open(MODEL_DIR / "artifacts.pkl", "wb") as f:
        pickle.dump(
            {
                "user_to_idx": data["user_to_idx"],
                "idx_to_user": data["idx_to_user"],
                "idx_to_movie": data["idx_to_movie"],
                "positive_items_by_user": data["positive_items_by_user"],
                "movie_lookup": data["movie_lookup"],
                "n_items": data["n_items"],
                "hit_rate_at_10": hit_rate,
                "ndcg_at_10": ndcg,
            },
            f,
        )
    print(f"Saved model and artifacts to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
