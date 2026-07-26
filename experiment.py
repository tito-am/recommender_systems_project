"""Compare a few NCF configurations (MLP layers, epochs) on MovieLens 1M.

Run locally, not through Docker -- this is meant for a fast iteration loop:

    python experiment.py

Edit the CONFIGS list below to try your own combinations. Once you've picked
a winner, set HIDDEN_UNITS / epochs in model.py and train.py accordingly and
run `python train.py` (or rebuild the Docker image) to produce the model the
Streamlit app actually serves.
"""
import time
from pathlib import Path

import pandas as pd

from data import build_interactions, load_ratings_and_movies, sample_eval_candidates, sample_negatives
from model import build_ncf_model
from train import BATCH_SIZE, DATA_DIR, RANDOM_STATE, TOP_K, evaluate_hit_rate_ndcg, make_tf_dataset

CONFIGS = [
    {"name": "baseline [64,32] x5", "hidden_units": [64, 32], "epochs": 5},
    {"name": "deeper [128,64,32] x5", "hidden_units": [128, 64, 32], "epochs": 5},
    {"name": "shallower [32,16] x5", "hidden_units": [32, 16], "epochs": 5},
    {"name": "baseline [64,32] x10", "hidden_units": [64, 32], "epochs": 10},
]


def main():
    print("Loading MovieLens 1M ...")
    ratings, movies = load_ratings_and_movies(DATA_DIR)
    data = build_interactions(ratings, movies, random_state=RANDOM_STATE)
    print(f"users={data['n_users']} items={data['n_items']} train={len(data['train_df'])} test={len(data['test_df'])}")

    print("Sampling negative examples ...")
    training_examples = sample_negatives(data["train_df"], data["positive_items_by_user"], data["n_items"], random_state=RANDOM_STATE)
    train_ds = make_tf_dataset(training_examples, BATCH_SIZE, shuffle=True)

    candidates_by_user, heldout_by_user = sample_eval_candidates(
        data["test_df"], data["positive_items_by_user"], data["n_items"], random_state=RANDOM_STATE + 1
    )

    results = []
    for config in CONFIGS:
        print(f"\n=== {config['name']} ===")
        start = time.time()
        model = build_ncf_model(data["n_users"], data["n_items"], hidden_units=config["hidden_units"])
        history = model.fit(train_ds, epochs=config["epochs"], verbose=2)
        train_seconds = round(time.time() - start, 1)

        hit_rate, ndcg = evaluate_hit_rate_ndcg(model, candidates_by_user, heldout_by_user, k=TOP_K)
        results.append(
            {
                "config": config["name"],
                "hidden_units": config["hidden_units"],
                "epochs": config["epochs"],
                "final_loss": round(history.history["loss"][-1], 4),
                f"hr_at_{TOP_K}": round(hit_rate, 3),
                f"ndcg_at_{TOP_K}": round(ndcg, 3),
                "train_seconds": train_seconds,
            }
        )

    results_df = pd.DataFrame(results)
    print("\n=== Comparison ===")
    print(results_df.to_string(index=False))

    Path("models").mkdir(exist_ok=True)
    results_df.to_csv("models/experiment_results.csv", index=False)
    print("\nSaved to models/experiment_results.csv")


if __name__ == "__main__":
    main()
