"""Streamlit demo for the NCF movie recommender trained by train.py."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

MODEL_DIR = Path("models")
TOP_K = 10


@st.cache_resource
def load_model_and_artifacts():
    model = tf.keras.models.load_model(MODEL_DIR / "ncf_model.keras")
    with open(MODEL_DIR / "artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)
    return model, artifacts


def recommend_for_user(model, artifacts, user_idx, top_k=TOP_K):
    seen_items = artifacts["positive_items_by_user"].get(user_idx, set())
    all_items = np.arange(artifacts["n_items"], dtype=np.int32)
    candidate_items = np.setdiff1d(all_items, np.fromiter(seen_items, dtype=np.int32))

    user_array = np.full(len(candidate_items), user_idx, dtype=np.int32)
    scores = model.predict({"user_idx": user_array, "item_idx": candidate_items}, verbose=0).reshape(-1)

    top_positions = np.argsort(scores)[-top_k:][::-1]
    top_items = candidate_items[top_positions]
    top_scores = scores[top_positions]

    movie_lookup = artifacts["movie_lookup"]
    rows = [
        {
            "title": movie_lookup.loc[item_idx, "title"],
            "genres": movie_lookup.loc[item_idx, "genres"],
            "score": float(score),
        }
        for item_idx, score in zip(top_items, top_scores)
    ]
    return pd.DataFrame(rows)


def already_liked(artifacts, user_idx, n=5):
    seen_items = list(artifacts["positive_items_by_user"].get(user_idx, set()))[:n]
    movie_lookup = artifacts["movie_lookup"]
    return movie_lookup.loc[seen_items, ["title", "genres"]]


st.set_page_config(page_title="Movie Recommender (NCF)", page_icon="🎬")
st.title("🎬 Movie Recommender — Neural Collaborative Filtering")
st.caption("MovieLens 1M · GMF + MLP (NeuMF) trained on implicit feedback")

model_path = MODEL_DIR / "ncf_model.keras"
if not model_path.exists():
    st.error("Aucun modèle entraîné trouvé. Lance d'abord `python train.py`.")
    st.stop()

model, artifacts = load_model_and_artifacts()

st.sidebar.header("Métriques du modèle")
st.sidebar.metric("HR@10", f"{artifacts['hit_rate_at_10']:.3f}")
st.sidebar.metric("NDCG@10", f"{artifacts['ndcg_at_10']:.3f}")

user_ids = sorted(artifacts["user_to_idx"].keys())
selected_user_id = st.selectbox("Choisis un utilisateur MovieLens", user_ids)
user_idx = artifacts["user_to_idx"][selected_user_id]

st.subheader("Quelques films déjà aimés par cet utilisateur")
st.dataframe(already_liked(artifacts, user_idx), hide_index=True, use_container_width=True)

st.subheader(f"Top {TOP_K} recommandations")
with st.spinner("Calcul des scores..."):
    recommendations = recommend_for_user(model, artifacts, user_idx)
st.dataframe(recommendations, hide_index=True, use_container_width=True)
