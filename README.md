# recommender_systems_project

Système de recommandation de films basé sur **Neural Collaborative Filtering (NCF)**,
inspiré du notebook de cours `neural-collaborative-filtering-ii.ipynb`, servi via une
app **Streamlit** dans un conteneur **Docker**.

## Idée du modèle

Comme dans le notebook, chaque note >= 4/5 est convertie en interaction implicite
(label `1`). Pour chaque interaction positive on tire quelques films non vus par
l'utilisateur comme négatifs (label `0`). Le modèle combine deux branches :

- **GMF** : produit élément par élément des embeddings user/item
- **MLP** : concaténation des embeddings passée dans un petit réseau dense

Contrairement au notebook (qui pré-entraîne GMF et MLP séparément puis transfère
les poids dans NeuMF), ici le modèle combiné est entraîné **directement** en une
seule passe — plus simple à suivre, même architecture.

L'évaluation utilise un split *leave-one-out* : la dernière interaction de chaque
utilisateur est retirée du train et sert à calculer HR@10 / NDCG@10 en la classant
parmi 100 films non vus tirés au hasard (même protocole que le notebook).

Dataset : [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) (téléchargé
automatiquement au premier lancement).

## Fichiers

- `data.py` — téléchargement, prétraitement, negative sampling
- `model.py` — architecture NeuMF (Keras)
- `train.py` — entraîne le modèle et sauvegarde `models/ncf_model.keras` + `models/artifacts.pkl`
- `app.py` — interface Streamlit : choisir un utilisateur, voir ses recommandations

## Lancer avec Docker

```bash
docker compose up --build
```

Au premier lancement, le conteneur télécharge MovieLens 1M et entraîne le modèle
(quelques minutes sur CPU). Les dossiers `data/` et `models/` sont montés en volumes,
donc les lancements suivants sont instantanés (pas de re-téléchargement/re-entraînement).

Puis ouvre [http://localhost:8501](http://localhost:8501).

Pour ré-entraîner depuis zéro : supprime `models/` et relance.

## Lancer en local (sans Docker)

```bash
pip install -r requirements.txt
python train.py      # entraîne et sauvegarde le modèle
streamlit run app.py
```
