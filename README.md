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
- `experiment.py` — compare plusieurs configs (couches MLP, epochs) sans toucher à Docker
- `app.py` — interface Streamlit : choisir un utilisateur, voir ses recommandations

## Expérimenter sur l'architecture (couches, epochs...)

Ne fais pas ça à travers Docker : le code est copié dans l'image au build, donc
chaque changement demanderait un rebuild, et comme `models/` persiste en volume,
`entrypoint.sh` ne réentraînerait même pas automatiquement (il ne s'exécute que si
`ncf_model.keras` est absent). Pour itérer vite, travaille en local :

```bash
pip install -r requirements.txt
python experiment.py
```

Le script essaie plusieurs configurations (définies dans `CONFIGS` en haut du
fichier — modifie-les librement) et affiche/enregistre un tableau comparatif
HR@10 / NDCG@10 / temps d'entraînement dans `models/experiment_results.csv`.

Une fois la meilleure config trouvée, reporte-la dans `model.py`
(`HIDDEN_UNITS`, `EMBEDDING_DIM`, ...) et/ou `train.py` (`EPOCHS`), puis :

```bash
rm -rf models/            # sinon l'ancien modèle est réutilisé tel quel
docker compose up --build
```

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

## Historique des changements

**Architecture** : `HIDDEN_UNITS` passé de `[64, 32]` à `[128, 64, 32]` dans
`model.py`. `experiment.py` a montré que cette config bat la baseline sur
HR@10 (+0.011) et NDCG@10 (+0.003) pour ~20s d'entraînement en plus, alors que
passer de 5 à 10 epochs faisait légèrement *baisser* les deux métriques
(signe de surapprentissage). Voir `models/experiment_results.csv` pour le
détail des configs testées.

**Bug corrigé — Streamlit plantait (Segmentation fault) dès qu'on changeait
d'utilisateur** : la cause n'était ni le modèle ni le dataset, mais `pyarrow`
(dépendance de Streamlit pour sérialiser les tableaux vers le navigateur), qui
n'était pas épinglé dans `requirements.txt`. La toute dernière version
installée (25.0.0) plante en combinaison avec TensorFlow, spécifiquement au
*second* rendu d'un tableau dans le même process (donc jamais au premier
chargement, uniquement après une interaction — d'où le crash "instantané" au
changement d'utilisateur). Fix : épingler `pyarrow==16.1.0` dans
`requirements.txt`. Reproduit et vérifié via `streamlit.testing.v1.AppTest`
(exécute l'app comme le ferait un vrai navigateur, sans en ouvrir un).

**Taille de l'image Docker (~2.4 Go)** : ce n'est pas lié à MovieLens 1M —
`data/` et `models/` sont montés en volumes, jamais copiés dans l'image (voir
`Dockerfile`/`docker-compose.yml`). Le poids vient de TensorFlow lui-même,
qui fait ~1.5-2 Go une fois installé avec ses dépendances ; passer à MovieLens
small ne changerait rien à la taille de l'image, seulement au temps
d'entraînement.
