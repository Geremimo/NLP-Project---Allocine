# Analyse de sentiment sur les critiques Allociné 🎬

Mini-projet de machine learning / NLP : prédire si une critique de film en français est **positive** ou **négative**, à partir du texte seul.

Projet réalisé pendant l'été avant le début d'un M2 Data Science, dans le but de se remettre dans le bain du ML et de découvrir les bases du NLP.

## Objectif

Étant donné le texte d'une critique de film, prédire automatiquement le sentiment de l'auteur (positif/négatif) — un problème de **classification binaire supervisée**, ici appliqué à du texte plutôt qu'à des données tabulaires classiques.

## Dataset

[Allociné](https://huggingface.co/datasets/tblard/allocine) — un dataset public de ~200 000 vraies critiques de films en français, écrites par la communauté Allociné.fr entre 2006 et 2020.

- **Labels** : dérivés des notes originales (≤2 → négatif, ≥4 → positif)
- **Répartition** : train (160k), validation (20k), test (20k)
- **Source** : dataset assemblé par Théophile Blard ([GitHub](https://github.com/TheophileBlard/french-sentiment-analysis-with-bert)), disponible sur HuggingFace

Le dataset n'est pas versionné dans ce repo — il est téléchargé automatiquement via la librairie `datasets` au premier lancement du notebook.

## Démarche

1. **Exploration** des données (équilibre des classes, longueur des textes)
2. **Nettoyage** du texte (minuscules, ponctuation, stopwords)
3. **Vectorisation** avec TF-IDF (unigrammes + bigrammes)
4. **Modèle baseline** : régression logistique
5. **Évaluation** : accuracy, precision/recall/F1, matrice de confusion

D'autres étapes (comparaison avec d'autres modèles, interprétation des coefficients, éventuellement un modèle pré-entraîné type CamemBERT) sont prévues en complément.

## Résultats

*(à compléter une fois les expériences terminées)*

| Modèle | Accuracy (validation) | F1-score |
|---|---|---|
| TF-IDF + Régression Logistique | — | — |

## Installation

```bash
python3 -m venv venv
source venv/bin/activate       # sous macOS/Linux
pip install -r requirements.txt
```

## Lancer le projet

```bash
jupyter notebook notebooks/allocine_sentiment_jour1_2.ipynb
```

Exécuter les cellules dans l'ordre. Le premier chargement du dataset peut prendre 1-2 minutes (téléchargement + mise en cache).

## Structure du repo

```
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    └── allocine_sentiment_jour1_2.ipynb
```

## Remerciements

Dataset Allociné construit et publié par [Théophile Blard](https://github.com/TheophileBlard), à partir de contenus de la communauté [Allociné.fr](https://www.allocine.fr/).
