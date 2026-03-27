# Analyse d'avis clients — Assurance française

Ce projet applique des techniques de NLP à un corpus de 34 388 avis clients issus de 56 compagnies d'assurance françaises.
L'objectif est d'analyser, classifier et explorer ces avis via plusieurs modules : prédiction de sentiment, résumé automatique, recherche d'information et question-réponse.
Le modèle de classification repose sur un pipeline TF-IDF + Régression Logistique.

Nassim LOUDIYI & Paul-Adrien LU-YEN-TUNG — ESILV DIA4 2025/2026

---

## Fichiers nécessaires

Placer les fichiers suivants à la racine du projet :

```
avis_clients_clean.csv    # Dataset principal (34 388 avis)
sentiment_classifier.pkl  # Modèle de classification entraîné
lda_model.pkl             # Modèle LDA
count_vectorizer.pkl      # Vectoriseur associé au modèle LDA
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Lancer l'application

```bash
python -m streamlit run app.py
```

---

## Pages disponibles

| Page | Description |
|------|-------------|
| **Accueil** | Présentation du projet, statistiques du dataset et aperçu des fonctionnalités. |
| **Prediction** | Saisissez un avis client : le modèle prédit son sentiment et estime la note associée. |
| **Summary** | Résumé extractif automatique des avis d'un assureur, filtrable par sentiment. |
| **Explanation** | Visualisation des mots les plus influents dans la prédiction via les coefficients du modèle. |
| **Information Retrieval** | Recherche d'avis pertinents par mots-clés avec BM25 ou similarité cosinus TF-IDF. |
| **RAG** | Retrieval-Augmented Generation : récupération des avis proches d'une requête puis résumé automatique. |
| **QA** | Question-réponse : le système retrouve les avis sources et en extrait la phrase la plus pertinente. |
