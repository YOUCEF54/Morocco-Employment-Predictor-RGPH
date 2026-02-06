# 📊 Morocco Employment Predictor (RGPH 2014)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://VOTRE_APPLICATION.streamlit.app)

Ce projet utilise l'Intelligence Artificielle pour modéliser et prédire les probabilités d'emploi au Maroc en se basant sur les données massives du recensement **RGPH 2014** (HCP).

## 🚀 Fonctionnalités
- **Analyse de données massives** : Traitement et fusion de plus de 3 millions de lignes de données censitaires.
- **Modèle Prédictif** : Random Forest Classifier avec une précision de **83.7%**.
- **Interface Interactive** : Application Streamlit permettant de tester des profils (Âge, Sexe, Éducation, Région).
- **Explainable AI** : Visualisation en temps réel des facteurs les plus influents (Importance des variables).

## 🛠️ Installation
Pour faire tourner l'application localement :
1. Cloner le repo : `git clone https://github.com/VOTRE_NOM/Morocco-Employment-Predictor-RGPH.git`
2. Installer les dépendances : `pip install -r requirements.txt`
3. Lancer l'app : `streamlit run app.py`

## 📈 Résultats du Modèle
Le modèle identifie le genre, l'éducation et l'âge comme les piliers de l'accès à l'emploi en 2014.
- **Précision globale** : 83.7%
- **Rappel (Recall) Employés** : 78%
