import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Configuration de la page
st.set_page_config(page_title="Simulateur Emploi Maroc 2014", layout="centered")

# 1. Chargement du modèle et des colonnes
@st.cache_resource # Pour éviter de recharger le modèle à chaque clic
def load_model():
    model = joblib.load('morocco_employment_model.pkl')
    features = joblib.load('model_features.pkl')
    return model, features

model, feature_names = load_model()

# Dictionnaire de traduction des noms de features
traduction_noms = {
    'AGE5': 'Âge',
    'NIV_ET': 'Niveau d\'études',
    'LIR_ECR': 'Alphabétisation',
    'net': 'Internet',
    'voit': 'Voiture',
    'sexe_2': 'Sexe',
    'mil_2.0': 'Milieu'
}

st.title("📊 Simulateur de Probabilité d'Emploi")
st.markdown("Basé sur les données du recensement **RGPH 2014**.")

# 2. Formulaire utilisateur dans la barre latérale
with st.sidebar:
    st.header("Profil de l'individu")
    age = st.slider("Âge", 15, 100, 30)
    education = st.number_input("Niveau d'études (Années)", 0, 25, 16)
    sexe = st.selectbox("Sexe", ["Homme", "Femme"])
    milieu = st.selectbox("Milieu de résidence", ["Urbain", "Rural"])
    
    st.divider()
    has_net = st.toggle("Possède Internet", value=True)
    has_car = st.toggle("Possède une voiture", value=False)

# 3. Préparation du vecteur de prédiction
input_vector = [0.0] * len(feature_names)

# Remplissage des valeurs numériques
input_vector[feature_names.index('AGE5')] = float(age)
input_vector[feature_names.index('NIV_ET')] = float(education)
input_vector[feature_names.index('LIR_ECR')] = 1.0 # Par défaut lettré

if 'net' in feature_names:
    input_vector[feature_names.index('net')] = 1.0 if has_net else 2.0
if 'voit' in feature_names:
    input_vector[feature_names.index('voit')] = 1.0 if has_car else 0.0

# Encodage Sexe (Femme = sexe_2)
if sexe == "Femme" and 'sexe_2' in feature_names:
    input_vector[feature_names.index('sexe_2')] = 1.0

# Encodage Milieu (Rural = mil_2.0)
if milieu == "Rural" and 'mil_2.0' in feature_names:
    input_vector[feature_names.index('mil_2.0')] = 1.0
# ... (votre code précédent reste identique jusqu'au bouton) ...

# 4. Prédiction et Affichage
if st.button("Calculer la probabilité"):
    prob = model.predict_proba([input_vector])[0][1]
    
    # Stockage dans la session pour que les sections suivantes y aient accès
    st.session_state['prob'] = prob 

# Vérifier si la probabilité a été calculée avant d'afficher la suite
if 'prob' in st.session_state:
    prob = st.session_state['prob'] # Récupération de la valeur calculée
    
    # Affichage du score
    st.metric(label="Probabilité d'être employé", value=f"{prob:.2%}")
    st.progress(prob)
    
    if prob > 0.70:
        st.success("Ce profil présente une très forte intégration au marché du travail.")
    elif prob > 0.40:
        st.info("Ce profil présente une intégration modérée.")
    else:
        st.warning("Ce profil présente des barrières structurelles à l'emploi (contexte 2014).")

    # --- Section Graphique ---
    st.divider()
    st.header("🔍 Analyse des déterminants")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]
    noms_propre = [traduction_noms.get(feature_names[i], feature_names[i]) for i in indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(indices)), importances[indices], color='#2ecc71', align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(noms_propre)
    ax.set_title('Top 10 des facteurs influençant l\'emploi')
    st.pyplot(fig)

    # --- Section Export ---
    st.divider()
    st.header("💾 Exporter le résultat")

    data_export = {
        "Variable": ["Âge", "Éducation", "Sexe", "Milieu", "Internet", "Voiture", "Probabilité d'emploi"],
        "Valeur": [age, education, sexe, milieu, 
                   "Oui" if has_net else "Non", 
                   "Oui" if has_car else "Non", 
                   f"{prob:.2%}"]
    }

    df_export = pd.DataFrame(data_export)

    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8-sig')

    csv_data = convert_df(df_export)

    st.download_button(
        label="📥 Télécharger le rapport de simulation (CSV)",
        data=csv_data,
        file_name=f"simulation_emploi_{sexe}_{age}ans.csv",
        mime="text/csv"
    )

st.info("Note : Ce modèle reflète les tendances historiques de 2014.")