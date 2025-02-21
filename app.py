import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 📌 Charger les données
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv('HDHI Admission Data.csv', encoding='utf-8')

    # 📌 Vérifier les colonnes
    df.columns = df.columns.str.strip()  # Supprime les espaces autour des noms de colonnes
    print("Colonnes disponibles:", df.columns)

    # Sélectionner les colonnes nécessaires
    selected_columns = ["month year", "HB", "TLC", "PLATELETS", "GLUCOSE", "UREA", "CREATININE", "BNP", "EF"]
    df = df[selected_columns]

    # 📌 Convertir 'month year' en datetime et utiliser comme index
    df["month year"] = pd.to_datetime(df["month year"], format="%b-%y", errors='coerce')

    # Vérifier si certaines dates n'ont pas pu être converties
    if df["month year"].isna().sum() > 0:
        print("⚠️ Certaines dates n'ont pas pu être converties. Vérifie le format dans le CSV.")
        print(df[df["month year"].isna()])

    df.set_index("month year", inplace=True)

    # 📌 Assurer que toutes les colonnes de maladies sont numériques et gérer les NaN
    for col in df.columns:
        # Convertir en numérique (forces les erreurs NaN)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Gérer les NaN (remplacer par 0, ou choisir une autre méthode comme la moyenne)
        df[col].fillna(0, inplace=True)

    return df

# 📌 Entraînement du modèle ARIMA et prévision
@st.cache_data
def forecast_arima(df_monthly, disease, forecast_steps):
    predictions = {}

    if disease not in df_monthly.columns:
        st.error(f"La maladie {disease} n'existe pas dans les données.")
        return predictions

    print(f"🔎 Entraînement du modèle ARIMA pour {disease}...")

    if df_monthly[disease].nunique() < 5:
        print(f"⚠️ Trop peu de données pour {disease}, skipping...")
        return predictions

    # 📌 Vérifier si les données sont bien numériques
    print(f"Type de données pour {disease}: {df_monthly[disease].dtype}")
    
    try:
        # Entraînement du modèle ARIMA
        model = ARIMA(df_monthly[disease], order=(2, 1, 2))  # (p, d, q) ajustables
        model_fit = model.fit()

        # Prédire les prochains mois
        forecast = model_fit.forecast(steps=forecast_steps)
        predictions[disease] = forecast  # Stocke les prévisions pour la maladie

        # 📊 Affichage des prévisions
        fig, ax = plt.subplots(figsize=(10, 4))  # Créer une figure et un axe
        ax.plot(df_monthly[disease], label="Données réelles")
        ax.axvline(df_monthly.index[-1], color='r', linestyle='--', label="Début de la prévision")

        # Corrigé: Utilisation des parenthèses pour la compréhension de générateur
        ax.scatter([df_monthly.index[-1] + pd.DateOffset(months=i) for i in range(1, forecast_steps + 1)],
                   [forecast[i-1] for i in range(1, forecast_steps + 1)],
                   color='red', label="Prévisions")
        
        ax.set_title(f"Prévision ARIMA pour {disease}")
        ax.legend()
        st.pyplot(fig)  # Passer la figure à st.pyplot()

    except Exception as e:
        print(f"❌ Erreur pour {disease} : {e}")
        return predictions

    return predictions

# Main Streamlit interface
def main():
    st.title("Prévisions des maladies des patients")

    # Charger et préparer les données
    df_monthly = load_and_prepare_data()

    # Agréger les données par mois (utiliser "ME" au lieu de "M")
    df_monthly = df_monthly.resample("ME").sum()

    # Sélectionner la maladie
    disease_options = df_monthly.columns.tolist()
    disease = st.selectbox("Sélectionne la maladie", disease_options)

    # Sélectionner le nombre de mois à prédire
    forecast_steps = st.slider("Sélectionne le nombre de mois à prédire", min_value=1, max_value=12, value=1)

    # Prévoir avec ARIMA
    predictions = forecast_arima(df_monthly, disease, forecast_steps)

    # Afficher les prévisions
    if predictions:
        st.subheader(f"Prévisions pour {disease}")
        for i, prediction in enumerate(predictions[disease]):
            prediction_date = df_monthly.index[-1] + pd.DateOffset(months=i+1)
            st.write(f"📅 {prediction_date.strftime('%b %Y')} : {prediction:.2f} patients")

if __name__ == "__main__":
    main()
