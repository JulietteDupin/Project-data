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

    # Sélectionner les colonnes nécessaires, y compris le nombre de lits d'hôpital
    selected_columns = ["month year", "DURATION OF STAY"]  # Supposons que "DURATION OF STAY" représente les lits
    df = df[selected_columns]

    # 📌 Convertir 'month year' en datetime et utiliser comme index
    df["month year"] = pd.to_datetime(df["month year"], format="%b-%y", errors='coerce')

    # Vérifier si certaines dates n'ont pas pu être converties
    if df["month year"].isna().sum() > 0:
        print("⚠️ Certaines dates n'ont pas pu être converties. Vérifie le format dans le CSV.")
        print(df[df["month year"].isna()])

    df.set_index("month year", inplace=True)

    # 📌 Assurer que toutes les colonnes sont numériques et gérer les NaN
    for col in df.columns:
        # Convertir en numérique (forces les erreurs NaN)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Gérer les NaN (remplacer par 0, ou choisir une autre méthode comme la moyenne)
        df[col].fillna(0, inplace=True)

    return df

# 📌 Entraînement du modèle ARIMA et prévision
@st.cache_data
def forecast_arima(df_monthly, forecast_steps):
    predictions = {}

    if "DURATION OF STAY" not in df_monthly.columns:
        st.error("La colonne 'DURATION OF STAY' n'existe pas dans les données.")
        return predictions

    print(f"🔎 Entraînement du modèle ARIMA pour le nombre de lits d'hôpital...")

    # 📌 Vérifier si les données sont bien numériques
    print(f"Type de données pour 'DURATION OF STAY': {df_monthly['DURATION OF STAY'].dtype}")
    
    try:
        # Entraînement du modèle ARIMA
        model = ARIMA(df_monthly["DURATION OF STAY"], order=(2, 1, 2))  # (p, d, q) ajustables
        model_fit = model.fit()

        # Prédire les prochains mois
        forecast = model_fit.forecast(steps=forecast_steps)
        predictions["DURATION OF STAY"] = forecast  # Stocke les prévisions pour le nombre de lits

        # 📊 Affichage des prévisions
        fig, ax = plt.subplots(figsize=(10, 4))  # Créer une figure et un axe
        ax.plot(df_monthly["DURATION OF STAY"], label="Données réelles")
        ax.axvline(df_monthly.index[-1], color='r', linestyle='--', label="Début de la prévision")

        # Corrigé: Utilisation des parenthèses pour la compréhension de générateur
        ax.scatter([df_monthly.index[-1] + pd.DateOffset(months=i) for i in range(1, forecast_steps + 1)],
                   [forecast[i-1] for i in range(1, forecast_steps + 1)],
                   color='red', label="Prévisions")
        
        ax.set_title("Prévision ARIMA pour le nombre de lits d'hôpital")
        ax.legend()
        st.pyplot(fig)  # Passer la figure à st.pyplot()

    except Exception as e:
        print(f"❌ Erreur pour le nombre de lits d'hôpital : {e}")
        return predictions

    return predictions

# Main Streamlit interface
def main():
    st.title("Prévisions des lits d'hôpital")

    # Charger et préparer les données
    df_monthly = load_and_prepare_data()

    # Agréger les données par mois (utiliser "ME" au lieu de "M")
    df_monthly = df_monthly.resample("ME").sum()

    # Sélectionner le nombre de mois à prédire
    forecast_steps = st.slider("Sélectionne le nombre de mois à prédire", min_value=1, max_value=12, value=1)

    # Prévoir avec ARIMA
    predictions = forecast_arima(df_monthly, forecast_steps)

    # Afficher les prévisions
    if predictions:
        st.subheader("Prévisions pour le nombre de lits d'hôpital")
        for i, prediction in enumerate(predictions["DURATION OF STAY"]):
            prediction_date = df_monthly.index[-1] + pd.DateOffset(months=i+1)
            st.write(f"📅 {prediction_date.strftime('%b %Y')} : {prediction:.2f} lits")

if __name__ == "__main__":
    main()
