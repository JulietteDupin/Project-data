import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
from datetime import datetime

st.set_page_config(layout="wide")

col1, space1, col2, space2, col3 = st.columns([1.5, 1, 4, 1, 2])

# 📌 Charger et préparer les données médicales avec le nouveau dataset
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv('Admissions Hospitalières Complètes.csv', encoding='utf-8')
    df.columns = df.columns.str.strip()

    # S'assurer que les colonnes sont correctement nommées
    selected_columns = ["Date d'Entrée", "Durée Hospitalisation (jours)"]
    df = df[selected_columns]
    
    # Convertir les dates au format datetime
    df["Date d'Entrée"] = pd.to_datetime(df["Date d'Entrée"], format="%Y-%m-%d", errors='coerce')
    df.set_index("Date d'Entrée", inplace=True)

    # Assurer que la durée de l'hospitalisation est un nombre et remplacer les erreurs par 0
    df["Durée Hospitalisation (jours)"] = pd.to_numeric(df["Durée Hospitalisation (jours)"], errors='coerce').fillna(0)

    # Agréger les données par mois pour les prévisions
    df_monthly = df.resample("M").sum()  # Résumer par mois en ajoutant la durée totale d'hospitalisation
    return df_monthly

# 📌 Charger les données utilisateur depuis CSV
def load_user_data():
    file_name = "user_inputs.csv"
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        return df.tail(1).iloc[0]  # Charger la dernière ligne
    else:
        return None

# 📌 Entraînement du modèle ARIMA
@st.cache_data
def forecast_arima(df_monthly, forecast_steps, display=True):
    predictions = {}
    try:
        model = SARIMAX(df_monthly["Durée Hospitalisation (jours)"], order=(0, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_steps)
        predictions["Durée Hospitalisation (jours)"] = forecast

        if display:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_monthly["Durée Hospitalisation (jours)"], label="Données réelles")
            ax.axvline(df_monthly.index[-1], color='r', linestyle='--', label="Début prévision")
            ax.scatter([df_monthly.index[-1] + pd.DateOffset(months=i) for i in range(1, forecast_steps + 1)],
                       forecast, color='red', label="Prévisions")
            ax.set_title("Prévision ARIMA pour la durée d'hospitalisation")
            ax.legend()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Erreur : {e}")

    return predictions

# 📌 Enregistrer les données utilisateur
def save_user_data(lits_disponibles, infirmiers, gants, compresses, seringues):
    file_name = "user_inputs.csv"
    data = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Lits disponibles": lits_disponibles,
        "Infirmiers": infirmiers,
        "Gants": gants,
        "Compresses": compresses,
        "Seringues": seringues
    }

    df = pd.DataFrame([data])

    if os.path.exists(file_name):
        df.to_csv(file_name, mode='a', header=False, index=False)
    else:
        df.to_csv(file_name, mode='w', index=False)

    st.success("✅ Données sauvegardées avec succès !")

# Main Streamlit interface
def main():
    # Charger les données mensuelles
    with col2:
        st.title("Prévisions de la durée d'hospitalisation")
        df_monthly = load_and_prepare_data()
        forecast_steps = st.slider("Nombre de mois à prédire", min_value=1, max_value=12, value=1)
        forecast_arima(df_monthly, forecast_steps)

    # Charger les données utilisateur
    user_data = load_user_data()
    lits_disponibles_default = 100
    if user_data is not None:
        lits_disponibles_default = int(user_data["Lits disponibles"])
    
    with col1:
        predictions = forecast_arima(df_monthly, 12, display=False)
        lits_disponibles = st.number_input("Nombre de lits disponibles", min_value=1, value=lits_disponibles_default)
        mois_crise = 1
        alert_triggered = False

        if predictions:
            for i, prediction in enumerate(predictions["Durée Hospitalisation (jours)"]):
                taux = (prediction / lits_disponibles) * 100
                mois_crise = i + 1
                if taux >= 90:
                    alert_triggered = True
                    break

            if alert_triggered:
                st.warning(f"🚨 Alerte : Plus de 90% des lits seront occupés au mois {mois_crise} !")
            else:
                st.success("✅ Prévision stable pour les 12 prochains mois")

    with col3:
        infirmiers_default = 2
        gants_default = 2
        compresses_default = 2
        seringues_default = 2
    
        if user_data is not None:
            infirmiers_default = int(user_data["Infirmiers"])
            gants_default = int(user_data["Gants"])
            compresses_default = int(user_data["Compresses"])
            seringues_default = int(user_data["Seringues"])
        else:
            infirmiers_default, gants_default, compresses_default, seringues_default = 2, 2, 2, 2
    
        infirmiers = st.number_input("Infirmiers pour 5 patients", min_value=1, value=infirmiers_default)
        gants = st.number_input("Gants par infirmier", min_value=1, value=gants_default)
        compresses = st.number_input("Compresses par patient", min_value=1, value=compresses_default)
        seringues = st.number_input("Seringues par patient", min_value=1, value=seringues_default)
    
        total_infirmiers = int(infirmiers * forecast_steps / 5)
        total_gants = int(total_infirmiers * gants)
        total_compresses = int(forecast_steps * compresses)
        total_seringues = int(forecast_steps * seringues)
    
        st.markdown(f"**Infirmiers nécessaires :** {total_infirmiers}")
        st.markdown(f"**Gants nécessaires :** {total_gants}")
        st.markdown(f"**Compresses nécessaires :** {total_compresses}")
        st.markdown(f"**Seringues nécessaires :** {total_seringues}")
    
        if st.button("Enregistrer les données"):
            save_user_data(lits_disponibles, infirmiers, gants, compresses, seringues)

if __name__ == "__main__":
    main()
