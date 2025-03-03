import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(layout="wide")

col1, space1, col2, space2, col3 = st.columns([1.5, 1, 4, 1, 2])

# 📌 Pré-traitement des données médicales
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv('Admissions Hospitalières Nettoyées 2021-2025.csv', encoding='utf-8', delimiter=',')
    
    df.columns = df.columns.str.strip()
    selected_columns = ["Date d'Entrée", "Durée Hospitalisation (jours)"]
    df = df[selected_columns]

    df["Date d'Entrée"] = pd.to_datetime(df["Date d'Entrée"], errors='coerce')
    df["Durée Hospitalisation (jours)"] = pd.to_numeric(df["Durée Hospitalisation (jours)"], errors='coerce').fillna(0)

    nombre_patients_par_mois = []
    for _, row in df.iterrows():
        start_date = row["Date d'Entrée"]
        duration = int(row["Durée Hospitalisation (jours)"])
        months_range = pd.date_range(start=start_date, periods=duration, freq='D')

        for month in months_range:
            nombre_patients_par_mois.append({'mois': month, 'patients': 1})

    df_patients = pd.DataFrame(nombre_patients_par_mois)
    df_monthly = df_patients.groupby('mois').sum()
    df_monthly_agg = df_monthly.resample('ME').sum()  # Agrégation mensuelle
    
    today = datetime.now().date()
    df_monthly_agg = df_monthly_agg[df_monthly_agg.index.date <= today]  # Garde toutes les données historiques jusqu'à aujourd'hui


    return df_monthly_agg

# 📌 Prévision SARIMA
@st.cache_data
def forecast_sarima(df, forecast_steps):
    try:
        model = SARIMAX(df, order=(0, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit()
        forecast = model_fit.get_forecast(steps=forecast_steps).predicted_mean
        forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='ME')[1:]

        return forecast, forecast_index
    except Exception as e:
        st.error(f"Erreur lors de la prévision : {e}")
        return None, None

# Interface principale
def main():
    with col2:
        st.title("Prévisions des Admissions Hospitalières")
        
        df_monthly_agg = load_and_prepare_data()

        st.subheader("Visualisation des Données Nettoyées")
        st.line_chart(df_monthly_agg)

        forecast_steps = st.slider("Nombre de mois à prédire", min_value=1, max_value=24, value=12)
        forecast, forecast_index = forecast_sarima(df_monthly_agg, forecast_steps)

        if forecast is not None:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_monthly_agg, label="Données Réelles", marker='o')
            ax.plot(forecast_index, forecast, label="Prévisions SARIMA", color='red', linestyle='dashed', marker='x')
            ax.set_title("Prévisions SARIMA avec saisonnalité annuelle")
            ax.set_xlabel("Mois")
            ax.set_ylabel("Nombre de patients")
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig)

    with col1:
        st.subheader("Taux d'Occupation des Lits")
        lits_disponibles = st.number_input("Nombre de lits disponibles", min_value=1, value=100)
        crise = False

        if forecast is not None:
            for i, pred in enumerate(forecast):
                taux_occupation = (pred / lits_disponibles) * 100
                if taux_occupation >= 90:
                    st.warning(f"🚨 Alerte : Plus de 90% des lits seront occupés au mois {i + 1}")
                    crise = True
                    break

            if not crise:
                st.success("✅ Prévision stable pour les 24 prochains mois")

    with col3:
        st.subheader("Consommation Matériel Médical")

        infirmiers = st.number_input("Nombre d'infirmiers par patient", min_value=1, value=2)
        gants = st.number_input("Gants par infirmier", min_value=1, value=2)
        compresses = st.number_input("Compresses par patient", min_value=1, value=2)
        seringues = st.number_input("Seringues par patient", min_value=1, value=2)

        if forecast is not None:
            total_infirmiers = int(infirmiers * forecast.sum())  # ✅ Correction ici !
            total_gants = total_infirmiers * gants
            total_compresses = int(forecast.sum() * compresses)
            total_seringues = int(forecast.sum() * seringues)

            st.markdown(f"**Infirmiers nécessaires :** {total_infirmiers}")
            st.markdown(f"**Gants nécessaires :** {total_gants}")
            st.markdown(f"**Compresses nécessaires :** {total_compresses}")
            st.markdown(f"**Seringues nécessaires :** {total_seringues}")

if __name__ == "__main__":
    main()
