import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(layout="wide")

col1, space1, col2, space2, col3 = st.columns([1.5 , 1, 4, 1, 2])

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
def forecast_arima(df_monthly, forecast_steps, display=True):
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

        if display :
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
    with col2:
        st.title("Prévisions des lits d'hôpital")

        # Charger et préparer les données
        df_monthly = load_and_prepare_data()

        # Agréger les données par mois (utiliser "ME" au lieu de "M")
        df_monthly = df_monthly.resample("ME").sum()

        # Sélectionner le nombre de mois à prédire
        forecast_steps = st.slider("Sélectionne le nombre de mois à prédire", min_value=1, max_value=12, value=1)

        forecast_arima(df_monthly, forecast_steps)

    with col1:
        
        # Prévoir avec ARIMA pour tous les 12 mois
        predictions = forecast_arima(df_monthly, 12, display=False)

        st.markdown("<br><br>", unsafe_allow_html=True)  # Décalage esthétique
        
        # Nombre de lits disponibles
        lits_disponibles = st.number_input("Nombre de lits disponibles", min_value=1, step=1, value=100)
        mois_de_crise = 1

        if predictions:
            # 🔥 Vérification pour les 12 prochains mois
            alert_triggered = False
            for mois, prediction in enumerate(predictions["DURATION OF STAY"]):
                taux_occupation = (prediction / lits_disponibles) * 100
                mois_de_crise = mois
                if taux_occupation >= 90:
                    alert_triggered = True
                    break  # Dès qu'on trouve 1 mois avec +90%, on sort de la boucle

            if alert_triggered:
                st.warning(f"🚨 Alerte : Plus de **90%** des lits seront occupés au cours du {mois_de_crise}ème prochain mois !")
            else:
                st.success("✅ Prévision stable : Moins de **90%** des lits seront occupés au cours des 12 prochains mois.")

    # Afficher les prévisions et les ressources nécessaires
    with col3:
        infirmiers_par_5 = st.number_input("Nombre d'infirmiers nécessaires pour 5 patients", min_value=1, step=1, value=2)
        total_infirmiers = int(infirmiers_par_5 * predictions["DURATION OF STAY"][-1] / 5)
        st.markdown(f"**Nombre d'infirmiers nécessaires :** {total_infirmiers}", unsafe_allow_html=True)

        gants_steriles = st.number_input("Nombre de gants par infirmier", min_value=1, step=1, value=2)
        total_gants_steriles = int(infirmiers_par_5 * gants_steriles)
        st.markdown(f"**Nombre de gants stériles nécessaires :** {total_gants_steriles}", unsafe_allow_html=True)

        compresses_steriles = st.number_input("Nombre de compresses stériles par patient", min_value=1, step=1, value=2)
        total_compresses_steriles = int(compresses_steriles * predictions["DURATION OF STAY"][-1])
        st.markdown(f"**Nombre de compresses stériles nécessaires :** {total_compresses_steriles}", unsafe_allow_html=True)

        seringues_et_aiguilles = st.number_input("Nombre de seringues et aiguilles par patient", min_value=1, step=1, value=2)
        total_seringues_et_aiguilles = int(seringues_et_aiguilles * predictions["DURATION OF STAY"][-1])
        st.markdown(f"**Nombre de seringues et aiguilles nécessaires :** {total_seringues_et_aiguilles}", unsafe_allow_html=True)

        st.divider()

        if predictions:
            st.subheader("Prévisions pour le nombre de lits d'hôpital")
            for i, prediction in enumerate(predictions["DURATION OF STAY"]):
                prediction_date = df_monthly.index[-1] + pd.DateOffset(months=i+1)
                st.write(f"📅 {prediction_date.strftime('%b %Y')} : {prediction:.2f} lits")

if __name__ == "__main__":
    main()
