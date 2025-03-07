import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff

# Charger les données
file_path = "data/new_hospital_admissions.csv"
df = pd.read_csv(file_path)

# Convertir la date
df["Date d'Entrée"] = pd.to_datetime(df["Date d'Entrée"])
df["Mois"] = df["Date d'Entrée"].dt.to_period("M")

# Remplacement des valeurs pour Type Admission
df["Type Admission"] = df["Type Admission"].replace({"E": "Urgente", "O": "Non Urgente"})

# Sidebar pour navigation
st.sidebar.title("Filtres d'analyse")
selected_start_date = st.sidebar.date_input("Date de début", df["Date d'Entrée"].min())
selected_end_date = st.sidebar.date_input("Date de fin", df["Date d'Entrée"].max())
selected_admission_type = st.sidebar.multiselect("Type d'admission", df["Type Admission"].unique(), default=df["Type Admission"].unique())
selected_pathologie = st.sidebar.multiselect("Pathologie", df["Pathologie"].unique(), default=df["Pathologie"].unique())

# Filtrage des données
df_filtered = df[(df["Date d'Entrée"] >= pd.to_datetime(selected_start_date)) &
                 (df["Date d'Entrée"] <= pd.to_datetime(selected_end_date))]

if selected_admission_type:
    df_filtered = df_filtered[df_filtered["Type Admission"].isin(selected_admission_type)]
if selected_pathologie:
    df_filtered = df_filtered[df_filtered["Pathologie"].isin(selected_pathologie)]

# Admissions mensuelles
st.subheader("Évolution des admissions hospitalières")
admissions_mensuelles = df_filtered.groupby("Mois").size()
if not admissions_mensuelles.empty:
    plt.figure(figsize=(10, 5))
    admissions_mensuelles.plot(kind='bar', color='skyblue')
    plt.xlabel("Mois")
    plt.ylabel("Nombre d'admissions")
    plt.xticks(rotation=45)
    st.pyplot(plt)
st.write(f"Période analysée : {selected_start_date} à {selected_end_date}. Nombre total d'admissions : {admissions_mensuelles.sum()}.")

# Répartition par type d'admission
st.subheader("Proportion des admissions urgentes et non-urgentes")
admission_counts = df_filtered["Type Admission"].value_counts()
if not admission_counts.empty:
    st.bar_chart(admission_counts)
    st.write(f"Type d'admission le plus fréquent : {admission_counts.idxmax()} ({admission_counts.max()} cas).")

# Distribution des âges par sexe
st.subheader("Distribution des âges par sexe")
if not df_filtered.empty:
    age_sex_distribution = df_filtered.groupby("Sexe")["Âge"].describe()[["mean", "min", "max"]]
    st.table(age_sex_distribution)
    st.write(f"L'âge moyen des patients est de {df_filtered['Âge'].mean():.1f} ans, avec un minimum de {df_filtered['Âge'].min()} ans et un maximum de {df_filtered['Âge'].max()} ans.")

# Durée moyenne d'hospitalisation par pathologie
st.subheader("Durée moyenne d'hospitalisation par pathologie")
duree_par_pathologie = df_filtered.groupby("Pathologie")["Durée Hospitalisation (jours)"].mean().sort_values(ascending=False)
if not duree_par_pathologie.empty:
    st.bar_chart(duree_par_pathologie)
    st.write(f"Pathologie avec la durée d'hospitalisation la plus longue : {duree_par_pathologie.idxmax()} ({duree_par_pathologie.max():.1f} jours).")

# Pathologies les plus fréquentes
st.subheader("Pathologies les plus fréquentes")
top_pathologies = df_filtered["Pathologie"].value_counts().nlargest(10)
if not top_pathologies.empty:
    st.bar_chart(top_pathologies)
    st.write(f"Pathologie la plus fréquente : {top_pathologies.idxmax()} ({top_pathologies.max()} cas).")

# Visualisation avancée avec Plotly
st.subheader("Distribution des âges des patients hospitalisés")
if not df_filtered["Âge"].dropna().empty:
    fig = ff.create_distplot([df_filtered["Âge"].dropna()], ["Âge"], show_hist=True, show_rug=False)
    st.plotly_chart(fig)
    st.write("Ce graphique représente la distribution des âges des patients hospitalisés, mettant en évidence les groupes d'âge les plus concernés par les admissions.")
