"""Page Streamlit de demonstration des predictions."""

from pathlib import Path

import dill
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "best_logistic_regression_pipeline.joblib"
ENCODER_PATH = PROJECT_ROOT / "models" / "target_encoder.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_mental_health.csv"

st.set_page_config(
    page_title="Démonstration de prédiction | CampusWell AI",
    page_icon="🎯",
    layout="wide"
)


# ============================================================
# Chargement des ressources
# ============================================================

@st.cache_resource
def load_resources():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"Encodeur introuvable : {ENCODER_PATH}")

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    return model, encoder


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset introuvable : {DATA_PATH}")

    return pd.read_csv(DATA_PATH)


# ============================================================
# Interface
# ============================================================

st.title("🎯 Démonstration de prédiction")
st.markdown(
    """
    Cette interface permet de tester le modèle de Machine Learning
    en fournissant les informations d'un étudiant.
    """
)

st.warning(
    """
    **Avertissement :** cette démonstration est uniquement pédagogique.
    Le modèle ne réalise aucun diagnostic médical et ne remplace pas
    l'avis d'un professionnel de santé.
    """
)

try:
    model, target_encoder = load_resources()
    df = load_dataset()

except Exception as error:
    st.error(f"Erreur lors du chargement des ressources : {error}")
    st.stop()


# ============================================================
# Préparation des valeurs disponibles
# ============================================================

def get_options(column):
    return sorted(
        df[column].dropna().astype(str).str.strip().unique().tolist()
    )


# ============================================================
# Formulaire de saisie
# ============================================================

st.subheader("Informations de l'étudiant")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Informations générales")

        age = st.number_input(
            "Âge",
            min_value=int(df["Age"].min()),
            max_value=int(df["Age"].max()),
            value=int(df["Age"].median())
        )

        gender = st.selectbox(
            "Genre",
            options=get_options("Gender")
        )

        year_of_study = st.selectbox(
            "Année d'étude",
            options=sorted(df["Year_of_Study"].dropna().unique())
        )

        gpa = st.number_input(
            "GPA",
            min_value=float(df["GPA"].min()),
            max_value=float(df["GPA"].max()),
            value=float(df["GPA"].median()),
            step=0.1
        )

        attendance_rate = st.slider(
            "Taux de présence (%)",
            min_value=float(df["Attendance_Rate"].min()),
            max_value=float(df["Attendance_Rate"].max()),
            value=float(df["Attendance_Rate"].median())
        )

        part_time_job = st.selectbox(
            "Emploi à temps partiel",
            options=get_options("Part_Time_Job")
        )

        previous_history = st.selectbox(
            "Antécédents de santé mentale",
            options=get_options("Previous_Mental_Health_History")
        )

    with col2:
        st.markdown("#### Habitudes et bien-être")

        sleep_hours = st.slider(
            "Heures de sommeil par jour",
            min_value=float(df["Sleep_Hours"].min()),
            max_value=float(df["Sleep_Hours"].max()),
            value=float(df["Sleep_Hours"].median()),
            step=0.5
        )

        study_hours = st.slider(
            "Heures d'étude par semaine",
            min_value=float(df["Study_Hours_Per_Week"].min()),
            max_value=float(df["Study_Hours_Per_Week"].max()),
            value=float(df["Study_Hours_Per_Week"].median()),
            step=1.0
        )

        social_media_hours = st.slider(
            "Heures sur les réseaux sociaux par jour",
            min_value=float(df["Social_Media_Hours"].min()),
            max_value=float(df["Social_Media_Hours"].max()),
            value=float(df["Social_Media_Hours"].median()),
            step=0.5
        )

        screen_time_hours = st.slider(
            "Temps d'écran par jour",
            min_value=float(df["Screen_Time_Hours"].min()),
            max_value=float(df["Screen_Time_Hours"].max()),
            value=float(df["Screen_Time_Hours"].median()),
            step=0.5
        )

        financial_stress = st.slider(
            "Stress financier",
            min_value=float(df["Financial_Stress"].min()),
            max_value=float(df["Financial_Stress"].max()),
            value=float(df["Financial_Stress"].median()),
            step=1.0
        )

        family_support = st.slider(
            "Soutien familial",
            min_value=float(df["Family_Support"].min()),
            max_value=float(df["Family_Support"].max()),
            value=float(df["Family_Support"].median()),
            step=1.0
        )

        physical_activity = st.selectbox(
            "Activité physique",
            options=sorted(df["Physical_Activity"].dropna().unique())
        )

        diet_quality = st.selectbox(
            "Qualité de l'alimentation",
            options=get_options("Diet_Quality")
        )

    submitted = st.form_submit_button(
        "Prédire le niveau de risque",
        use_container_width=True
    )


# ============================================================
# Prédiction
# ============================================================

if submitted:

    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Year_of_Study": year_of_study,
        "Sleep_Hours": sleep_hours,
        "Study_Hours_Per_Week": study_hours,
        "GPA": gpa,
        "Social_Media_Hours": social_media_hours,
        "Screen_Time_Hours": screen_time_hours,
        "Physical_Activity": physical_activity,
        "Financial_Stress": financial_stress,
        "Family_Support": family_support,
        "Attendance_Rate": attendance_rate,
        "Part_Time_Job": part_time_job,
        "Diet_Quality": diet_quality,
        "Previous_Mental_Health_History": previous_history
    }])

    try:
        prediction = model.predict(input_data)
        predicted_class = target_encoder.inverse_transform(prediction)[0]

        st.divider()
        st.subheader("Résultat de la prédiction")

        if predicted_class == "Low":
            st.success("Niveau de risque prédit : LOW")

        elif predicted_class == "Medium":
            st.warning("Niveau de risque prédit : MEDIUM")

        else:
            st.error("Niveau de risque prédit : HIGH")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]

            probability_df = pd.DataFrame({
                "Classe": target_encoder.classes_,
                "Probabilité (%)": probabilities * 100
            })

            st.markdown("#### Probabilités par classe")

            st.bar_chart(
                probability_df.set_index("Classe"),
                y="Probabilité (%)"
            )

            st.dataframe(
                probability_df.style.format({
                    "Probabilité (%)": "{:.2f}%"
                }),
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "Les probabilités représentent les sorties du modèle "
            "et ne constituent pas une mesure clinique du risque."
        )

    except Exception as error:
        st.error(f"Erreur pendant la prédiction : {error}")
