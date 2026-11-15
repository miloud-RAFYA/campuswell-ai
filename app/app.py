"""Application Streamlit CampusWell."""

from pathlib import Path

import streamlit as st


# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================

st.set_page_config(
    page_title="CampusWell AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# STYLE CSS
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background-color: #0b1120;
        }

        .main-title {
            font-size: 3rem;
            font-weight: 800;
            color: #ffffff;
            margin-bottom: 0.3rem;
        }

        .subtitle {
            font-size: 1.15rem;
            color: #94a3b8;
            margin-bottom: 2rem;
        }

        .section-title {
            color: #38bdf8;
            font-size: 1.5rem;
            font-weight: 700;
            margin-top: 1.5rem;
        }

        .info-card {
            background-color: #172033;
            padding: 1.4rem;
            border-radius: 12px;
            border: 1px solid #263449;
            min-height: 150px;
        }

        .info-card h3 {
            color: #38bdf8;
        }

        .info-card p {
            color: #cbd5e1;
        }

        .warning-box {
            background-color: #332b16;
            border-left: 4px solid #f59e0b;
            padding: 1rem;
            border-radius: 8px;
            color: #fef3c7;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🎓 CampusWell AI")

    st.caption("Student Wellbeing Prediction")

    st.divider()

    st.markdown("### Navigation")

    st.markdown(
        """
        Utilisez le menu **Pages** pour accéder aux différentes
        fonctionnalités de l'application.
        """
    )

    st.divider()

    st.caption("Projet pédagogique | Machine Learning")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">CampusWell AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Analyse et prédiction pédagogique du niveau de risque
        de santé mentale chez les étudiants.
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# PRESENTATION
# ============================================================

st.markdown(
    '<div class="section-title">Présentation du projet</div>',
    unsafe_allow_html=True
)

st.write(
    """
    CampusWell AI est un projet pédagogique de Machine Learning
    visant à analyser des facteurs liés au bien-être des étudiants
    et à prédire une catégorie de risque à partir de données
    structurées.
    """
)

st.write(
    """
    Le projet utilise plusieurs techniques de Data Science,
    notamment l'analyse exploratoire des données (EDA),
    le prétraitement, l'entraînement de modèles supervisés
    et l'évaluation des performances.
    """
)


# ============================================================
# OBJECTIFS
# ============================================================

st.markdown(
    '<div class="section-title">Objectifs principaux</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="info-card">
            <h3>📊 Analyse des données</h3>
            <p>
                Explorer les caractéristiques des étudiants
                et identifier les tendances présentes dans
                le dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="info-card">
            <h3>🤖 Machine Learning</h3>
            <p>
                Entraîner et comparer plusieurs modèles
                de classification supervisée.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="info-card">
            <h3>🎯 Prédiction</h3>
            <p>
                Présenter une démonstration interactive
                de prédiction à partir de données saisies.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MODELE RETENU
# ============================================================

st.markdown(
    '<div class="section-title">Modèle retenu</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Modèle sélectionné",
        value="Logistic Regression"
    )

with col2:
    st.metric(
        label="Accuracy de validation finale",
        value="75.50 %"
    )


# ============================================================
# AVERTISSEMENT ETHIQUE
# ============================================================

st.divider()

st.markdown(
    """
    <div class="warning-box">
        <strong>⚠️ Avertissement important</strong>
        <p>
            Cette application est un prototype pédagogique basé
            sur un dataset synthétique. Elle ne constitue pas
            un outil de diagnostic médical et ne doit pas être
            utilisée pour prendre des décisions médicales,
            académiques ou personnelles concernant les étudiants.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CampusWell AI | Projet pédagogique de Machine Learning | 2026"
)
