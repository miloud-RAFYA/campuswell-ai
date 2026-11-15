"""Page Streamlit d'exploration des donnees."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CampusWell AI | EDA",
    page_icon="📊",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_dataset
from src.data.clean_data import clean_dataset


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_mental_health.csv"

TARGET_COLUMN = "Mental_Health_Risk"


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background-color: #0b1120;
        }

        .page-title {
            color: #ffffff;
            font-size: 2.4rem;
            font-weight: 800;
        }

        .page-description {
            color: #94a3b8;
            font-size: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_raw_data():
    """Charger les données brutes."""
    return load_dataset(str(DATA_PATH))


@st.cache_data
def clean_data(data):
    """Nettoyer les données chargées."""
    return clean_dataset(data)


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    '<div class="page-title">📊 Exploratory Data Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="page-description">
        Exploration, compréhension et analyse descriptive
        des données relatives au bien-être des étudiants.
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# LOAD AND CLEAN DATA
# ============================================================

try:
    if not DATA_PATH.exists():
        st.error(f"Dataset introuvable : {DATA_PATH}")
        st.stop()

    df_raw = load_raw_data()
    df_clean = clean_data(df_raw)

except Exception as error:
    st.error(f"Erreur lors du traitement des données : {error}")
    st.stop()


# ============================================================
# 1. DATASET OVERVIEW
# ============================================================

st.subheader("1. Vue générale du dataset")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Lignes initiales",
    f"{df_raw.shape[0]:,}"
)

col2.metric(
    "Colonnes initiales",
    df_raw.shape[1]
)

col3.metric(
    "Valeurs manquantes",
    int(df_raw.isna().sum().sum())
)

col4.metric(
    "Doublons",
    int(df_raw.duplicated().sum())
)

# Comparaison avant et après nettoyage
st.markdown("#### Comparaison avant et après nettoyage")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Lignes initiales",
    df_raw.shape[0]
)

col2.metric(
    "Lignes après nettoyage",
    df_clean.shape[0]
)

col3.metric(
    "Lignes supprimées",
    df_raw.shape[0] - df_clean.shape[0]
)

with st.expander("Structure des données brutes"):

    schema = pd.DataFrame({
        "Colonne": df_raw.columns,
        "Type": df_raw.dtypes.astype(str).values,
        "Valeurs manquantes": df_raw.isna().sum().values,
        "Valeurs uniques": df_raw.nunique().values
    })

    st.dataframe(
        schema,
        use_container_width=True
    )


# ============================================================
# 2. DATA PREVIEW
# ============================================================

st.subheader("2. Aperçu des données")

tab1, tab2 = st.tabs([
    "Données brutes",
    "Données nettoyées"
])

with tab1:
    st.dataframe(
        df_raw.head(10),
        use_container_width=True
    )

with tab2:
    st.dataframe(
        df_clean.head(10),
        use_container_width=True
    )


# ============================================================
# 3. TARGET ANALYSIS
# ============================================================

st.subheader("3. Distribution de la variable cible")

if TARGET_COLUMN in df_clean.columns:

    target_counts = (
        df_clean[TARGET_COLUMN]
        .value_counts()
        .reindex(["Low", "Medium", "High"])
        .fillna(0)
        .astype(int)
        .rename_axis("Niveau de risque")
        .reset_index(name="Effectif")
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            target_counts,
            x="Niveau de risque",
            y="Effectif",
            color="Niveau de risque",
            text="Effectif",
            title="Répartition des niveaux de risque"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.pie(
            target_counts,
            names="Niveau de risque",
            values="Effectif",
            hole=0.45,
            title="Proportion des classes"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

else:
    st.warning("La variable cible est introuvable.")


# ============================================================
# 4. NUMERICAL FEATURES
# ============================================================

st.subheader("4. Analyse des variables numériques")

numeric_columns = df_clean.select_dtypes(
    include="number"
).columns.tolist()

# Exclure les identifiants
numeric_columns = [
    column for column in numeric_columns
    if column != "Student_ID"
]

if numeric_columns:

    selected_numeric = st.selectbox(
        "Sélectionner une variable numérique",
        numeric_columns
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df_clean,
            x=selected_numeric,
            marginal="box",
            nbins=30,
            title=f"Distribution de {selected_numeric}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        if TARGET_COLUMN in df_clean.columns:

            fig = px.box(
                df_clean,
                x=TARGET_COLUMN,
                y=selected_numeric,
                color=TARGET_COLUMN,
                category_orders={
                    TARGET_COLUMN: ["Low", "Medium", "High"]
                },
                title=f"{selected_numeric} selon le niveau de risque"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with st.expander("Statistiques descriptives"):

        st.dataframe(
            df_clean[numeric_columns].describe().T,
            use_container_width=True
        )


# ============================================================
# 5. CATEGORICAL FEATURES
# ============================================================

st.subheader("5. Analyse des variables catégorielles")

categorical_columns = df_clean.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

# Exclure la variable cible
categorical_columns = [
    column for column in categorical_columns
    if column != TARGET_COLUMN
]

if categorical_columns:

    selected_category = st.selectbox(
        "Sélectionner une variable catégorielle",
        categorical_columns
    )

    category_counts = (
        df_clean[selected_category]
        .fillna("Non renseigné")
        .value_counts()
        .rename_axis("Catégorie")
        .reset_index(name="Effectif")
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            category_counts,
            x="Catégorie",
            y="Effectif",
            color="Catégorie",
            text="Effectif",
            title=f"Distribution de {selected_category}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        if TARGET_COLUMN in df_clean.columns:

            fig = px.histogram(
                df_clean,
                x=selected_category,
                color=TARGET_COLUMN,
                barmode="group",
                category_orders={
                    TARGET_COLUMN: ["Low", "Medium", "High"]
                },
                title=f"{selected_category} selon le niveau de risque"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# 6. CORRELATION MATRIX
# ============================================================

st.subheader("6. Matrice de corrélation")

correlation_data = df_clean.select_dtypes(
    include="number"
).drop(
    columns=["Student_ID"],
    errors="ignore"
)

correlation = correlation_data.corr()

if not correlation.empty:

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Corrélation entre les variables numériques"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# 7. CONCLUSION
# ============================================================

st.subheader("7. Résumé de l'analyse")

st.markdown(
    f"""
    - Nombre de lignes initiales : **{df_raw.shape[0]}**
    - Nombre de lignes après nettoyage : **{df_clean.shape[0]}**
    - Nombre de variables : **{df_clean.shape[1]}**
    - Variable cible : **{TARGET_COLUMN}**
    - Nombre de classes : **{df_clean[TARGET_COLUMN].nunique()}**
    """
)

st.info(
    """
    Cette analyse est réalisée dans un cadre pédagogique.
    Les résultats ne constituent pas un diagnostic médical.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CampusWell AI | Exploratory Data Analysis | Projet pédagogique"
)
