"""Page Streamlit des performances du modele."""

"""Page Streamlit de présentation des performances des modèles."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CampusWell AI | Model Performance",
    page_icon="📈",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_PATH = PROJECT_ROOT / "reports"
METRICS_PATH = REPORTS_PATH / "metrics"
FIGURES_PATH = REPORTS_PATH / "figures"

FINAL_EVALUATION_PATH = METRICS_PATH / "final_evaluation.csv"
MODEL_COMPARISON_PATH = METRICS_PATH / "model_comparison.csv"
OPTIMIZATION_PATH = METRICS_PATH / "optimization_results.csv"

CLASSIFICATION_REPORT_PATH = (
    METRICS_PATH / "classification_report_logistic_regression.csv"
)

CONFUSION_MATRIX_PATH = (
    FIGURES_PATH / "confusion_matrix_logistic_regression.png"
)


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
def load_csv(path):
    """Charger un rapport CSV."""
    return pd.read_csv(path)


def load_report(path):
    """Charger un rapport si le fichier existe."""
    if not path.exists():
        return None

    return load_csv(str(path))


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    '<div class="page-title">📈 Model Performance</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="page-description">
        Évaluation, comparaison et analyse des performances
        des modèles de Machine Learning.
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# LOAD REPORTS
# ============================================================

try:
    final_evaluation = load_report(FINAL_EVALUATION_PATH)
    model_comparison = load_report(MODEL_COMPARISON_PATH)
    optimization_results = load_report(OPTIMIZATION_PATH)
    classification_report = load_report(CLASSIFICATION_REPORT_PATH)

except Exception as error:
    st.error(f"Erreur lors du chargement des rapports : {error}")
    st.stop()


# ============================================================
# 1. FINAL MODEL PERFORMANCE
# ============================================================

st.subheader("1. Performance du modèle final")

if final_evaluation is not None and not final_evaluation.empty:

    st.dataframe(
        final_evaluation,
        use_container_width=True,
        hide_index=True
    )

    # Sélection du modèle Logistic Regression
    logistic_rows = final_evaluation[
        final_evaluation["Model"].astype(str).str.contains(
            "Logistic Regression",
            case=False,
            na=False
        )
    ]

    if not logistic_rows.empty:

        best_model = logistic_rows.iloc[0]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Accuracy",
            f"{best_model['Accuracy']:.2%}"
        )

        col2.metric(
            "F1 Weighted",
            f"{best_model['F1 Weighted']:.2%}"
        )

        col3.metric(
            "ROC-AUC Weighted",
            f"{best_model['ROC-AUC Weighted']:.2%}"
        )

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Precision Weighted",
            f"{best_model['Precision Weighted']:.2%}"
        )

        col5.metric(
            "Recall Weighted",
            f"{best_model['Recall Weighted']:.2%}"
        )

        col6.metric(
            "F1 Macro",
            f"{best_model['F1 Macro']:.2%}"
        )

    else:
        st.warning(
            "Le modèle Logistic Regression est introuvable "
            "dans le rapport final."
        )

else:
    st.warning(
        "Le rapport final_evaluation.csv est introuvable ou vide."
    )


# ============================================================
# 2. MODEL COMPARISON
# ============================================================

st.divider()
st.subheader("2. Comparaison des modèles")

if model_comparison is not None and not model_comparison.empty:

    st.markdown("#### Résultats de la validation croisée")

    st.dataframe(
        model_comparison,
        use_container_width=True,
        hide_index=True
    )

    accuracy_column = "Accuracy Mean"
    f1_column = "F1 Weighted Mean"

    if accuracy_column in model_comparison.columns:

        fig = px.bar(
            model_comparison,
            x="Model",
            y=accuracy_column,
            text=accuracy_column,
            color="Model",
            title="Comparaison des modèles - Accuracy moyenne"
        )

        fig.update_layout(
            yaxis_title="Accuracy moyenne",
            xaxis_title="Modèle"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="model_comparison_accuracy"
        )

    if f1_column in model_comparison.columns:

        fig = px.bar(
            model_comparison,
            x="Model",
            y=f1_column,
            text=f1_column,
            color="Model",
            title="Comparaison des modèles - F1 Weighted moyen"
        )

        fig.update_layout(
            yaxis_title="F1 Weighted moyen",
            xaxis_title="Modèle"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="model_comparison_f1"
        )

else:
    st.warning(
        "Le rapport model_comparison.csv est introuvable ou vide."
    )


# ============================================================
# 3. HYPERPARAMETER OPTIMIZATION
# ============================================================

st.divider()
st.subheader("3. Optimisation des hyperparamètres")

if optimization_results is not None and not optimization_results.empty:

    st.markdown(
        """
        Les résultats suivants présentent les configurations
        testées pendant la recherche des meilleurs hyperparamètres.
        """
    )

    st.dataframe(
        optimization_results,
        use_container_width=True,
        hide_index=True
    )

    # Identifier la colonne du score moyen
    score_columns = [
        column for column in optimization_results.columns
        if "mean_test_score" in column.lower()
        or "accuracy mean" in column.lower()
        or "mean score" in column.lower()
    ]

    if score_columns:

        score_column = score_columns[0]

        best_index = optimization_results[score_column].idxmax()
        best_configuration = optimization_results.loc[best_index]

        st.success(
            f"Meilleur score moyen : "
            f"{best_configuration[score_column]:.4f}"
        )

        st.markdown("**Meilleure configuration :**")

        st.json(
            best_configuration.to_dict()
        )

    else:
        st.info(
            "Le fichier ne contient pas de colonne de score moyen "
            "identifiable automatiquement."
        )

else:
    st.warning(
        "Le rapport optimization_results.csv est introuvable ou vide."
    )


# ============================================================
# 4. CLASSIFICATION REPORT
# ============================================================

st.divider()
st.subheader("4. Classification Report")

if classification_report is not None and not classification_report.empty:

    st.markdown(
        """
        Ce rapport présente les performances du modèle
        pour chaque classe prédite.
        """
    )

    st.dataframe(
        classification_report,
        use_container_width=True,
        hide_index=True
    )

else:
    st.warning(
        "Le rapport classification_report_logistic_regression.csv "
        "est introuvable ou vide."
    )

# Rapport alternatif si le rapport spécifique n'existe pas
if classification_report is None:

    alternative_path = METRICS_PATH / "classification_report.csv"

    alternative_report = load_report(alternative_path)

    if alternative_report is not None and not alternative_report.empty:

        st.markdown("#### Rapport de classification disponible")

        st.dataframe(
            alternative_report,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# 5. CONFUSION MATRIX
# ============================================================

st.divider()
st.subheader("5. Matrice de confusion")

if CONFUSION_MATRIX_PATH.exists():

    st.markdown(
        """
        La matrice de confusion permet de comparer
        les classes réelles et les classes prédites.
        """
    )

    st.image(
        str(CONFUSION_MATRIX_PATH),
        caption="Matrice de confusion - Logistic Regression",
        use_container_width=True
    )

else:

    st.warning(
        "La matrice de confusion est introuvable : "
        f"{CONFUSION_MATRIX_PATH}"
    )


# ============================================================
# 6. CONCLUSION
# ============================================================

st.divider()
st.subheader("6. Conclusion")

st.markdown(
    """
    Le modèle Logistic Regression a obtenu les meilleures
    performances parmi les modèles évalués dans ce projet.

    L'évaluation repose sur plusieurs métriques :
    Accuracy, Precision, Recall, F1-score et ROC-AUC.

    Les résultats doivent être interprétés dans le contexte
    du dataset pédagogique utilisé.
    """
)

st.info(
    """
    CampusWell AI est un prototype pédagogique.
    Les prédictions ne constituent pas un diagnostic médical
    et ne doivent pas être utilisées pour prendre des décisions cliniques.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CampusWell AI | Model Performance | Projet pédagogique"
)
