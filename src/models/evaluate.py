
"""Évaluation du modèle."""

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.utils.helpers import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    get_logger
)

logger = get_logger(__name__)


def evaluate_model():
    """
    Évaluer le modèle Logistic Regression sur le jeu de test.
    """

    # Chemins des fichiers
    split_path = PROCESSED_DATA_DIR / "data_splits.joblib"
    model_path = MODELS_DIR / "logistic_regression_pipeline.joblib"
    encoder_path = MODELS_DIR / "target_encoder.joblib"

    # Chargement des données et du modèle
    split_data = joblib.load(split_path)
    model = joblib.load(model_path)
    target_encoder = joblib.load(encoder_path)

    X_test = split_data["X_test"]
    y_test = split_data["y_test"]

    logger.info("Début de l'évaluation du modèle.")

    # Prédictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Noms et indices des classes
    class_names = target_encoder.classes_
    labels = list(range(len(class_names)))

    # Calcul des métriques
    metrics = {
        "Model": "Logistic Regression",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision Weighted": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "Recall Weighted": recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "F1 Weighted": f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "F1 Macro": f1_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "ROC-AUC Weighted": roc_auc_score(
            y_test,
            y_prob,
            labels=labels,
            multi_class="ovr",
            average="weighted"
        )
    }

    # Création des dossiers de sortie
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Sauvegarde des métriques globales
    metrics_df = pd.DataFrame([metrics])

    metrics_df.to_csv(
        METRICS_DIR / "final_evaluation.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Rapport de classification
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(
        METRICS_DIR / "classification_report.csv",
        encoding="utf-8-sig"
    )

    # Matrice de confusion
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    )

    cm_df.index.name = "Classe réelle"
    cm_df.columns.name = "Classe prédite"

    cm_df.to_csv(
        METRICS_DIR / "confusion_matrix_logistic_regression.csv",
        encoding="utf-8-sig"
    )

    # Visualisation de la matrice de confusion
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title("Matrice de confusion - Logistic Regression")
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe réelle")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "confusion_matrix_logistic_regression.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    logger.info("Évaluation terminée avec succès.")
    logger.info("Accuracy : %.4f", metrics["Accuracy"])
    logger.info("F1 Weighted : %.4f", metrics["F1 Weighted"])

    return metrics
