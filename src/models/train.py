
"""Entraînement du modèle Logistic Regression."""

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)

from src.features.preprocessing import build_preprocessor
from src.utils.helpers import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    METRICS_DIR,
    get_logger
)

logger = get_logger(__name__)


def train_models():
    """
    Entraîner et évaluer le modèle Logistic Regression.
    """

    # Vérifier l'existence du fichier de séparation
    split_path = PROCESSED_DATA_DIR / "data_splits.joblib"

    if not split_path.exists():
        raise FileNotFoundError(
            "Fichier data_splits.joblib introuvable."
        )

    # Charger les données d'entraînement
    split_data = joblib.load(split_path)

    X_train = split_data["X_train"]
    y_train = split_data["y_train"]

    # Définir le modèle
    classifier = LogisticRegression(
        max_iter=2000,
        random_state=42
    )

    # Construire le Pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier)
        ]
    )

    # Configurer la validation croisée
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "f1_weighted": "f1_weighted",
        "f1_macro": "f1_macro"
    }

    logger.info(
        "Évaluation de Logistic Regression avec Cross-Validation."
    )

    # Évaluer le modèle sur les 5 folds
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    # Entraîner le modèle sur toutes les données Training
    logger.info("Entraînement final de Logistic Regression.")

    pipeline.fit(X_train, y_train)

    # Préparer les résultats de Cross-Validation
    results_df = pd.DataFrame([{
        "Model": "Logistic Regression",
        "Accuracy Mean": scores["test_accuracy"].mean(),
        "Accuracy Std": scores["test_accuracy"].std(),
        "F1 Weighted Mean": scores["test_f1_weighted"].mean(),
        "F1 Macro Mean": scores["test_f1_macro"].mean()
    }])

    # Créer les répertoires nécessaires
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Sauvegarder le modèle entraîné
    model_path = MODELS_DIR / "logistic_regression_pipeline.joblib"

    joblib.dump(pipeline, model_path)

    # Sauvegarder les résultats
    results_df.to_csv(
        METRICS_DIR / "model_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    logger.info(
        "Modèle sauvegardé : %s",
        model_path
    )

    logger.info(
        "Accuracy moyenne : %.4f",
        scores["test_accuracy"].mean()
    )

    logger.info(
        "F1 Weighted moyen : %.4f",
        scores["test_f1_weighted"].mean()
    )

    logger.info(
        "Entraînement terminé avec succès."
    )

    return results_df
