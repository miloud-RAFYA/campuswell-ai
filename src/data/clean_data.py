"""Nettoyage et validation des donnees."""

import pandas as pd

from src.utils.helpers import get_logger

logger = get_logger(__name__)

TARGET_COLUMN = "Mental_Health_Risk"

EXPECTED_CLASSES = ["Low", "Medium", "High"]


def clean_dataset(df):
    """
    Nettoyer le dataset avant le preprocessing.
    """

    df = df.copy()

    # Vérification de la variable cible
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Colonne cible introuvable : {TARGET_COLUMN}"
        )

    initial_rows = len(df)

    # Suppression des doublons
    df = df.drop_duplicates()

    logger.info(
        "Doublons supprimés : %d",
        initial_rows - len(df)
    )

    # Suppression des observations sans cible
    df = df.dropna(subset=[TARGET_COLUMN])

    # Nettoyage des valeurs textuelles
    categorical_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in categorical_columns:
        df[column] = df[column].str.strip()

    # Vérification des classes de la cible
    unexpected_classes = (
        set(df[TARGET_COLUMN].unique())
        - set(EXPECTED_CLASSES)
    )

    if unexpected_classes:
        raise ValueError(
            f"Classes inattendues : {unexpected_classes}"
        )

    # Suppression de la colonne identifiant
    if "Student_ID" in df.columns:
        df = df.drop(columns=["Student_ID"])

    logger.info("Dimensions après nettoyage : %s", df.shape)

    return df
