"""Preparation des variables pour les modeles."""

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.utils.helpers import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    get_logger
)

logger = get_logger(__name__)

TARGET_COLUMN = "Mental_Health_Risk"

NUMERIC_FEATURES = [
    "Age",
    "Sleep_Hours",
    "Study_Hours_Per_Week",
    "GPA",
    "Social_Media_Hours",
    "Screen_Time_Hours",
    "Financial_Stress",
    "Family_Support",
    "Attendance_Rate"
]

NOMINAL_FEATURES = [
    "Gender",
    "Part_Time_Job",
    "Previous_Mental_Health_History"
]

ORDINAL_FEATURES = [
    "Year_of_Study",
    "Physical_Activity",
    "Diet_Quality"
]

ORDINAL_CATEGORIES = [
    [1, 2, 3, 4, 5],
    ["Low", "Moderate", "High"],
    ["Poor", "Average", "Good"]
]


def build_preprocessor():
    """
    Construire le pipeline de preprocessing.
    """

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    nominal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    ordinal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=ORDINAL_CATEGORIES,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("nominal", nominal_transformer, NOMINAL_FEATURES),
            ("ordinal", ordinal_transformer, ORDINAL_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor


def prepare_data(df, test_size=0.2, random_state=42):
    """
    Séparer les features et la cible,
    encoder la cible et sauvegarder les données.
    """

    df = df.copy()

    required_features = (
        NUMERIC_FEATURES
        + NOMINAL_FEATURES
        + ORDINAL_FEATURES
    )

    missing_columns = [
        column for column in required_features + [TARGET_COLUMN]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes : {missing_columns}"
        )

    X = df[required_features]
    y = df[TARGET_COLUMN]

    # Encodage de la variable cible
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    # Séparation Training / Testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )

    # Sauvegarde de l'encodeur
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        target_encoder,
        MODELS_DIR / "target_encoder.joblib"
    )

    # Sauvegarde des données séparées
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    split_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

    joblib.dump(
        split_data,
        PROCESSED_DATA_DIR / "data_splits.joblib"
    )

    logger.info("Séparation des données terminée.")
    logger.info("X_train : %s", X_train.shape)
    logger.info("X_test : %s", X_test.shape)

    return split_data
