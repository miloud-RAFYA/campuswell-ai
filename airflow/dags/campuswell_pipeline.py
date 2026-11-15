"""DAG Airflow du pipeline CampusWell."""

from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

from airflow.sdk import dag, task


# ============================================================
# Configuration du projet
# ============================================================

PROJECT_ROOT = next(
    (
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "src").is_dir()
    ),
    None,
)

if PROJECT_ROOT is None:
    raise RuntimeError("Impossible de localiser le dossier src.")

sys.path.insert(0, str(PROJECT_ROOT))

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

LOADED_DATA_PATH = DATA_PROCESSED / "loaded_students.csv"
CLEANED_DATA_PATH = DATA_PROCESSED / "cleaned_students.csv"


# ============================================================
# Imports des fonctions du projet
# ============================================================

from src.data.load_data import load_dataset
from src.data.clean_data import clean_dataset
from src.features.preprocessing import prepare_data
from src.models.train import train_models
from src.models.evaluate import evaluate_model


# ============================================================
# Définition du DAG
# ============================================================

@dag(
    dag_id="campuswell_ai_pipeline",
    description="Pipeline ML de prédiction du risque de santé mentale étudiant",
    start_date=datetime(2026, 11, 1),
    schedule=None,
    catchup=False,
    tags=["campuswell", "machine-learning", "students"],
)
def campuswell_pipeline():

    # --------------------------------------------------------
    # Task 1 : Chargement des données
    # --------------------------------------------------------

    @task
    def load_data():
        df = load_dataset()

        LOADED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(LOADED_DATA_PATH, index=False)

        print(f"Données chargées : {df.shape}")
        return str(LOADED_DATA_PATH)

    # --------------------------------------------------------
    # Task 2 : Nettoyage des données
    # --------------------------------------------------------

    @task
    def clean_data(input_path: str):
        df = pd.read_csv(input_path)

        df_cleaned = clean_dataset(df)
        df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)

        print(f"Données nettoyées : {df_cleaned.shape}")
        return str(CLEANED_DATA_PATH)

    # --------------------------------------------------------
    # Task 3 : Prétraitement des données
    # --------------------------------------------------------

    @task
    def preprocess_data(input_path: str):
        df = pd.read_csv(input_path)

        prepare_data(df)

        print("Prétraitement terminé.")
        return "data_splits.joblib"

    # --------------------------------------------------------
    # Task 4 : Entraînement des modèles
    # --------------------------------------------------------

    @task
    def train_models_task():
        train_models()
        print("Entraînement des modèles terminé.")

    # --------------------------------------------------------
    # Task 5 : Évaluation finale
    # --------------------------------------------------------

    @task
    def evaluate_model_task():
        evaluate_model()
        print("Évaluation finale terminée.")

    # --------------------------------------------------------
    # Dépendances entre les tâches
    # --------------------------------------------------------

    loaded = load_data()
    cleaned = clean_data(loaded)
    preprocessed = preprocess_data(cleaned)
    trained = train_models_task()
    evaluated = evaluate_model_task()

    preprocessed >> trained
    trained  >> evaluated


campuswell_pipeline()
