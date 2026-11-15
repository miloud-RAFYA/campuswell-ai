"""Chargement des données sources."""

from pathlib import Path

import pandas as pd

from src.utils.helpers import RAW_DATA_DIR, get_logger

logger = get_logger(__name__)


def load_dataset(file_path=None):
    """Charger le dataset depuis le répertoire raw."""

    if file_path is None:
        file_path = RAW_DATA_DIR / "student_mental_health.csv"

    file_path = Path(file_path)   # accepte aussi une chaîne

    logger.info("Chargement du dataset : %s", file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {file_path}")

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("Le dataset est vide.")

    logger.info("Dimensions du dataset : %s", df.shape)

    return df


if __name__ == "__main__":
    data = load_dataset()
    print(data.head())