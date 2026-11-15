# CampusWell AI

Structure initiale du projet d'analyse de donnees et de machine learning.

## Structure

- `data/raw/` et `data/processed/` : donnees sources et preparees.
- `notebooks/` : exploration, preprocessing, entrainement et evaluation.
- `src/` : code Python reutilisable pour les donnees, variables et modeles.
- `airflow/dags/` : orchestration du pipeline.
- `app/` : application Streamlit et ses pages.
- `models/` : artefacts de modeles entraines.
- `reports/` : figures et metriques produites.
- `tests/` : tests automatises.

Les scripts et notebooks sont des squelettes a completer. Le modele
`campuswell_pipeline.joblib` sera genere apres l'entrainement et n'est pas
fourni dans cette structure initiale.

## Installation

```bash
python -m venv .venv
```

Activez l'environnement virtuel, puis installez les dependances :

```bash
pip install -r requirements.txt
```
