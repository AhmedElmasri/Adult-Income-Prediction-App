"""Project-level configuration for the Adult Income K-Fold pipeline."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "AdultIncome.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

OUTPUT_DIR = BASE_DIR / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"
CV_RESULTS_PATH = METRICS_DIR / "cross_validation_results.csv"
EDA_SUMMARY_PATH = METRICS_DIR / "eda_summary.csv"

TARGET_COLUMN = "IncomeClass"
POSITIVE_LABEL = " >50K"
NEGATIVE_LABEL = " <=50K"

NUMERICAL_FEATURES = ["age", "hours per week"]
CATEGORICAL_FEATURES = ["wc", "education", "marital status", "race", "gender"]
FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

CV_FOLDS = 10
RANDOM_STATE = 42
