"""Prediction utilities and command-line inference for the Adult Income model."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.config import FEATURE_COLUMNS, MODEL_PATH


def load_model_bundle(model_path: str | Path = MODEL_PATH) -> dict:
    """Load the saved model bundle."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model was not found at {model_path}. Run `python -m src.train` first."
        )
    return joblib.load(model_path)


def predict_income(input_data: pd.DataFrame, model_path: str | Path = MODEL_PATH) -> pd.DataFrame:
    """Predict income class for one or more rows of input features."""
    bundle = load_model_bundle(model_path)
    pipeline = bundle["pipeline"]

    missing_columns = set(FEATURE_COLUMNS).difference(input_data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input data is missing required feature columns: {missing}")

    predictions = pipeline.predict(input_data[FEATURE_COLUMNS])
    labels = [">50K" if pred == 1 else "<=50K" for pred in predictions]

    output = input_data.copy()
    output["predicted_income_class"] = labels
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run income class prediction.")
    parser.add_argument("--age", type=int, required=True)
    parser.add_argument("--wc", type=str, required=True, help="Work class value, e.g. ' Private'.")
    parser.add_argument("--education", type=str, required=True, help="Education value, e.g. ' Bachelors'.")
    parser.add_argument("--marital-status", type=str, required=True, dest="marital_status")
    parser.add_argument("--race", type=str, required=True, help="Race value, e.g. ' White'.")
    parser.add_argument("--gender", type=str, required=True, help="Gender value, e.g. ' Male'.")
    parser.add_argument("--hours-per-week", type=int, required=True, dest="hours_per_week")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_df = pd.DataFrame(
        [
            {
                "age": args.age,
                "wc": args.wc,
                "education": args.education,
                "marital status": args.marital_status,
                "race": args.race,
                "gender": args.gender,
                "hours per week": args.hours_per_week,
            }
        ]
    )
    prediction_df = predict_income(input_df)
    print(prediction_df.to_string(index=False))


if __name__ == "__main__":
    main()
