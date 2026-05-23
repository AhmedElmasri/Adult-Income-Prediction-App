"""Train and evaluate the Adult Income classifiers with K-Fold cross-validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from src.config import (
    CV_FOLDS,
    CV_RESULTS_PATH,
    FEATURE_COLUMNS,
    METADATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    POSITIVE_LABEL,
    RANDOM_STATE,
)
from src.data_loader import load_dataset
from src.eda import generate_eda_outputs
from src.models import get_models
from src.preprocessing import build_preprocessor, split_features_target


def build_pipeline(estimator) -> Pipeline:
    """Create a preprocessing + model pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    model_names: Iterable[str] | None = None,
    cv: int = CV_FOLDS,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, dict]:
    """Run K-Fold cross-validation for the selected models."""
    all_models = get_models()
    selected_names = list(model_names) if model_names else list(all_models.keys())

    unknown_models = sorted(set(selected_names).difference(all_models.keys()))
    if unknown_models:
        raise ValueError(f"Unknown model name(s): {', '.join(unknown_models)}")

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
    }

    rows = []
    fitted_pipelines = {}

    for model_name in selected_names:
        print(f"Running {cv}-fold cross-validation for: {model_name}")
        pipeline = build_pipeline(all_models[model_name])

        cv_results = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            return_train_score=True,
            n_jobs=n_jobs,
            scoring=scoring,
        )

        rows.append(
            {
                "model": model_name,
                "mean_train_accuracy": cv_results["train_accuracy"].mean(),
                "std_train_accuracy": cv_results["train_accuracy"].std(),
                "mean_test_accuracy": cv_results["test_accuracy"].mean(),
                "std_test_accuracy": cv_results["test_accuracy"].std(),
                "mean_train_precision": cv_results["train_precision"].mean(),
                "std_train_precision": cv_results["train_precision"].std(),
                "mean_test_precision": cv_results["test_precision"].mean(),
                "std_test_precision": cv_results["test_precision"].std(),
                "mean_train_recall": cv_results["train_recall"].mean(),
                "std_train_recall": cv_results["train_recall"].std(),
                "mean_test_recall": cv_results["test_recall"].mean(),
                "std_test_recall": cv_results["test_recall"].std(),
                "mean_fit_time": cv_results["fit_time"].mean(),
                "mean_score_time": cv_results["score_time"].mean(),
            }
        )

        fitted_pipelines[model_name] = pipeline

    results_df = pd.DataFrame(rows).sort_values(
        by="mean_test_accuracy",
        ascending=False,
    )
    return results_df, fitted_pipelines


def train_final_model(best_model_name: str, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Fit the best model on the complete dataset and return the trained pipeline."""
    models = get_models()
    pipeline = build_pipeline(models[best_model_name])
    pipeline.fit(X, y)
    return pipeline


def save_artifacts(
    pipeline: Pipeline,
    results_df: pd.DataFrame,
    best_model_name: str,
    cv: int,
    n_jobs: int,
    selected_models: list[str] | None,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
    metrics_path: Path = CV_RESULTS_PATH,
) -> None:
    """Save trained model, metadata, and cross-validation metrics."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(metrics_path, index=False)

    bundle = {
        "pipeline": pipeline,
        "best_model_name": best_model_name,
        "feature_columns": FEATURE_COLUMNS,
        "positive_label": POSITIVE_LABEL,
        "target_mapping": {"<=50K": 0, ">50K": 1},
    }
    joblib.dump(bundle, model_path)

    metadata = {
        "best_model_name": best_model_name,
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "random_state": RANDOM_STATE,
        "cv_folds": cv,
        "n_jobs": n_jobs,
        "models_evaluated": selected_models if selected_models else list(results_df["model"]),
        "selection_metric": "mean_test_accuracy",
        "cross_validation_results": results_df.to_dict(orient="records"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=4), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Adult Income ML pipeline.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to CSV dataset.")
    parser.add_argument("--cv", type=int, default=CV_FOLDS, help="Number of cross-validation folds.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=list(get_models().keys()),
        help="Specific models to run. Defaults to all notebook models.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for cross-validation. Use -1 to use all processors.",
    )
    parser.add_argument(
        "--generate-eda",
        action="store_true",
        help="Generate simple EDA summaries and figures before training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.generate_eda:
        generate_eda_outputs(args.data_path)

    df = load_dataset(args.data_path) if args.data_path else load_dataset()
    X, y = split_features_target(df)

    results_df, _ = evaluate_models(X, y, model_names=args.models, cv=args.cv, n_jobs=args.n_jobs)
    best_model_name = results_df.iloc[0]["model"]
    print("\nCross-validation summary:")
    print(results_df.to_string(index=False))
    print(f"\nBest model based on mean_test_accuracy: {best_model_name}")

    best_pipeline = train_final_model(best_model_name, X, y)
    save_artifacts(best_pipeline, results_df, best_model_name, args.cv, args.n_jobs, args.models)
    print(f"\nSaved trained model to: {MODEL_PATH}")
    print(f"Saved metrics to: {CV_RESULTS_PATH}")


if __name__ == "__main__":
    main()
