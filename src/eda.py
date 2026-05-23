"""Small EDA utilities mirroring the exploratory checks from the notebook."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import EDA_SUMMARY_PATH, FIGURES_DIR, TARGET_COLUMN
from src.data_loader import load_dataset


def generate_eda_outputs(data_path: str | Path | None = None) -> None:
    """Generate simple dataset summaries and figures."""
    df = load_dataset(data_path) if data_path else load_dataset()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    EDA_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "missing_values": df.isna().sum().values,
            "unique_values": df.nunique().values,
        }
    )
    summary.to_csv(EDA_SUMMARY_PATH, index=False)

    df[TARGET_COLUMN].value_counts().to_csv(
        EDA_SUMMARY_PATH.parent / "income_class_counts.csv",
        header=["count"],
    )

    df["marital status"].value_counts().to_csv(
        EDA_SUMMARY_PATH.parent / "marital_status_counts.csv",
        header=["count"],
    )

    plt.figure(figsize=(7, 5))
    df[TARGET_COLUMN].value_counts().plot(kind="bar")
    plt.title("Income Class Distribution")
    plt.xlabel("Income Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "income_class_distribution.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    df["marital status"].value_counts().plot(kind="bar")
    plt.title("Marital Status Distribution")
    plt.xlabel("Marital Status")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "marital_status_distribution.png")
    plt.close()


if __name__ == "__main__":
    generate_eda_outputs()
    print("EDA outputs were saved under outputs/metrics and outputs/figures.")
