"""Data loading and validation utilities."""
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import FEATURE_COLUMNS, RAW_DATA_PATH, TARGET_COLUMN


def load_dataset(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the Adult Income dataset from CSV and validate the required columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset was not found at: {path}")

    df = pd.read_csv(path)
    validate_columns(df.columns)
    return df


def validate_columns(columns: Iterable[str]) -> None:
    """Ensure the raw dataset has the columns expected by the original notebook."""
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns.difference(set(columns))
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")
