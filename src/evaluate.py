"""Read and display the saved cross-validation results."""
from pathlib import Path

import pandas as pd

from src.config import CV_RESULTS_PATH


def load_results(path: str | Path = CV_RESULTS_PATH) -> pd.DataFrame:
    """Load saved cross-validation results."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Metrics file was not found at {path}. Run `python -m src.train` first."
        )
    return pd.read_csv(path)


def main() -> None:
    results_df = load_results()
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
