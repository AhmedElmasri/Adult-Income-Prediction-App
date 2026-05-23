# Adult Income K-Fold Cross Validation Pipeline

## Project Overview

This project is a production-style refactoring of the original Jupyter Notebook: `00-K-Fold Cross Validation.ipynb`.

The original notebook performs the following workflow:

1. Loads the `AdultIncome.csv` dataset.
2. Inspects the dataset using basic exploratory checks.
3. Encodes categorical features using one-hot encoding with `drop_first=True`.
4. Splits the dataset into features and the target income class.
5. Compares three machine learning classifiers using 10-fold cross-validation:
   - Decision Tree Classifier
   - Random Forest Classifier
   - Support Vector Classifier
6. Evaluates models using accuracy, precision, and recall.

This refactored version keeps the same main ML logic while organizing the work into reusable Python modules and adding a Streamlit app for inference.

## Dataset Description

The dataset file is located at:

```text
data/raw/AdultIncome.csv
```

The dataset contains 19,787 rows and 8 columns.

### Input Features

| Column | Description |
|---|---|
| `age` | Person age |
| `wc` | Work class |
| `education` | Education level |
| `marital status` | Marital status |
| `race` | Race category |
| `gender` | Gender |
| `hours per week` | Weekly working hours |

### Target Column

| Column | Description |
|---|---|
| `IncomeClass` | Income class: `<=50K` or `>50K` |

In the original notebook, `pd.get_dummies(df, drop_first=True)` creates a target column named `IncomeClass_ >50K`. In this production version, the target is encoded as:

```text
IncomeClass == " >50K" -> 1
otherwise              -> 0
```

## Project Structure

```text
adult_income_kfold_pipeline/
│
├── data/
│   └── raw/
│       └── AdultIncome.csv
│
├── notebooks/
│   └── 00-K-Fold Cross Validation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── evaluate.py
│   ├── models.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
│
├── models/
│   ├── best_model.joblib
│   └── model_metadata.json
│
├── outputs/
│   ├── figures/
│   └── metrics/
│       ├── cross_validation_results.csv
│       ├── eda_summary.csv
│       ├── income_class_counts.csv
│       └── marital_status_counts.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation Steps

From the project root folder, create and activate a virtual environment:

### Windows PowerShell

```bash
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run the Pipeline

### Fast Verified Run

The included saved model and metrics were generated with this faster command so the package is immediately usable:

```bash
python -m src.train --generate-eda --cv 3 --models decision_tree random_forest
```

The SVC model from the original notebook is still available in `src/models.py`. Because default kernel SVC can be slow on this dataset, run it when you specifically need the complete notebook-style comparison.


Run the full training pipeline using all models and 10-fold cross-validation:

```bash
python -m src.train --generate-eda
```

Use all CPU cores for cross-validation if desired:

```bash
python -m src.train --generate-eda --n-jobs -1
```

Run only selected models:

```bash
python -m src.train --models decision_tree random_forest
```

Change the number of folds:

```bash
python -m src.train --cv 5
```

## How to View Saved Evaluation Results

After training, view the saved cross-validation table:

```bash
python -m src.evaluate
```

The metrics are saved to:

```text
outputs/metrics/cross_validation_results.csv
```

## How to Run Prediction from Command Line

Train the model first:

```bash
python -m src.train
```

Then run prediction:

```bash
python -m src.predict ^
  --age 38 ^
  --wc " Private" ^
  --education " HS-grad" ^
  --marital-status " Divorced" ^
  --race " White" ^
  --gender " Male" ^
  --hours-per-week 40
```

For macOS / Linux, use backslashes instead of `^`:

```bash
python -m src.predict \
  --age 38 \
  --wc " Private" \
  --education " HS-grad" \
  --marital-status " Divorced" \
  --race " White" \
  --gender " Male" \
  --hours-per-week 40
```

## How to Run the Streamlit App

After training and saving the model, run:

```bash
streamlit run src/app.py
```

The app allows the user to enter feature values and predicts whether the income class is:

```text
<=50K
```

or

```text
>50K
```

## Expected Outputs

After running the pipeline, the following artifacts are generated:

```text
models/best_model.joblib
models/model_metadata.json
outputs/metrics/cross_validation_results.csv
outputs/metrics/eda_summary.csv
outputs/metrics/income_class_counts.csv
outputs/metrics/marital_status_counts.csv
outputs/figures/income_class_distribution.png
outputs/figures/marital_status_distribution.png
```

## Notes on Refactoring

- The notebook used `pd.get_dummies(drop_first=True)` directly on the full dataframe.
- The refactored project uses `OneHotEncoder(drop="first", handle_unknown="ignore")` inside a scikit-learn `Pipeline`.
- This keeps the same encoding idea while making the workflow safer and reusable for inference.
- The original model choices are preserved: Decision Tree, Random Forest, and SVC.
- The best model is selected using the highest mean cross-validation test accuracy.
