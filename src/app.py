"""Streamlit app for Adult Income prediction."""
from __future__ import annotations

import streamlit as st
import pandas as pd

from src.config import CATEGORICAL_FEATURES, MODEL_PATH, NUMERICAL_FEATURES
from src.data_loader import load_dataset
from src.predict import load_model_bundle, predict_income


st.set_page_config(
    page_title="Adult Income Classifier",
    page_icon="💼",
    layout="centered",
)


@st.cache_data
def get_reference_data() -> pd.DataFrame:
    """Load dataset once to populate input options."""
    return load_dataset()


@st.cache_resource
def get_model_bundle() -> dict:
    """Load the trained model once."""
    return load_model_bundle(MODEL_PATH)


def format_option(value: str) -> str:
    """Display categorical values without leading spaces while keeping raw values for inference."""
    return str(value).strip()


def main() -> None:
    st.title("Adult Income Prediction App")
    st.write(
        "This app predicts whether a person's income class is `>50K` or `<=50K` "
        "using the ML pipeline converted from the original K-Fold Cross Validation notebook."
    )

    df = get_reference_data()

    with st.sidebar:
        st.header("Dataset Preview")
        st.write(f"Rows: **{df.shape[0]}**")
        st.write(f"Columns: **{df.shape[1]}**")
        st.dataframe(df.head(10), use_container_width=True)

    if not MODEL_PATH.exists():
        st.warning(
            "No trained model was found. Please run `python -m src.train` from the project root first."
        )
        return

    st.subheader("Enter Input Features")

    with st.form("prediction_form"):
        age = st.number_input(
            "Age",
            min_value=int(df["age"].min()),
            max_value=int(df["age"].max()),
            value=int(df["age"].median()),
        )
        hours_per_week = st.number_input(
            "Hours per week",
            min_value=int(df["hours per week"].min()),
            max_value=int(df["hours per week"].max()),
            value=int(df["hours per week"].median()),
        )

        categorical_inputs = {}
        for column in CATEGORICAL_FEATURES:
            options = sorted(df[column].dropna().unique().tolist(), key=lambda x: str(x).strip())
            categorical_inputs[column] = st.selectbox(
                label=column.title(),
                options=options,
                format_func=format_option,
            )

        submitted = st.form_submit_button("Predict Income Class")

    if submitted:
        input_df = pd.DataFrame(
            [
                {
                    "age": age,
                    "hours per week": hours_per_week,
                    **categorical_inputs,
                }
            ]
        )

        prediction_df = predict_income(input_df)
        predicted_label = prediction_df.loc[0, "predicted_income_class"]

        st.success(f"Predicted income class: **{predicted_label}**")
        st.subheader("Input Used for Prediction")
        st.dataframe(prediction_df, use_container_width=True)

        bundle = get_model_bundle()
        st.caption(f"Current saved model: {bundle.get('best_model_name', 'unknown')}")


if __name__ == "__main__":
    main()
