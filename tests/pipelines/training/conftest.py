import joblib
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def model():
    data = joblib.load("data/06_model/model.pkl")
    return data["model"]  # extraire le modèle du dictionnaire


@pytest.fixture(scope="module")
def X_test():
    return pd.read_csv("data/05_model_input/X_test.csv")


@pytest.fixture(scope="module")
def y_test():
    return pd.read_csv("data/05_model_input/y_test.csv").values.flatten()
