import numpy as np
import pandas as pd

from purchase_predict.pipelines.processing.nodes import encode_features, split_dataset

BALANCE_THRESHOLD = 0.1
MIN_SAMPLES = 5000


def test_encode_features(dataset_not_encoded):
    encoded = encode_features(dataset_not_encoded)
    df = encoded["features"]
    assert isinstance(df, pd.DataFrame)
    assert df["purchased"].isin([0, 1]).all()
    for col in df.columns:
        assert pd.api.types.is_numeric_dtype(df.dtypes[col])
    assert df.shape[0] > MIN_SAMPLES
    assert (df["purchased"].value_counts() / df.shape[0] > BALANCE_THRESHOLD).all()


def test_split_dataset(dataset_encoded, test_ratio):
    X_train, y_train, X_test, y_test = split_dataset(
        dataset_encoded, test_ratio
    ).values()
    # Checks both sets size
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[0] + X_test.shape[0] == dataset_encoded.shape[0]
    # Note that train_test_split of scikit-learn use np.ceil for test split
    # https://github.com/scikit-learn/scikit-learn/blob/42aff4e2edd8e8887478f6ff1628f27de97be6a3/sklearn/model_selection/_split.py#L1797
    assert np.ceil(dataset_encoded.shape[0] * test_ratio) == X_test.shape[0]
