"""
This is a boilerplate pipeline 'training'
generated using Kedro 1.2.0
"""

import warnings
from collections.abc import Callable
from typing import Any, TypedDict

import numpy as np
import optuna
import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedKFold

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_model(
    instance: BaseEstimator,
    training_set: tuple[np.ndarray, np.ndarray],
    params: dict[str, Any] | None = None,
) -> BaseEstimator:
    params = params or {}
    model = clone(instance)
    model.set_params(**params)
    model.fit(*training_set)
    return model


def optimize_hyp(
    X: pd.DataFrame,
    y: pd.Series,
    max_evals: int = 40,
) -> dict:
    def objective(trial):
        params = {
            "objective": "binary",
            "verbose": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 1),
            "n_estimators": trial.suggest_int("num_iterations", 100, 1000, step=20),
            "max_depth": trial.suggest_int("max_depth", 4, 12, step=6),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128, step=10),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1),
            "subsample": trial.suggest_float("subsample", 0.5, 1),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 20, step=10),
            "reg_alpha": trial.suggest_categorical("reg_alpha", [0, 1e-1, 1, 2, 5, 10]),
            "reg_lambda": trial.suggest_categorical("reg_lambda", [0, 1e-1, 1, 2, 5, 10]),
        }

        rep_kfold = RepeatedKFold(n_splits=4, n_repeats=1)
        scores_test = []
        for train_I, test_I in rep_kfold.split(X):
            X_fold_train = X.iloc[train_I, :]
            y_fold_train = y.iloc[train_I].values.flatten()
            X_fold_test = X.iloc[test_I, :]
            y_fold_test = y.iloc[test_I].values.flatten()
            model = LGBMClassifier(**params)
            model.fit(X_fold_train, y_fold_train)
            scores_test.append(f1_score(y_fold_test, model.predict(X_fold_test)))

        return np.mean(scores_test)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=max_evals)
    return study.best_params


def auto_ml(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_evals: int = 40,
) -> dict[str, BaseEstimator]:
    X = pd.concat((X_train, X_test))
    y = pd.concat((y_train, y_test))

    best_params = optimize_hyp(X, y, max_evals=max_evals)

    model = LGBMClassifier(**best_params, objective="binary", verbose=-1)
    model.fit(X_train, y_train)

    score = f1_score(y_test, model.predict(X_test))

    return dict(model=model, score=score)
