import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import PrecisionRecallDisplay, f1_score, precision_recall_curve

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Chargement des données
X_train = pd.read_csv("data/05_model_input/X_train.csv")
X_test = pd.read_csv("data/05_model_input/X_test.csv")
y_train = pd.read_csv("data/05_model_input/y_train.csv").values.flatten()
y_test = pd.read_csv("data/05_model_input/y_test.csv").values.flatten()

# Hyper-paramètres de base
hyp_params = {
    "num_leaves": 60,
    "min_child_samples": 10,
    "max_depth": 12,
    "n_estimators": 100,
    "learning_rate": 0.1,
}


def save_pr_curve(X, y, model):
    plt.figure(figsize=(16, 11))
    prec, recall, _ = precision_recall_curve(
        y, model.predict_proba(X)[:, 1], pos_label=1
    )
    PrecisionRecallDisplay(precision=prec, recall=recall).plot(ax=plt.gca())
    plt.title("PR Curve", fontsize=16)
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    plt.savefig("data/pr_curve.png")
    plt.close()


def train_model(params):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("purchase_predict")

    with mlflow.start_run():
        model = LGBMClassifier(**params, objective="binary", verbose=-1)
        model.fit(X_train, y_train)
        score = f1_score(y_test, model.predict(X_test))
        save_pr_curve(X_test, y_test, model)

        mlflow.log_params(params)
        mlflow.log_metric("f1", score)
        mlflow.log_artifact("data/pr_curve.png", artifact_path="plots")
        mlflow.log_artifact("data/04_feature/transform_pipeline.pkl")

        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_test.iloc[0:1].copy()
        mlflow.sklearn.log_model(
            model, "model", signature=signature, input_example=input_example
        )


# Lancer plusieurs expériences
train_model({**hyp_params, **{"n_estimators": 200, "learning_rate": 0.05}})
train_model({**hyp_params, **{"n_estimators": 500, "learning_rate": 0.025}})
train_model({**hyp_params, **{"n_estimators": 1000, "learning_rate": 0.01}})
