import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import f1_score

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

X_train = pd.read_csv("data/05_model_input/X_train.csv")
X_test  = pd.read_csv("data/05_model_input/X_test.csv")
y_train = pd.read_csv("data/05_model_input/y_train.csv").values.flatten()
y_test  = pd.read_csv("data/05_model_input/y_test.csv").values.flatten()

hyp_params = {
    "num_leaves": 60,
    "min_child_samples": 10,
    "max_depth": 12,
    "n_estimators": 100,
    "learning_rate": 0.1,
}

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("purchase_predict")

with mlflow.start_run() as run:
    model = LGBMClassifier(**hyp_params, objective="binary", verbose=-1)
    model.fit(X_train, y_train)
    score = f1_score(y_test, model.predict(X_test))

    mlflow.log_params(hyp_params)
    mlflow.log_metric("f1", score)
    #print("Artifact URI:", mlflow.get_artifact_uri())

    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X_test.iloc[0:1].copy()

    mlflow.log_artifact("data/04_feature/transform_pipeline.pkl")
    mlflow.sklearn.log_model(
        model, "model",
        signature=signature,
        input_example=input_example
    )
