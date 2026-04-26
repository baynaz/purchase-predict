from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://127.0.0.1:5000")

# Lister tous les runs de l'expérience
experiment = client.get_experiment_by_name("purchase_predict")
runs = client.search_runs(experiment.experiment_id)

'''for run in runs:
    print(f"Run ID : {run.info.run_id}")
    print(f"F1     : {run.data.metrics.get('f1'):.4f}")
    print(f"Params : {run.data.params}")
    print("---")'''
