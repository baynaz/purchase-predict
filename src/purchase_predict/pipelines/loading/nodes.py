"""
This is a boilerplate pipeline 'loading'
generated using Kedro 1.2.0
"""

import glob
import os
import tempfile

import pandas as pd
from google.cloud import storage


def load_csv_from_bucket(project: str, bucket_path: str) -> pd.DataFrame:
    """
    Loads multiple CSV files from a GCS bucket (as generated from Spark).
    """
    storage_client = storage.Client(project=project)

    bucket_name = bucket_path.split("/", 1)[0]
    folder = bucket_path.split("/", 1)[1]
    '''bucket_name = bucket_path.split("/", maxsplit=1)[0]
    folder = "/".join(bucket_path.split("/")[1:]) + "/part-"d'''

    # dossier temporaire portable
    tmp_dir = tempfile.gettempdir()

    # téléchargement des fichiers CSV
    for blob in storage_client.list_blobs(bucket_name, prefix=folder):
        filename = os.path.basename(blob.name)

        if filename.endswith(".csv"):
            local_path = os.path.join(tmp_dir, filename)
            blob.download_to_filename(local_path)

    # récupération des fichiers téléchargés
    all_files = glob.glob(os.path.join(tmp_dir, "*.csv"))

    li = []
    for file in all_files:
        df = pd.read_csv(file, index_col=None, sep=",")
        li.append(df)

    if not li:
        raise ValueError("No CSV files found in bucket path")

    df = pd.concat(li, axis=0, ignore_index=True)
    return df
