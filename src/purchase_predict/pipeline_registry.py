from __future__ import annotations

from kedro.pipeline import Pipeline

from purchase_predict.pipelines import loading, processing, training


def register_pipelines() -> dict[str, Pipeline]:
    processing_pipeline = processing.create_pipeline()
    training_pipeline = training.create_pipeline()
    loading_pipeline = loading.create_pipeline()

    return {
        "processing": processing_pipeline,
        "training": training_pipeline,
        "loading": loading_pipeline,
        "__default__": processing_pipeline + training_pipeline,
    }
