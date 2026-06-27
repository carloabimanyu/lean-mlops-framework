import os
import importlib
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException

# Make models/ importable when running inside container (bind-mounted at /models)
sys.path.insert(0, "/")
# Make models/ importable when running tests locally
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
except IndexError:
    pass

import mlflow

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

app = FastAPI(title="MLOps Inference API")

MODEL_MODULE_MAP = {
    "iris-classifier": "models.iris_classifier.inference",
    "wine-clustering": "models.wine_clustering.inference",
    "rule-engine": "models.rule_engine.inference",
}


def get_inference_module(model_name: str):
    module_path = MODEL_MODULE_MAP.get(model_name)
    if not module_path:
        return None
    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError:
        return None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def list_models():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    registered = client.search_registered_models()
    return [m.name for m in registered]


@app.post("/predict/{model_name}")
def predict(model_name: str, data: dict):
    module = get_inference_module(model_name)
    if module is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    try:
        return module.predict(data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
