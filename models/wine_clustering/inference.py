import os
import mlflow.sklearn
from pydantic import BaseModel

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MODEL_NAME = "wine-clustering"

_model = None


class InputSchema(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315: float
    proline: float


def load_model():
    global _model
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")


def predict(data: dict) -> dict:
    if _model is None:
        load_model()
    validated = InputSchema(**data)
    features = [[
        validated.alcohol, validated.malic_acid, validated.ash,
        validated.alcalinity_of_ash, validated.magnesium,
        validated.total_phenols, validated.flavanoids,
        validated.nonflavanoid_phenols, validated.proanthocyanins,
        validated.color_intensity, validated.hue,
        validated.od280_od315, validated.proline,
    ]]
    cluster = int(_model.predict(features)[0])
    return {"cluster": cluster}
