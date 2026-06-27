import os
import pytest

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"

def test_predict_returns_cluster():
    from models.wine_clustering.inference import predict, load_model
    load_model()
    sample = {
        "alcohol": 13.2, "malic_acid": 1.78, "ash": 2.14,
        "alcalinity_of_ash": 11.2, "magnesium": 100.0,
        "total_phenols": 2.65, "flavanoids": 2.76,
        "nonflavanoid_phenols": 0.26, "proanthocyanins": 1.28,
        "color_intensity": 4.38, "hue": 1.05,
        "od280_od315": 3.40, "proline": 1050.0
    }
    result = predict(sample)
    assert isinstance(result, dict)
    assert "cluster" in result
    assert result["cluster"] in [0, 1, 2]
