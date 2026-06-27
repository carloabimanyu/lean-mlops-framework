import os
import pytest

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"

def test_predict_returns_dict():
    from models.iris_classifier.inference import predict, load_model
    load_model()
    result = predict({
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert isinstance(result, dict)
    assert "prediction" in result
    assert "label" in result

def test_predict_valid_class():
    from models.iris_classifier.inference import predict, load_model
    load_model()
    result = predict({
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert result["prediction"] in [0, 1, 2]
    assert result["label"] in ["setosa", "versicolor", "virginica"]
