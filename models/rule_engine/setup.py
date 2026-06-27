import os
import mlflow
import mlflow.pyfunc
from models.rule_engine.rules import evaluate

MODEL_NAME = "rule-engine"


class RuleEngineModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        if hasattr(model_input, "to_dict"):
            records = model_input.to_dict(orient="records")
        else:
            records = model_input
        return [evaluate(r) for r in records]


def setup() -> str:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    mlflow.set_experiment("rule-engine")

    rules_definition = {
        "underage": "age < 18",
        "low_income": "income < 1000",
        "poor_credit": "credit_score < 500",
        "risk_levels": {
            "high": "2+ rules triggered",
            "medium": "1 rule triggered",
            "low": "0 rules triggered",
        },
    }

    with mlflow.start_run() as run:
        mlflow.log_param("engine_type", "rule_based")
        mlflow.log_param("n_rules", 3)
        mlflow.log_metric("version", 1.0)
        mlflow.log_dict(rules_definition, "rules.json")

        mlflow.pyfunc.log_model(
            name="model",
            python_model=RuleEngineModel(),
        )

        mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME)

        return run.info.run_id


if __name__ == "__main__":
    run_id = setup()
    print(f"Rule engine registered. Run ID: {run_id}")
