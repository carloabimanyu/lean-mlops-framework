from pydantic import BaseModel
from models.rule_engine.rules import evaluate

MODEL_NAME = "rule-engine"


class InputSchema(BaseModel):
    age: int
    income: float
    credit_score: int


def predict(data: dict) -> dict:
    validated = InputSchema(**data)
    return evaluate(validated.model_dump())
