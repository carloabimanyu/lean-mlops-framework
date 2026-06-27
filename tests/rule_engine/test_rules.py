def test_high_risk():
    from models.rule_engine.rules import evaluate
    result = evaluate({"age": 17, "income": 500, "credit_score": 400})
    assert result["risk_level"] == "high"
    assert len(result["triggered_rules"]) > 0


def test_low_risk():
    from models.rule_engine.rules import evaluate
    result = evaluate({"age": 35, "income": 80000, "credit_score": 750})
    assert result["risk_level"] == "low"
    assert result["triggered_rules"] == []


def test_medium_risk():
    from models.rule_engine.rules import evaluate
    result = evaluate({"age": 25, "income": 25000, "credit_score": 450})
    assert result["risk_level"] == "medium"
