def evaluate(data: dict) -> dict:
    triggered = []

    if data.get("age", 99) < 18:
        triggered.append("underage")
    if data.get("income", 0) < 1000:
        triggered.append("low_income")
    if data.get("credit_score", 999) < 500:
        triggered.append("poor_credit")

    if len(triggered) >= 2:
        risk_level = "high"
    elif len(triggered) == 1:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {"risk_level": risk_level, "triggered_rules": triggered}
