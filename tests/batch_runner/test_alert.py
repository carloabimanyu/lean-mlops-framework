import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services" / "batch_runner"))


def test_send_alert_calls_smtp(monkeypatch):
    monkeypatch.setenv("GMAIL_SENDER", "sender@gmail.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "fake-password")
    monkeypatch.setenv("GMAIL_RECIPIENT", "recipient@gmail.com")

    with patch("smtplib.SMTP_SSL") as mock_smtp:
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__ = lambda s: mock_server
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        import alert
        import importlib
        importlib.reload(alert)
        alert.send_alert(drift_score=0.75, report_path="/reports/drift_report_test.html")

        mock_smtp.assert_called_once_with("smtp.gmail.com", 465)


def test_send_alert_missing_env_raises(monkeypatch):
    monkeypatch.delenv("GMAIL_SENDER", raising=False)
    monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)

    import importlib
    import alert
    importlib.reload(alert)

    import pytest
    with pytest.raises(ValueError, match="GMAIL_"):
        alert.send_alert(0.8, "/reports/test.html")
