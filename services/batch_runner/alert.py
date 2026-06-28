import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

log = logging.getLogger(__name__)


def send_alert(drift_score: float, report_path: str) -> None:
    sender = os.getenv("GMAIL_SENDER")
    password = os.getenv("GMAIL_APP_PASSWORD")
    recipient = os.getenv("GMAIL_RECIPIENT")

    if not sender or not password:
        raise ValueError("GMAIL_SENDER and GMAIL_APP_PASSWORD must be set")

    recipient = recipient or sender

    subject = f"[MLOps Alert] Data drift detected — score: {drift_score:.3f}"
    body = f"""
Data drift has been detected in the wine_clustering model.

Drift score: {drift_score:.3f}
Report: {report_path}

Please review the Evidently report and consider retraining the model.
    """.strip()

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, recipient, msg.as_string())

    log.info(f"Alert email sent to {recipient}")
