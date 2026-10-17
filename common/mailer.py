"""
Email for the student and tutor tools.

EMAIL_MODE=demo (default): nothing leaves the app. Mail goes through the mock university API
(uni/api.py): it lands in the sender's Sent box and in the inbox of any demo user it is addressed to,
private to the visitor's browser session. So the public demo is safe.
EMAIL_MODE=gmail: real SMTP/IMAP with GMAIL_ADDRESS and GMAIL_APP_PASSWORD from .env. Every send
still needs the user's approval in the chat first.
"""
import email
import imaplib
import os
import smtplib
from email.header import decode_header, make_header
from email.message import EmailMessage
from typing import Any

from dotenv import load_dotenv

from uni import api as uni_api

load_dotenv()


def email_mode() -> str:
    return os.getenv("EMAIL_MODE", "demo").strip().lower()


def _gmail_credentials() -> tuple[str, str]:
    address, app_password = os.getenv("GMAIL_ADDRESS"), os.getenv("GMAIL_APP_PASSWORD")
    if not address or not app_password:
        raise ValueError("EMAIL_MODE=gmail needs GMAIL_ADDRESS and GMAIL_APP_PASSWORD in .env")
    return address, app_password


def send_email(session_id: str, sender_id: str, sender_email: str, to: list[str], subject: str, body: str) -> str:
    if email_mode() == "gmail":
        address, app_password = _gmail_credentials()
        msg = EmailMessage()
        msg["From"] = address
        msg["To"] = ", ".join(to)
        msg["Subject"] = subject
        msg.set_content(body)
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(address, app_password)
            server.send_message(msg)
        return f"Email sent to {', '.join(to)} through Gmail."

    uni_api.deliver(session_id, {"user_id": sender_id, "email": sender_email}, to, subject, body)
    return f"Email delivered to {', '.join(to)} (demo mail inside the app, no real email was sent)."


def get_outbox(session_id: str, user_id: str) -> list[dict[str, Any]]:
    return uni_api.outbox(session_id, user_id)


def read_inbox(session_id: str, user_id: str, search: str = "", limit: int = 5) -> list[dict[str, Any]]:
    if email_mode() == "gmail":
        return _read_gmail(search, limit)

    inbox = uni_api.inbox(session_id, user_id)
    if search:
        needle = search.lower()
        inbox = [m for m in inbox if needle in f"{m['from']} {m['subject']} {m['body']}".lower()]
    return inbox[:limit]


def _read_gmail(search: str, limit: int) -> list[dict[str, Any]]:
    address, app_password = _gmail_credentials()
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    try:
        mail.login(address, app_password)
        mail.select("inbox", readonly=True)
        criteria = f'(TEXT "{search}")' if search else "ALL"
        status, data = mail.search(None, criteria)
        if status != "OK":
            raise RuntimeError("Could not search the inbox")

        messages = []
        for eid in reversed(data[0].split()[-limit:]):
            status, parts = mail.fetch(eid, "(RFC822)")
            if status != "OK":
                continue
            msg = email.message_from_bytes(parts[0][1])
            messages.append({
                "from": str(make_header(decode_header(msg.get("From", "")))),
                "subject": str(make_header(decode_header(msg.get("Subject", "")))),
                "date": msg.get("Date", ""),
                "body": _plain_body(msg)[:2000],
            })
        return messages
    finally:
        mail.logout()


def _plain_body(msg: email.message.Message) -> str:
    parts = msg.walk() if msg.is_multipart() else [msg]
    for part in parts:
        if part.get_content_type() == "text/plain" and "attachment" not in str(part.get("Content-Disposition")):
            payload = part.get_payload(decode=True) or b""
            return payload.decode(part.get_content_charset() or "utf-8", errors="replace")
    return ""
