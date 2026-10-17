"""
Mock AOU university API (think of the SIS behind the student portal).

In production these functions would call the university's REST API. Here they read the fictional
records in auth/demo_users.json and keep every change in a per-session overlay, so each visitor of
the public demo gets their own private copy: profile edits and emails never leak between visitors.
"""
import copy
from collections import OrderedDict
from datetime import date
from threading import Lock
from typing import Any, Optional

from auth import get_modules, get_user, load_demo_data

# what each role may change on the Profile page, academic records stay read-only
EDITABLE_FIELDS = {
    "student": ["phone", "personal_email", "address", "emergency_contact"],
    "tutor": ["phone", "office", "office_hours", "bio"],
}

MAX_SESSIONS = 500  # oldest demo sessions are dropped first

_sessions: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_lock = Lock()


def _session(session_id: str) -> dict[str, Any]:
    with _lock:
        if session_id not in _sessions:
            _sessions[session_id] = {"profiles": {}, "inbox": {}, "outbox": {}}
            while len(_sessions) > MAX_SESSIONS:
                _sessions.popitem(last=False)
        _sessions.move_to_end(session_id)
        return _sessions[session_id]


# ============================================================================
# PEOPLE
# ============================================================================

def get_person(session_id: str, user_id: str) -> Optional[dict[str, Any]]:
    """A student or tutor record with this session's edits applied"""
    record = get_user(user_id)
    if record is None:
        return None
    record = copy.deepcopy(record)
    edits = _session(session_id)["profiles"].get(user_id, {})
    record["contact"] = {**record.get("contact", {}), **edits.get("contact", {})}
    for field in ("office", "office_hours"):
        if field in edits:
            record[field] = edits[field]
    return record


def find_person_by_email(email_address: str) -> Optional[dict[str, Any]]:
    data = load_demo_data()
    for record in data["students"] + data["tutors"]:
        if record["email"].lower() == email_address.strip().lower():
            return record
    return None


def update_profile(session_id: str, user_id: str, role: str, changes: dict[str, str]) -> dict[str, Any]:
    """Apply allowed edits, ignore anything else. Returns the updated record."""
    allowed = EDITABLE_FIELDS.get(role, [])
    edits = _session(session_id)["profiles"].setdefault(user_id, {})
    for field, value in changes.items():
        if field not in allowed:
            continue
        value = str(value).strip()[:200]
        if field in ("office", "office_hours"):
            edits[field] = value
        else:
            edits.setdefault("contact", {})[field] = value
    return get_person(session_id, user_id)


# ============================================================================
# MODULES
# ============================================================================

def get_module(code: str) -> Optional[dict[str, Any]]:
    module = get_modules().get(code.strip().upper())
    return {"code": code.strip().upper(), **module} if module else None


def tutor_modules(tutor_id: str) -> list[str]:
    return [code for code, module in get_modules().items() if module["tutor_id"] == tutor_id]


def module_students(code: str) -> list[dict[str, Any]]:
    code = code.strip().upper()
    return [s for s in load_demo_data()["students"] if code in s["enrolled"]]


# ============================================================================
# DEMO MAIL
# ============================================================================

def deliver(session_id: str, sender: dict[str, str], to: list[str], subject: str, body: str) -> None:
    """Store in the sender's Sent box and in the inbox of every demo user it is addressed to"""
    state = _session(session_id)
    message = {"from": sender["email"], "to": to, "subject": subject, "body": body, "date": date.today().isoformat()}
    state["outbox"].setdefault(sender["user_id"], []).append(message)
    for address in to:
        recipient = find_person_by_email(address)
        if recipient:
            state["inbox"].setdefault(recipient["id"], []).append(message)


def inbox(session_id: str, user_id: str) -> list[dict[str, Any]]:
    """Newest first: mail delivered in this session, then the seeded demo mail"""
    delivered = list(reversed(_session(session_id)["inbox"].get(user_id, [])))
    seeded = (get_user(user_id) or {}).get("inbox", [])
    return delivered + seeded


def outbox(session_id: str, user_id: str) -> list[dict[str, Any]]:
    return list(reversed(_session(session_id)["outbox"].get(user_id, [])))
