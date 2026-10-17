"""
Personal tools for logged in students and tutors.

Every tool reads the user from `runtime.context` (set by the app after login), never from model
arguments, so the model cannot ask for another user's data. Each tool also re-checks the role.
All data goes through the mock university API in uni/api.py.
"""
import json
from typing import Optional

from langchain.tools import ToolRuntime, tool

from common import mailer
from graph.schema import UserContext
from uni import api as uni_api


def _as_text(data) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def _require(runtime: ToolRuntime[UserContext], role: str) -> Optional[str]:
    """Error message when the current user is not allowed to use a tool"""
    if runtime.context.role != role:
        return f"This tool is only available to logged in {role}s."
    return None


def _me(runtime: ToolRuntime[UserContext]) -> dict:
    return uni_api.get_person(runtime.context.session_id, runtime.context.user_id)


def _module_label(code: str) -> str:
    module = uni_api.get_module(code) or {}
    return f"{code} {module.get('title', '')}".strip()


# ============================================================================
# STUDENT TOOLS
# ============================================================================

@tool
def get_my_student_profile(runtime: ToolRuntime[UserContext]) -> str:
    """Get the logged in student's profile: program, level, GPA, credits, fee balance, contact info and enrolled modules."""
    if error := _require(runtime, "student"):
        return error
    record = _me(runtime)
    return _as_text({
        "name": record["name"],
        "student_id": record["id"],
        "university_email": record["email"],
        "program": record["program"],
        "level": record["level"],
        "gpa": record["gpa"],
        "completed_credits": record["completed_credits"],
        "fees": record["fees"],
        "contact": record["contact"],
        "enrolled_modules": [_module_label(code) for code in record["enrolled"]],
    })


@tool
def get_my_marks(runtime: ToolRuntime[UserContext], module_code: Optional[str] = None) -> str:
    """
    Get the logged in student's marks (TMA, MTA, final exam, attendance) and assessment deadlines for current modules.
    Args:
        module_code: optional module code such as TM354, leave empty for all modules
    """
    if error := _require(runtime, "student"):
        return error
    enrolled = _me(runtime)["enrolled"]
    if module_code:
        code = module_code.strip().upper()
        if code not in enrolled:
            return f"You are not enrolled in {code}. Your modules are: {', '.join(enrolled)}."
        enrolled = {code: enrolled[code]}
    return _as_text({
        _module_label(code): {"marks": marks, "deadlines": uni_api.get_module(code)["deadlines"]}
        for code, marks in enrolled.items()
    })


@tool
def get_my_schedule(runtime: ToolRuntime[UserContext], day: Optional[str] = None) -> str:
    """
    Get the logged in student's weekly tutorial schedule.
    Args:
        day: optional day name such as Monday, leave empty for the whole week
    """
    if error := _require(runtime, "student"):
        return error
    schedule = _me(runtime)["schedule"]
    if day:
        schedule = [s for s in schedule if s["day"].lower() == day.strip().lower()]
        if not schedule:
            return f"You have no tutorials on {day}."
    return _as_text([{**s, "module": _module_label(s["module"])} for s in schedule])


@tool
def get_my_tutors(runtime: ToolRuntime[UserContext]) -> str:
    """Get the tutors of the logged in student's modules with their email, office and office hours."""
    if error := _require(runtime, "student"):
        return error
    tutors = []
    for code in _me(runtime)["enrolled"]:
        tutor = uni_api.get_person(runtime.context.session_id, uni_api.get_module(code)["tutor_id"])
        tutors.append({
            "module": _module_label(code),
            "tutor": f"{tutor['title']} {tutor['name']}",
            "email": tutor["email"],
            "office": tutor["office"],
            "office_hours": tutor["office_hours"],
        })
    return _as_text(tutors)


# ============================================================================
# TUTOR TOOLS
# ============================================================================

@tool
def get_my_tutor_profile(runtime: ToolRuntime[UserContext]) -> str:
    """Get the logged in tutor's profile: faculty, office, office hours, contact info, and their modules with deadlines."""
    if error := _require(runtime, "tutor"):
        return error
    record = _me(runtime)
    return _as_text({
        "name": f"{record['title']} {record['name']}",
        "email": record["email"],
        "faculty": record["faculty"],
        "office": record["office"],
        "office_hours": record["office_hours"],
        "contact": record["contact"],
        "modules": {
            _module_label(code): {"deadlines": uni_api.get_module(code)["deadlines"]}
            for code in uni_api.tutor_modules(record["id"])
        },
    })


@tool
def get_module_roster(runtime: ToolRuntime[UserContext], module_code: str) -> str:
    """
    Get the students enrolled in one of the logged in tutor's modules, with their marks and attendance.
    Args:
        module_code: module code such as TM354, must be a module this tutor teaches
    """
    if error := _require(runtime, "tutor"):
        return error
    code = module_code.strip().upper()
    own_modules = uni_api.tutor_modules(runtime.context.user_id)
    if code not in own_modules:
        return f"You do not teach {code}. Your modules are: {', '.join(own_modules)}."

    roster = [
        {"student_id": s["id"], "name": s["name"], "email": s["email"], **s["enrolled"][code]}
        for s in uni_api.module_students(code)
    ]
    return _as_text({"module": _module_label(code), "deadlines": uni_api.get_module(code)["deadlines"], "students": roster})


# ============================================================================
# SHARED
# ============================================================================

@tool
def read_my_inbox(runtime: ToolRuntime[UserContext], search: Optional[str] = None, limit: int = 5) -> str:
    """
    Read the logged in user's latest emails, newest first.
    Args:
        search: optional word to filter by sender, subject or body
        limit: how many emails to return, default 5
    """
    context = runtime.context
    if context.role not in ("student", "tutor"):
        return "Please log in to read emails."
    try:
        messages = mailer.read_inbox(context.session_id, context.user_id, search or "", max(1, min(limit, 20)))
    except Exception as e:
        return f"Could not read the inbox: {e}"
    return _as_text(messages) if messages else "No emails found."


@tool
def send_email(runtime: ToolRuntime[UserContext], to: list[str], subject: str, body: str) -> str:
    """
    Send an email from the logged in user. The user is asked to approve it before it is sent.
    Students can only email the tutors of their own modules. Write a complete, polite email body.
    Args:
        to: list of recipient email addresses
        subject: email subject
        body: full email text, signed with the user's name
    """
    context = runtime.context
    if context.role not in ("student", "tutor"):
        return "Please log in to send emails."
    if not to:
        return "No recipient given."

    if context.role == "student":
        enrolled = _me(runtime)["enrolled"]
        allowed = {uni_api.get_person(context.session_id, uni_api.get_module(code)["tutor_id"])["email"] for code in enrolled}
        blocked = [address for address in to if address.lower() not in allowed]
        if blocked:
            return f"Students can only email their own tutors ({', '.join(sorted(allowed))}). Not allowed: {', '.join(blocked)}."

    try:
        return mailer.send_email(context.session_id, context.user_id, context.email, to, subject, body)
    except Exception as e:
        return f"Failed to send email: {e}"


STUDENT_TOOLS = [get_my_student_profile, get_my_marks, get_my_schedule, get_my_tutors, read_my_inbox, send_email]
TUTOR_TOOLS = [get_my_tutor_profile, get_module_roster, read_my_inbox, send_email]
