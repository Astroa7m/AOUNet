"""Login, per-role tool guards and the mock university API. No API keys needed: tools are called directly with a fake runtime."""
import json
import uuid
from types import SimpleNamespace

import pytest

from auth import authenticate
from graph import user_tools
from graph.schema import UserContext
from uni import api as uni_api


@pytest.fixture
def session():
    """A fresh demo session per test, like a new browser"""
    return str(uuid.uuid4())


@pytest.fixture
def users(session):
    return SimpleNamespace(
        layla=UserContext("S1001", "student", "Layla Al-Harthy", "s1001@demo.aou.edu.om", session),
        omar=UserContext("S1002", "student", "Omar Al-Busaidi", "s1002@demo.aou.edu.om", session),
        sara=UserContext("T2001", "tutor", "Dr. Sara Haddad", "t2001@demo.aou.edu.om", session),
        guest=UserContext(session_id=session),
    )


@pytest.fixture(autouse=True)
def demo_email(monkeypatch):
    monkeypatch.setenv("EMAIL_MODE", "demo")


def call(tool, context, **kwargs):
    """Run a tool's function the way the agent would, with the user injected as runtime"""
    return tool.func(runtime=SimpleNamespace(context=context), **kwargs)


@pytest.mark.parametrize("user_id, password, role", [
    ("S1001", "student123", "student"),
    ("s1001", "student123", "student"),  # IDs are case-insensitive
    ("T2001", "tutor123", "tutor"),
])
def test_login_success(user_id, password, role):
    user = authenticate(user_id, password)
    assert user is not None and user.role == role
    assert "password_hash" not in user.record and "salt" not in user.record


@pytest.mark.parametrize("user_id, password", [("S1001", "wrong"), ("T2001", "student123"), ("X9999", "x"), ("", "")])
def test_login_failure(user_id, password):
    assert authenticate(user_id, password) is None


def test_model_cannot_see_or_set_the_user():
    for tool in user_tools.STUDENT_TOOLS + user_tools.TUTOR_TOOLS:
        assert "runtime" not in tool.tool_call_schema.model_json_schema().get("properties", {})


def test_demo_student_and_tutor_are_linked():
    """The one click demo accounts must share modules, that is the whole demo story"""
    assert set(uni_api.tutor_modules("T2001")) & set(uni_api.get_person("x", "S1001")["enrolled"])


def test_student_marks_include_real_deadlines(users):
    marks = json.loads(call(user_tools.get_my_marks, users.layla, module_code="tm354"))
    assert marks["TM354 Software Engineering"]["deadlines"]["TMA02"] == "2026-10-16"
    assert "not enrolled in B207A" in call(user_tools.get_my_marks, users.layla, module_code="B207A")


@pytest.mark.parametrize("tool, kwargs", [
    (user_tools.get_module_roster, {"module_code": "TM354"}),
    (user_tools.get_my_tutor_profile, {}),
])
def test_student_and_guest_cannot_use_tutor_tools(users, tool, kwargs):
    for context in (users.layla, users.guest):
        assert "only available to logged in tutors" in call(tool, context, **kwargs)


def test_tutor_cannot_use_student_tools(users):
    assert "only available to logged in students" in call(user_tools.get_my_marks, users.sara)


def test_tutor_roster_limited_to_own_modules(users):
    roster = json.loads(call(user_tools.get_module_roster, users.sara, module_code="tm354"))
    assert [s["student_id"] for s in roster["students"]] == ["S1001"]
    assert "You do not teach B207A" in call(user_tools.get_module_roster, users.sara, module_code="B207A")


def test_student_can_only_email_own_tutors(users, session):
    blocked = call(user_tools.send_email, users.omar, to=["t2001@demo.aou.edu.om"], subject="Hi", body="Hello")
    assert "Not allowed" in blocked and not uni_api.outbox(session, "S1002")
    assert "demo mail" in call(user_tools.send_email, users.omar, to=["t2002@demo.aou.edu.om"], subject="Hi", body="Hello")


def test_email_from_student_arrives_in_tutor_inbox(users, session):
    call(user_tools.send_email, users.layla, to=["t2001@demo.aou.edu.om"], subject="TMA02 question", body="Hello Dr. Sara")
    inbox = json.loads(call(user_tools.read_my_inbox, users.sara, search="TMA02 question"))
    assert inbox[0]["from"] == "s1001@demo.aou.edu.om"
    # and back again
    call(user_tools.send_email, users.sara, to=["s1001@demo.aou.edu.om"], subject="Re: TMA02 question", body="Hi Layla")
    assert json.loads(call(user_tools.read_my_inbox, users.layla))[0]["subject"] == "Re: TMA02 question"


def test_sessions_are_private(users, session):
    call(user_tools.send_email, users.layla, to=["t2001@demo.aou.edu.om"], subject="Private", body="x")
    other_visitor = UserContext("T2001", "tutor", "Dr. Sara Haddad", "t2001@demo.aou.edu.om", "another-browser")
    assert "Private" not in call(user_tools.read_my_inbox, other_visitor, limit=20)


def test_profile_edit_only_allowed_fields(users, session):
    updated = uni_api.update_profile(session, "S1001", "student", {"phone": "+968 1234 5678", "gpa": "4.0"})
    assert updated["contact"]["phone"] == "+968 1234 5678"
    assert updated["gpa"] == 3.42  # academic record is read-only
    profile = json.loads(call(user_tools.get_my_student_profile, users.layla))
    assert profile["contact"]["phone"] == "+968 1234 5678"  # the assistant sees the edit
    assert uni_api.get_person("another-browser", "S1001")["contact"]["phone"] != "+968 1234 5678"


def test_guest_cannot_send_email(users):
    assert "log in" in call(user_tools.send_email, users.guest, to=["t2001@demo.aou.edu.om"], subject="Hi", body="x")
