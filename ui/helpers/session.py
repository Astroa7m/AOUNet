"""Session state shared by all pages: who is logged in, the chat thread, and query logging."""
import uuid
from pathlib import Path

import streamlit as st

from auth import GUEST, User, authenticate
from graph.schema import UserContext
from ui.helpers.query_logger import QueryLogger

ASSETS = Path(__file__).parent.parent / "assets"
BOT_ICON = ASSETS / "aou_bot_icon.png"
USER_ICON = ASSETS / "user_icon.png"
BG_IMAGE = ASSETS / "bg.jpg"

# one click demo logins. Layla takes TM354 and M269, both taught by Dr. Sara, so the two accounts talk to each other.
DEMO_STUDENT = ("S1001", "student123")
DEMO_TUTOR = ("T2001", "tutor123")


def init_session():
    st.session_state.setdefault("user", GUEST)
    st.session_state.setdefault("thread_id", str(uuid.uuid4()))
    st.session_state.setdefault("messages", [])
    # survives log out, so an email sent as the student is still in the tutor's inbox after switching
    st.session_state.setdefault("demo_session_id", str(uuid.uuid4()))


def current_user() -> User:
    return st.session_state.user


def demo_session_id() -> str:
    return st.session_state.demo_session_id


def user_context() -> UserContext:
    user = current_user()
    return UserContext(
        user_id=user.user_id,
        role=user.role,
        name=user.display_name,
        email=user.email,
        session_id=demo_session_id(),
    )


def start_new_chat():
    """Fresh thread, so the agent memory and the screen both start empty"""
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.pop("email_decision", None)


def log_in(user: User):
    st.session_state.user = user
    start_new_chat()


def log_in_demo(account: tuple[str, str]):
    log_in(authenticate(*account))


def log_out():
    st.session_state.user = GUEST
    start_new_chat()


@st.cache_resource
def get_logger():
    """Initialize query logger once, or None when Supabase is not configured"""
    try:
        return QueryLogger()  # reads SUPABASE_URL and SUPABASE_KEY
    except Exception as e:
        print(f"Query logging disabled: {e}")
        return None


def safe_log(**kwargs):
    """Log to Supabase without ever breaking the chat"""
    logger = get_logger()
    if logger is None:
        return None
    try:
        return logger.log_query(**kwargs)
    except Exception as e:
        print(f"Could not log query: {e}")
        return None
