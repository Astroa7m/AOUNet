"""Basic demo auth: fictional accounts in demo_users.json, passwords stored as salted PBKDF2 hashes."""
import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional

DEMO_USERS_PATH = Path(__file__).parent / "demo_users.json"
PBKDF2_ITERATIONS = 200_000

Role = Literal["guest", "student", "tutor"]


@dataclass(frozen=True)
class User:
    """The logged in user. `record` is the full demo record and never goes into a prompt as a whole."""
    user_id: str
    role: Role
    name: str
    email: str = ""
    title: str = ""
    record: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    @property
    def display_name(self) -> str:
        return f"{self.title} {self.name}".strip()


GUEST = User(user_id="guest", role="guest", name="Guest")


def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """Return (salt, hash) as hex strings"""
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), PBKDF2_ITERATIONS)
    return salt, digest.hex()


@lru_cache(maxsize=1)
def load_demo_data() -> dict[str, Any]:
    with open(DEMO_USERS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    """Drop the password fields so they never leave this module"""
    return {k: v for k, v in record.items() if k not in ("salt", "password_hash")}


def authenticate(user_id: str, password: str) -> Optional[User]:
    """Check an ID and password against the demo accounts"""
    user_id = user_id.strip().upper()
    data = load_demo_data()
    for role, key in (("student", "students"), ("tutor", "tutors")):
        for record in data[key]:
            if record["id"] != user_id:
                continue
            _, candidate = hash_password(password, record["salt"])
            if not hmac.compare_digest(candidate, record["password_hash"]):
                return None
            return User(
                user_id=record["id"],
                role=role,
                name=record["name"],
                email=record["email"],
                title=record.get("title", ""),
                record=_public_record(record),
            )
    return None


def get_user(user_id: str) -> Optional[dict[str, Any]]:
    """Public record of any demo student or tutor, used by the tools"""
    data = load_demo_data()
    for record in data["students"] + data["tutors"]:
        if record["id"] == user_id:
            return _public_record(record)
    return None


def get_modules() -> dict[str, dict[str, Any]]:
    return load_demo_data()["modules"]
