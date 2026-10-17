"""The Streamlit page streams with astream_events (async), scripts use invoke (sync). A middleware that only
implements the sync hook crashes the app with NotImplementedError, so every hook needs both versions."""
import pytest
from langchain.agents.middleware import AgentMiddleware

from graph.graph import build_middleware

HOOK_PAIRS = [("wrap_model_call", "awrap_model_call"), ("wrap_tool_call", "awrap_tool_call")]


def overrides(middleware, name: str) -> bool:
    return getattr(type(middleware), name, None) is not getattr(AgentMiddleware, name, None)


@pytest.mark.parametrize("sync_hook, async_hook", HOOK_PAIRS)
def test_every_middleware_works_sync_and_async(sync_hook, async_hook):
    for middleware in build_middleware():
        if overrides(middleware, sync_hook):
            assert overrides(middleware, async_hook), f"{type(middleware).__name__} has {sync_hook} but no {async_hook}"
