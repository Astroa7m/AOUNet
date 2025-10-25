import random
import time

import streamlit as st
from langchain_core.messages import HumanMessage

# from common.helpers import tool_name_mapper
from graph.graph import build_assistant

st.title("AOU Assistant")


def debugger(message):
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    print(f"{YELLOW}{message}{RESET}")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


async def query_assistant(prompt):
    """token-by-token streaming"""
    human_message = {"messages": [HumanMessage(content=prompt)]}

    async for event in build_assistant().astream_events(
            human_message,
            version="v1"
    ):
        event_type = event["event"]

        # detect when a tool is about to be called
        if event_type == "on_tool_start":
            tool_name = event["name"]
            tool_input = event["data"].get("input", {})

            # status card for the tool
            with st.status(f"🟢 Running **{tool_name}**...", expanded=True) as _:
                st.markdown(f"`{tool_input}`")

        elif event_type == "on_tool_end":
            tool_name = event["name"]
            output_data = event["data"].get("output", "No output returned")

            # just for debugging
            if debugging := False:
                with st.status(f"✅ **{tool_name}** completed", expanded=True) as status:
                    st.markdown("**Response:**")
                    st.code(str(output_data), language="json")
                    status.update(label=f"Tool `{tool_name}` finished", state="complete")

        # stream the actual content tokens
        elif event_type == "on_chat_model_stream":
            chunk = event["data"].get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                yield chunk.content


if prompt := st.chat_input("What is up?"):
    # user message
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # assistant message
    with st.chat_message("assistant"):
        # keep this for debugging later
        with st.spinner("Thinking... please wait"):
            response = st.write_stream(query_assistant(prompt))
    st.session_state.messages.append({"role": "assistant", "content": response})
