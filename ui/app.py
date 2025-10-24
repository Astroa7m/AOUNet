import random
import time

import streamlit as st
from langchain_core.messages import HumanMessage
from graph.graph import build_assistant
st.title("Echo bot mate")

def debugger(message):
    print("\n" + "DEBUGGER=" * 80)
    print(message)
    print("=DEBUGGER" * 80 + "\n")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def query_assistant(prompt):
    human_message = {"messages": [HumanMessage(content=prompt)]}
    return build_assistant().stream(human_message)


if prompt:=st.chat_input("What is up?"):

    # user message
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # assistant message
    with st.chat_message("assistant"):
        # keep this for debugging later
        response = st.write_stream(query_assistant(prompt))
    st.session_state.messages.append({"role": "assistant", "content": response})