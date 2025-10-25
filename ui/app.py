import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from ..graph.graph import build_assistant
def debugger(message):
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    print(f"{YELLOW}{message}{RESET}")

st.title("AOU Assistant")




# initialize the assistant (singleton)
if "assistant" not in st.session_state:
    st.session_state.assistant = build_assistant()


# initialize chat history and thread_id
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    import uuid
    st.session_state.thread_id = str(uuid.uuid4())

with st.sidebar:
    st.header("Conversation Management")

    if st.button("Clear Conversation"):
        # create a new thread ID
        import uuid

        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.caption(f"Thread ID: `{st.session_state.thread_id[:8]}...`")

def get_conversation_history():
    """Retrieve conversation history from LangGraph's checkpointer"""
    try:
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        state = st.session_state.assistant.get_state(config)

        # extract only HumanMessage and AIMessage for display
        messages = []
        for msg in state.values.get("messages", []):
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage) and msg.content:
                # skip tool calls and empty messages, so we do not populate the token
                if not hasattr(msg, 'tool_calls') or not msg.tool_calls:
                    messages.append({"role": "assistant", "content": msg.content})

        return messages
    except Exception as e:
        print(f"Error retrieving history: {e}")
        return []


# display chat messages from LangGraph's memory
for message in get_conversation_history():
    with st.chat_message(message["role"]):
        st.write(message["content"])

async def query_assistant(prompt):
    """token-by-token streaming"""
    human_message = {"messages": [HumanMessage(content=prompt)]}
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    async for event in st.session_state.assistant.astream_events(
            human_message,
            config=config,
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
    # display user message
    with st.chat_message("user"):
        st.write(prompt)

    # display assistant message with streaming
    with st.chat_message("assistant"):
        # keep this for debugging later
        with st.spinner("Thinking... please wait"):
            response = st.write_stream(query_assistant(prompt))

    # force rerun to refresh conversation history from checkpointer
    st.rerun()
