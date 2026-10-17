import sys

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from graph.graph import get_agent
from ui.helpers.client import get_client_info
from ui.helpers.session import BOT_ICON, USER_ICON, current_user, safe_log, user_context

user = current_user()
agent = get_agent(user.role)
config = {"configurable": {"thread_id": st.session_state.thread_id}}

suggestions = {
    "guest": [
        "Hello, how do I reset my password?",
        "What services are available for students?",
        "Show me the latest announcements.",
    ],
    "student": [
        "What are my marks and upcoming deadlines?",
        "When are my tutorials this week?",
        "Email Dr. Sara asking for 3 more days on TM354 TMA02",
    ],
    "tutor": [
        "How are my TM354 students doing?",
        "Summarize my latest emails",
        "Reply to Layla's extension request and approve it",
    ],
}[user.role]


def get_conversation_history():
    """Retrieve conversation history from LangGraph's checkpointer"""
    try:
        state = agent.get_state(config)

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


def pending_emails() -> list[dict]:
    """Emails the agent paused on, waiting for the user to approve or reject"""
    try:
        interrupts = agent.get_state(config).interrupts
    except Exception:
        return []
    return [request for interrupt in interrupts for request in interrupt.value.get("action_requests", [])]


async def query_assistant(agent_input, log_id=None):
    """token-by-token streaming, agent_input is either new messages or a Command that resumes a paused run"""
    error_msg = None
    full_response = ""
    try:
        async for event in agent.astream_events(
                agent_input,
                config=config,
                context=user_context(),
                version="v2"
        ):
            event_type = event["event"]

            # detect when a tool is about to be called
            if event_type == "on_tool_start":
                tool_name = event["name"]
                # the injected runtime holds the user context, not useful on screen
                tool_input = {k: v for k, v in (event["data"].get("input") or {}).items() if k != "runtime"}

                # status card for the tool
                with st.status(f"🟢 Running **{tool_name}**...", expanded=False) as _:
                    st.markdown(f"`{tool_input}`")

            elif event_type == "on_tool_end":
                full_response += f"\nUsed tool: {event['name']}\n"

            # only the agent's "model" node, so the summarization model's output never reaches the user
            elif event_type == "on_chat_model_stream" and event["metadata"].get("langgraph_node") == "model":
                chunk = event["data"].get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    full_response += chunk.content
                    yield chunk.content
    except Exception as e:
        error_msg = str(e)
        # flush, otherwise the line sits in a buffer and never shows in the Streamlit Cloud logs
        print(f"Error in query_assistant: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        raise
    finally:
        # update the log with the complete response
        if log_id:
            safe_log(
                id=log_id,
                response=full_response if full_response else "empty response",
                error_message=error_msg
            )


def friendly_error(error: Exception) -> str:
    """Plain words for the errors a free Groq key hits, the full error is in the logs"""
    text = str(error)
    if "413" in text or "Request too large" in text:
        return "This conversation got too long for the free AI tier. Press **Clear Conversation** in the sidebar and ask again."
    if "429" in text or "rate limit" in text.lower():
        return "The free AI quota is busy right now. Please wait a minute and ask again."
    return "Something went wrong while answering. Please try again."


def run_and_render(agent_input, log_id=None):
    """Stream one agent run into an assistant bubble, then refresh if it paused on an email"""
    with st.chat_message("assistant", avatar=BOT_ICON):
        try:
            response = st.write_stream(query_assistant(agent_input, log_id))
            if not response and not pending_emails():
                # messages added without streaming, e.g. the retry middleware's apology
                last = (agent.get_state(config).values.get("messages") or [None])[-1]
                if isinstance(last, AIMessage) and last.content and not last.tool_calls:
                    response = last.content
                    st.write(response)
            if response:
                st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(friendly_error(e))
            if log_id:
                safe_log(id=log_id, error_message=str(e))
    if pending_emails():
        # rerun so the approval card shows up
        st.rerun()


# only fetch history if messages cache is empty
if not st.session_state.messages:
    st.session_state.messages = get_conversation_history()

# display chat messages from LangGraph's memory
for message in st.session_state.messages:
    role = message["role"]
    icon_path = BOT_ICON if role == "assistant" else USER_ICON
    with st.chat_message(role, avatar=icon_path):
        st.write(message["content"])


# the user clicked Approve or Don't send on the previous run
if decision := st.session_state.pop("email_decision", None):
    count = len(pending_emails())
    if decision == "approve":
        decisions = [{"type": "approve"}] * count
    else:
        decisions = [{"type": "reject", "message": "The user chose not to send this email."}] * count
    run_and_render(Command(resume={"decisions": decisions}))

waiting = pending_emails()
if waiting:
    with st.chat_message("assistant", avatar=BOT_ICON):
        st.markdown("**Please review before I send:**")
        for request in waiting:
            args = request["args"]
            with st.container(border=True):
                st.markdown(f"**To:** {', '.join(args.get('to', []))}  \n**Subject:** {args.get('subject', '')}")
                st.text(args.get("body", ""))
        approve_col, reject_col = st.columns(2)
        if approve_col.button("Approve and send", type="primary", use_container_width=True):
            st.session_state.email_decision = "approve"
            st.rerun()
        if reject_col.button("Don't send", use_container_width=True):
            st.session_state.email_decision = "reject"
            st.rerun()


prompt = None

# if we do not have any history show the pills
if not st.session_state.messages and not waiting:
    greeting = "What's on your mind today?" if user.role == "guest" else f"Welcome back, {user.display_name}. What can I do for you?"
    st.write(greeting)

    # creating columns based on suggestions count
    cols = st.columns(len(suggestions))

    for i, (col, prompt_text) in enumerate(zip(cols, suggestions)):
        with col:
            # if a button is clicked, set the 'prompt' variable
            if st.button(prompt_text, key=f"pill_{i}", use_container_width=True):
                prompt = prompt_text

placeholder = "Approve or reject the email above first" if waiting else "Ask about AOU Oman..."
if chat_input_prompt := st.chat_input(placeholder, disabled=bool(waiting)):
    prompt = chat_input_prompt

if prompt:
    # get client info
    ip_address, user_agent = get_client_info()

    # log the query immediately and get the log ID
    log_result = safe_log(
        query_text=prompt,
        ip_address=ip_address,
        user_agent=user_agent
    )
    log_id = log_result.data[0]['id'] if log_result and log_result.data else None

    # adding user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # display user message
    with st.chat_message("user", avatar=USER_ICON):
        st.write(prompt)

    run_and_render({"messages": [HumanMessage(content=prompt)]}, log_id)
