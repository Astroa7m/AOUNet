import base64
import time
from pathlib import Path
from string import Template

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from streamlit_theme import st_theme
from graph.graph import build_assistant, get_agent
from ui.helpers.client import get_client_info
from ui.helpers.query_logger import QueryLogger

# setups

bot_icon_path = Path(__file__).parent / "assets" / "aou_bot_icon.png"
user_icon_path = Path(__file__).parent / "assets" / "user_icon.png"
bg_image_path = Path(__file__).parent / "assets" / "bg.jpg"
# Initialize logger
@st.cache_resource
def get_logger():
    """Initialize query logger once"""
    return QueryLogger()  # Reads from DATABASE_URL env variable

logger = get_logger()

# have to set it initially
def get_theme():
    try:
        return st_theme()['base']
    except:
        return 'dark'

@st.cache_data
def get_theme_colors(theme):
    overlay_color = "#ffffff" if theme == "light" else "#0e1117"
    header_text_color = "#002d57" if theme == "light" else "#ffffff"
    return overlay_color, header_text_color

theme = get_theme()
overlay_color, header_text_color = get_theme_colors(theme)


suggestions = [
    "Hello, how do I reset my password?",
    "What services are available for students?",
    "Show me the latest announcements."
]

def debugger(message):
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    print(f"{YELLOW}{message}{RESET}")

# page code

st.set_page_config(
    page_title="AOUNet",
    page_icon=str(bot_icon_path),
)

@st.cache_data
def get_base64_image(image_path):
    """Convert image to base64 string"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# overriding styles/creating elements and making them adapt to ui theme change
@st.cache_data
def get_custom_styles(overlay_color, header_text_color, bg_image_base64):
    return Template("""
<style>

[data-testid="stAppViewContainer"] {
    background-image: url("data:image/jpeg;base64,$bg_image");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    position: relative;
}

    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: $overlay_color;
        opacity: 0.8;
        z-index: 0;
        pointer-events: none;
    }

        header[data-testid="stHeader"]::before {
        content: "AOUNet";
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        font-size: 1.5rem;
        font-weight: bold;
        color: $header_text_color;
        pointer-events: none;
    }


        /* Target buttons inside columns, which we'll use for pills */
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button {
        border: 1px solid #888888; /* Light border */
        border-radius: 20px;       /* Rounded corners */
        padding: 5px 15px;         /* Padding */
        background-color: transparent; /* No background */
        color: inherit;            /* Inherit text color */
        font-weight: 500;
    }
    /* Hover effect */
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:hover {
        border-color: #007bff;     /* Highlight border on hover */
        color: #007bff;            /* Highlight text on hover */
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button:focus {
        box-shadow: none;          /* Remove default focus shadow */
        border-color: #007bff;
        color: #007bff;
    }
</style>
""").substitute(overlay_color=overlay_color, header_text_color=header_text_color, bg_image=bg_image_base64)


st.markdown(get_custom_styles(overlay_color, header_text_color, get_base64_image(bg_image_path) if bg_image_path.exists() else ""), unsafe_allow_html=True)

# session state init

if "assistant" not in st.session_state:
    #st.session_state.assistant = build_assistant()
    st.session_state.assistant = get_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    import uuid
    st.session_state.thread_id = str(uuid.uuid4())

with st.sidebar:

    if bot_icon_path.exists():
        col_logo = st.columns([1, 2, 1])
        with col_logo[1]:
            st.image(str(bot_icon_path), width=100)

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

async def query_assistant(prompt):
    """token-by-token streaming"""
    # Get client info
    error_msg = None
    full_response = ""
    try:
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
                full_response += f"\nUsed tool: {tool_name}\n"
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
                    full_response += chunk.content
                    yield chunk.content
    except Exception as e:
        error_msg = str(e)
        print(f"Error in query_assistant: {e}")
        raise
    finally:
        # update the log with the complete response
        if log_id:
            logger.log_query(
                id=log_id,
                response=full_response if full_response else "empty response",
                error_message=error_msg
            )


# only fetch history if messages cache is empty
if not st.session_state.messages:
    st.session_state.messages = get_conversation_history()

# display chat messages from LangGraph's memory
for message in st.session_state.messages:
    role = message["role"]
    icon_path = bot_icon_path if role == "assistant" else user_icon_path
    with st.chat_message(role, avatar=icon_path):
        st.write(message["content"])


prompt = None

# if we do not have any history show the pills
if not st.session_state.messages:
    st.write("What's on your mind today?")

    # creating columns based on suggestions count
    cols = st.columns(len(suggestions))

    for i, (col, prompt_text) in enumerate(zip(cols, suggestions)):
        with col:
            # if a button is clicked, set the 'prompt' variable
            if st.button(prompt_text, key=f"pill_{i}", use_container_width=True):
                prompt = prompt_text

if chat_input_prompt := st.chat_input("What is up?"):
    prompt = chat_input_prompt

if prompt:
    # get client info
    ip_address, user_agent = get_client_info()

    # log the query immediately and get the log ID
    log_id = logger.log_query(
        query_text=prompt,
        ip_address=ip_address,
        user_agent=user_agent
    ).data[0]['id']

    # adding user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # display user message
    with st.chat_message("user", avatar=user_icon_path):
        st.write(prompt)

    # display assistant message with streaming
    with st.chat_message("assistant", avatar=bot_icon_path):
        # keep this for debugging later
        try:
            response = st.write_stream(query_assistant(prompt))
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"An error occurred processing your query., {e}")
            # update log with error if response generation failed
            if log_id:
                logger.log_query(id=log_id, error_message=str(e))
    # force rerun to refresh conversation history from checkpointer
    # we will let streamlit handle this naturally
    # st.rerun()
