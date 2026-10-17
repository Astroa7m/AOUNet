import base64
from string import Template

import streamlit as st
from streamlit_theme import st_theme

from common.mailer import email_mode
from ui.helpers.session import (
    BG_IMAGE,
    BOT_ICON,
    DEMO_STUDENT,
    DEMO_TUTOR,
    current_user,
    init_session,
    log_in_demo,
    log_out,
    start_new_chat,
)

st.set_page_config(
    page_title="AOUNet",
    page_icon=str(BOT_ICON),
)

init_session()


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
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"] {
        border: 1px solid #888888; /* Light border */
        border-radius: 20px;       /* Rounded corners */
        padding: 5px 15px;         /* Padding */
        background-color: transparent; /* No background */
        color: inherit;            /* Inherit text color */
        font-weight: 500;
    }
    /* Hover effect */
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"]:hover {
        border-color: #007bff;     /* Highlight border on hover */
        color: #007bff;            /* Highlight text on hover */
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] > button[data-testid="stBaseButton-secondary"]:focus {
        box-shadow: none;          /* Remove default focus shadow */
        border-color: #007bff;
        color: #007bff;
    }
</style>
""").substitute(overlay_color=overlay_color, header_text_color=header_text_color, bg_image=bg_image_base64)


st.markdown(get_custom_styles(overlay_color, header_text_color, get_base64_image(BG_IMAGE) if BG_IMAGE.exists() else ""), unsafe_allow_html=True)

user = current_user()

with st.sidebar:

    if BOT_ICON.exists():
        col_logo = st.columns([1, 2, 1])
        with col_logo[1]:
            st.image(str(BOT_ICON), width=100)

    if user.role == "guest":
        st.info("Browsing as a **guest**. Log in as a student or tutor for personal tools.")
    else:
        st.success(f"Logged in as **{user.display_name}**  \n{user.role.title()} · `{user.user_id}`")
        if st.button("Log out", use_container_width=True):
            log_out()
            st.rerun()

    st.subheader("Try the demo")
    st.caption("Layla studies TM354 with Dr. Sara. Email one from the other, then switch accounts to see it arrive.")
    if st.button("🎓 Student: Layla", use_container_width=True, disabled=user.user_id == DEMO_STUDENT[0]):
        log_in_demo(DEMO_STUDENT)
        st.rerun()
    if st.button("🧑‍🏫 Tutor: Dr. Sara", use_container_width=True, disabled=user.user_id == DEMO_TUTOR[0]):
        log_in_demo(DEMO_TUTOR)
        st.rerun()

    st.header("Conversation Management")

    if st.button("Clear Conversation"):
        start_new_chat()
        st.rerun()

    st.caption(f"Thread ID: `{st.session_state.thread_id[:8]}...`")
    st.caption(f"Email mode: `{email_mode()}`")

chat_page = st.Page("views/chat.py", title="Chat", icon=":material/chat:", default=True)
pages = [chat_page]
if user.role == "guest":
    pages.append(st.Page("views/login.py", title="Log in", icon=":material/login:"))
else:
    pages.append(st.Page("views/profile.py", title="My profile", icon=":material/person:"))
    pages.append(st.Page("views/mailbox.py", title="Mailbox", icon=":material/mail:"))

st.navigation(pages).run()
