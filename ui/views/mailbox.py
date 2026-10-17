import streamlit as st

from common import mailer
from ui.helpers.session import current_user, demo_session_id

user = current_user()
st.title("Mailbox")

if user.role not in ("student", "tutor"):
    st.warning("Log in to see your mailbox.")
    st.stop()

if mailer.email_mode() == "demo":
    st.caption("Demo mode: emails stay inside the app. Ask the assistant to read or send emails in the chat. "
               "Mail between Layla and Dr. Sara shows up in the other account's inbox.")

inbox_tab, sent_tab = st.tabs(["Inbox", "Sent"])

with inbox_tab:
    try:
        inbox = mailer.read_inbox(demo_session_id(), user.user_id, limit=20)
    except Exception as e:
        inbox = []
        st.error(f"Could not read the inbox: {e}")
    if not inbox:
        st.write("No emails.")
    for message in inbox:
        with st.expander(f"**{message['subject']}**  ·  {message['from']}  ·  {message['date']}"):
            st.text(message["body"])

with sent_tab:
    outbox = mailer.get_outbox(demo_session_id(), user.user_id)
    if not outbox:
        st.write("Nothing sent yet." if mailer.email_mode() == "demo" else "Sent mail lives in your Gmail account.")
    for message in outbox:
        with st.expander(f"**{message['subject']}**  ·  to {', '.join(message['to'])}  ·  {message['date']}"):
            st.text(message["body"])
