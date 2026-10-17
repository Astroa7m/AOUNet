import streamlit as st

from auth import authenticate
from ui.helpers.session import DEMO_STUDENT, DEMO_TUTOR, log_in, log_in_demo

st.title("Log in to AOUNet")
st.write("Students see their marks, schedule, fees and tutors. Tutors see their modules, rosters and inbox.")

st.subheader("Try it in one click")
st.caption("Layla studies TM354 and M269 with Dr. Sara. Email one from the other, then switch accounts to see it arrive.")
student_col, tutor_col = st.columns(2)
if student_col.button("🎓 Try as student (Layla)", type="primary", use_container_width=True):
    log_in_demo(DEMO_STUDENT)
    st.rerun()
if tutor_col.button("🧑‍🏫 Try as tutor (Dr. Sara)", type="primary", use_container_width=True):
    log_in_demo(DEMO_TUTOR)
    st.rerun()

st.subheader("Or log in with an ID")
with st.form("login"):
    user_id = st.text_input("ID", placeholder="S1001 or T2001")
    password = st.text_input("Password", type="password")
    submitted = st.form_submit_button("Log in", use_container_width=True)

if submitted:
    user = authenticate(user_id, password)
    if user is None:
        st.error("Wrong ID or password.")
    else:
        log_in(user)
        st.rerun()

with st.expander("Demo accounts (fictional people)"):
    st.markdown(
        """
| Role | ID | Password | Name |
|---|---|---|---|
| Student | `S1001` | `student123` | Layla Al-Harthy, BSc ITC, takes TM354 and M269 with Dr. Sara |
| Student | `S1002` | `student456` | Omar Al-Busaidi, Business Studies |
| Tutor | `T2001` | `tutor123` | Dr. Sara Haddad, Computer Studies |
| Tutor | `T2002` | `tutor456` | Mr. Khalid Al-Rawahi, Business Studies |
"""
    )
