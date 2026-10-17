import streamlit as st

from ui.helpers.session import current_user, demo_session_id
from uni import api as uni_api

user = current_user()
st.title("My profile")
st.caption("Loaded from the AOU university API (a mock for this demo). Your changes are private to this browser session "
           "and the assistant sees them right away.")

if user.role not in ("student", "tutor"):
    st.warning("Log in to see your profile.")
    st.stop()

record = uni_api.get_person(demo_session_id(), user.user_id)
contact = record.get("contact", {})

# read-only academic record
if user.role == "student":
    col1, col2, col3 = st.columns(3)
    col1.metric("GPA", record["gpa"])
    col2.metric("Credits completed", record["completed_credits"])
    col3.metric("Fees due (OMR)", record["fees"]["balance_omr"])
    st.markdown(f"**{record['name']}** · `{record['id']}` · {record['email']}  \n{record['program']}, level {record['level']}")
    st.markdown("**Modules:** " + ", ".join(
        f"{code} ({uni_api.get_person(demo_session_id(), uni_api.get_module(code)['tutor_id'])['title']} "
        f"{uni_api.get_person(demo_session_id(), uni_api.get_module(code)['tutor_id'])['name']})"
        for code in record["enrolled"]
    ))
else:
    st.markdown(f"**{record['title']} {record['name']}** · `{record['id']}` · {record['email']}  \n{record['faculty']}")
    st.markdown("**Teaching:** " + ", ".join(
        f"{code} {uni_api.get_module(code)['title']}" for code in uni_api.tutor_modules(record["id"])
    ))

st.divider()
st.subheader("Edit my details")

labels = {
    "phone": "Phone",
    "personal_email": "Personal email",
    "address": "Address",
    "emergency_contact": "Emergency contact",
    "office": "Office",
    "office_hours": "Office hours",
    "bio": "Short bio",
}
current = {**contact, "office": record.get("office", ""), "office_hours": record.get("office_hours", "")}

with st.form("profile"):
    values = {
        field: (st.text_area if field == "bio" else st.text_input)(labels[field], value=current.get(field, ""))
        for field in uni_api.EDITABLE_FIELDS[user.role]
    }
    if st.form_submit_button("Save changes", type="primary"):
        uni_api.update_profile(demo_session_id(), user.user_id, user.role, values)
        st.success("Saved. Ask the assistant \"what is my phone number?\" to see it use the new data.")
