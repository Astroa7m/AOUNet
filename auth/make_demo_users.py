"""Builds auth/demo_users.json. All people and numbers are fictional. Run: python -m auth.make_demo_users"""
import json, sys
from auth.users import hash_password, DEMO_USERS_PATH

def account(id_, password, **rest):
    salt, h = hash_password(password)
    return {"id": id_, "salt": salt, "password_hash": h, **rest}

week = lambda *rows: [dict(zip(("day", "time", "module", "type", "room"), r)) for r in rows]

data = {
    "_note": "Fictional demo data. Passwords are in the README and on the login page.",
    "modules": {
        "TM354": {"title": "Software Engineering", "credit_hours": 8, "tutor_id": "T2001",
                  "deadlines": {"TMA01": "2026-09-28", "TMA02": "2026-10-16", "MTA": "2026-11-02", "Final exam": "2026-12-20"}},
        "M269": {"title": "Algorithms, Data Structures and Computability", "credit_hours": 8, "tutor_id": "T2001",
                 "deadlines": {"TMA01": "2026-10-05", "TMA02": "2026-10-26", "MTA": "2026-11-09", "Final exam": "2026-12-22"}},
        "TM351": {"title": "Data Management and Analysis", "credit_hours": 8, "tutor_id": "T2002",
                  "deadlines": {"TMA01": "2026-10-01", "TMA02": "2026-10-22", "MTA": "2026-11-05", "Final exam": "2026-12-21"}},
        "B207A": {"title": "Shaping Business Opportunities", "credit_hours": 8, "tutor_id": "T2002",
                  "deadlines": {"TMA01": "2026-10-08", "TMA02": "2026-10-29", "MTA": "2026-11-12", "Final exam": "2026-12-23"}},
    },
    "students": [
        account("S1001", "student123",
            name="Layla Al-Harthy", email="s1001@demo.aou.edu.om",
            contact={"phone": "+968 9100 0001", "personal_email": "layla.demo@example.com", "address": "Al Khuwair, Muscat", "emergency_contact": "Mother, +968 9100 0002"},
            inbox=[
                {"from": "t2001@demo.aou.edu.om", "subject": "Welcome to TM354",
                 "date": "2026-09-10", "body": "Dear Layla,\n\nWelcome to TM354 Software Engineering. My office hours are Sunday and Monday, 12:00-14:00 in Building B, Office 12. Feel free to email me with any questions.\n\nDr. Sara Haddad"},
            ],
            program="BSc Information Technology and Computing", level=3, gpa=3.42, completed_credits=72,
            fees={"balance_omr": 185.5, "due_date": "2026-11-15"},
            enrolled={
                "TM354": {"tma": "17/20", "mta": "24/30", "final": None, "attendance": "92%"},
                "M269": {"tma": "14/20", "mta": "19/30", "final": None, "attendance": "81%"},
                "TM351": {"tma": "18/20", "mta": "27/30", "final": None, "attendance": "97%"},
            },
            schedule=week(
                ("Sunday", "10:00-12:00", "TM354", "Tutorial", "Lab 2"),
                ("Monday", "14:00-16:00", "M269", "Tutorial", "Room 105"),
                ("Tuesday", "10:00-12:00", "TM351", "Tutorial", "Lab 4"),
                ("Wednesday", "16:00-18:00", "TM354", "Online session", "Teams"),
            )),
        account("S1002", "student456",
            name="Omar Al-Busaidi", email="s1002@demo.aou.edu.om",
            contact={"phone": "+968 9200 0001", "personal_email": "omar.demo@example.com", "address": "Seeb, Muscat", "emergency_contact": "Father, +968 9200 0002"},
            inbox=[],
            program="BA Business Studies (Marketing)", level=2, gpa=3.05, completed_credits=48,
            fees={"balance_omr": 0, "due_date": None},
            enrolled={
                "B207A": {"tma": "15/20", "mta": "21/30", "final": None, "attendance": "88%"},
                "TM351": {"tma": "12/20", "mta": "18/30", "final": None, "attendance": "75%"},
            },
            schedule=week(
                ("Monday", "10:00-12:00", "B207A", "Tutorial", "Room 210"),
                ("Tuesday", "10:00-12:00", "TM351", "Tutorial", "Lab 4"),
            )),
    ],
    "tutors": [
        account("T2001", "tutor123",
            title="Dr.", name="Sara Haddad", email="t2001@demo.aou.edu.om",
            contact={"phone": "+968 2400 1201", "bio": "Lecturer in software engineering and algorithms."},
            faculty="Faculty of Computer Studies", office="Building B, Office 12",
            office_hours="Sunday and Monday, 12:00-14:00",
            inbox=[
                {"from": "s1001@demo.aou.edu.om", "subject": "TM354 TMA02 extension request",
                 "date": "2026-10-14", "body": "Dear Dr. Sara, I was ill last week and would like to ask for a 3 day extension for TMA02. I can share the medical report. Kind regards, Layla"},
                {"from": "programme.coordinator@demo.aou.edu.om", "subject": "M269 midterm marks due",
                 "date": "2026-10-12", "body": "Reminder: please upload M269 midterm marks to SIS before 20 October."},
                {"from": "it.helpdesk@demo.aou.edu.om", "subject": "LMS maintenance on Friday",
                 "date": "2026-10-10", "body": "The LMS will be offline on Friday from 08:00 to 12:00 for maintenance."},
            ]),
        account("T2002", "tutor456",
            title="Mr.", name="Khalid Al-Rawahi", email="t2002@demo.aou.edu.om",
            contact={"phone": "+968 2400 0702", "bio": "Lecturer in business studies and data analysis."},
            faculty="Faculty of Business Studies", office="Building A, Office 7",
            office_hours="Tuesday, 12:00-15:00",
            inbox=[
                {"from": "s1002@demo.aou.edu.om", "subject": "Question about B207A group project",
                 "date": "2026-10-13", "body": "Hello Mr. Khalid, can our group have 4 members instead of 3? Thanks, Omar"},
            ]),
    ],
}

with open(DEMO_USERS_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print("wrote", DEMO_USERS_PATH)
