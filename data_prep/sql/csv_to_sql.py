import os
import csv
from supabase import create_client
from urllib.parse import urlparse
from dotenv import load_dotenv
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("Set SUPABASE_URL and SUPABASE_KEY environment variables")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

MODULES_CSV = "../../data/csv/modules.csv"
TUTORS_CSV = "../../data/csv/tutors.csv"

def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def normalize_list_field(value):
    """
    Normalize a multi-value field. Accepts semicolon or comma separated lists.
    Returns list of trimmed non-empty strings.
    """
    if value is None:
        return []
    v = value.strip()
    if v == "":
        return []
    sep = ';' if ';' in v else ','
    parts = [p.strip() for p in v.split(sep)]
    return [p for p in parts if p]

def upsert_faculty(name):
    if not name or name.strip() == "":
        return None
    name = name.strip()
    res = supabase.table("faculties").select("faculty_id").eq("faculty_name", name).execute()
    if res.data and len(res.data) > 0:
        return res.data[0]["faculty_id"]
    ins = supabase.table("faculties").insert({"faculty_name": name}).execute()
    return ins.data[0]["faculty_id"]

def bulk_insert_courses(modules):
    """
    Insert all courses first and return mapping course_code -> course_id
    """
    mapping = {}
    for m in modules:
        course_code = (m.get("course_code") or "").strip()
        if not course_code:
            continue
        payload = {
            "course_code": course_code,
            "course_title": (m.get("course_title") or "").strip(),
            "credit_hours": int(m["credit_hours"]) if m.get("credit_hours") and m["credit_hours"].strip().isdigit() else None,
            "course_desc": (m.get("course_desc") or "").strip(),
            "course_objectives": (m.get("course_objectives") or "").strip(),
            "course_outcomes": (m.get("course_outcomes") or "").strip()
        }
        res = supabase.table("courses").insert(payload).execute()
        if res.error:
            sel = supabase.table("courses").select("course_id").eq("course_code", course_code).execute()
            if sel.data and len(sel.data) > 0:
                mapping[course_code] = sel.data[0]["course_id"]
        else:
            mapping[course_code] = res.data[0]["course_id"]
    return mapping

def insert_course_prerequisites(modules, code_to_id):
    for m in modules:
        course_code = (m.get("course_code") or "").strip()
        if not course_code:
            continue
        course_id = code_to_id.get(course_code)
        if not course_id:
            continue
        prereq_field = m.get("pre-requisite") or m.get("pre_requisite") or m.get("pre-requisite") or ""
        prereqs = normalize_list_field(prereq_field)
        for pcode in prereqs:
            prereq_id = code_to_id.get(pcode)
            if not prereq_id:
                sel = supabase.table("courses").select("course_id,course_code").ilike("course_code", pcode).execute()
                if sel.data and len(sel.data) > 0:
                    prereq_id = sel.data[0]["course_id"]
            if prereq_id:
                supabase.table("course_prerequisites").insert({
                    "course_id": course_id,
                    "prerequisite_course_id": prereq_id
                }).execute()

def import_tutors(tutors_rows, code_to_id):
    for t in tutors_rows:
        name = (t.get("name") or "").strip()
        if not name:
            continue
        faculty_name = (t.get("faculty") or "").strip()
        faculty_id = upsert_faculty(faculty_name) if faculty_name else None

        tutor_payload = {
            "name": name,
            "title": (t.get("title") or "").strip(),
            "email": (t.get("email") or "").strip() or None,
            "phone": (t.get("phone") or "").strip() or None,
            "office": (t.get("office") or "").strip() or None,
            "faculty_id": faculty_id,
            "biography": (t.get("biography") or "").strip() or None,
            "profile_url": (t.get("profile url") or t.get("profile_url") or "").strip() or None
        }
        res = supabase.table("tutors").insert(tutor_payload).execute()
        if res.error:
            tutor_id = None
            if tutor_payload["email"]:
                sel = supabase.table("tutors").select("tutor_id").eq("email", tutor_payload["email"]).execute()
                if sel.data and len(sel.data) > 0:
                    tutor_id = sel.data[0]["tutor_id"]
            if not tutor_id:
                sel = supabase.table("tutors").select("tutor_id").ilike("name", name).execute()
                if sel.data and len(sel.data) > 0:
                    tutor_id = sel.data[0]["tutor_id"]
        else:
            tutor_id = res.data[0]["tutor_id"]

        if not tutor_id:
            continue

        specs = normalize_list_field(t.get("specialization") or t.get("specializations") or "")
        for s in specs:
            supabase.table("specializations").insert({
                "tutor_id": tutor_id,
                "specialization_name": s
            }).execute()

        gs = (t.get("google scholar url") or t.get("google_scholar_url") or "").strip()
        rg = (t.get("research gate url") or t.get("research_gate_url") or "").strip()
        if gs:
            supabase.table("tutor_links").insert({
                "tutor_id": tutor_id,
                "platform": "Google Scholar",
                "url": gs
            }).execute()
        if rg:
            supabase.table("tutor_links").insert({
                "tutor_id": tutor_id,
                "platform": "ResearchGate",
                "url": rg
            }).execute()

        teaching_field = t.get("teaching") or ""
        teach_codes = normalize_list_field(teaching_field)
        for code in teach_codes:
            course_id = code_to_id.get(code)
            if not course_id:
                sel = supabase.table("courses").select("course_id").ilike("course_code", code).execute()
                if sel.data and len(sel.data) > 0:
                    course_id = sel.data[0]["course_id"]
            if course_id:
                supabase.table("teaching").insert({
                    "tutor_id": tutor_id,
                    "course_id": course_id
                }).execute()

        experiences = normalize_list_field(t.get("experience") or t.get("experiences") or "")
        for e in experiences:
            supabase.table("experiences").insert({
                "tutor_id": tutor_id,
                "experience_text": e
            }).execute()

        pubs = normalize_list_field(t.get("publications") or "")
        for p in pubs:
            supabase.table("publications").insert({
                "tutor_id": tutor_id,
                "publication_text": p
            }).execute()

def main():
    print("Reading CSV files...")
    modules = read_csv(MODULES_CSV)
    tutors = read_csv(TUTORS_CSV)

    print("Inserting courses...")
    code_to_id = bulk_insert_courses(modules)
    print(f"Inserted {len(code_to_id)} courses")

    print("Inserting course prerequisites...")
    insert_course_prerequisites(modules, code_to_id)

    print("Importing tutors...")
    import_tutors(tutors, code_to_id)

    print("Done.")

if __name__ == "__main__":
    main()
