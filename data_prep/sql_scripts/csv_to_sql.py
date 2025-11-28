import os
import re
import pandas as pd
from typing import List, Dict, Optional
from dotenv import load_dotenv
import supabase


class DatabaseMigrator:
    """Migrates CSV data to normalized Supabase database"""

    MODULES_CSV = "../../data/csv/modules.csv"
    TUTORS_CSV = "../../data/csv/tutors.csv"

    def __init__(self):
        """Initialize Supabase client"""
        load_dotenv()
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")

        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY are required")

        self.client = supabase.create_client(self.url, self.key)
        print("✓ Connected to Supabase")

    def parse_teaching_modules(self, teaching_text: str) -> List[str]:
        """Extract module codes from teaching text"""
        if pd.isna(teaching_text) or not teaching_text:
            return []

        # Pattern: TM351, TU170, etc.
        pattern = r'\b[A-Z]{2}\d{3}\b'
        modules = re.findall(pattern, teaching_text)
        return list(set(modules))  # Remove duplicates

    def parse_experience(self, exp_text: str) -> List[Dict]:
        """Parse experience text into structured data"""
        if pd.isna(exp_text) or not exp_text:
            return []

        experiences = []
        lines = exp_text.split('\n')

        current_exp = {}
        for line in lines:
            line = line.strip()
            if not line:
                if current_exp:
                    experiences.append(current_exp)
                    current_exp = {}
                continue

            # Check if line contains dates (format: "Month YYYY  Month YYYY" or "To date")
            if '|' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 2:
                    current_exp['position'] = parts[0]
                    current_exp['organization'] = parts[1]
                    if len(parts) >= 3:
                        current_exp['location'] = parts[2]
            elif any(month in line for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
                # This is a date line
                dates = line.split()
                if len(dates) >= 2:
                    current_exp['start_date'] = f"{dates[0]} {dates[1]}"
                    if len(dates) >= 4:
                        current_exp['end_date'] = f"{dates[2]} {dates[3]}"
                    elif 'date' in line.lower():
                        current_exp['end_date'] = "To date"

        if current_exp:
            experiences.append(current_exp)

        return experiences

    def parse_publications(self, pub_text: str) -> List[Dict]:
        """Parse publications text into structured data"""
        if pd.isna(pub_text) or not pub_text:
            return []

        publications = []
        sections = pub_text.split('\n\n')

        for section in sections:
            if not section.strip():
                continue

            lines = section.split('\n', 1)
            if len(lines) < 2:
                continue

            header = lines[0].strip()
            content = lines[1].strip() if len(lines) > 1 else ""

            pub_type = header.replace(':', '').strip()

            # Extract year range if present
            year_match = re.search(r'\((\d{4}[–−-]\d{4})\)', content)
            year = year_match.group(1) if year_match else None

            publications.append({
                'publication_type': pub_type,
                'title': None,
                'year': year,
                'venue': None,
                'details': content
            })

        return publications

    def parse_objectives_outcomes(self, text: str) -> List[str]:
        """Parse objectives/outcomes text into list"""
        if pd.isna(text) or not text:
            return []

        # Split by "To" at the beginning of sentences
        items = re.split(r'(?:^|\s)To\s+', text)
        items = [item.strip() for item in items if item.strip()]

        # Clean up items
        cleaned = []
        for item in items:
            # Remove extra whitespace and quotes
            item = re.sub(r'\s+', ' ', item).strip().strip('"')
            if item:
                cleaned.append('To ' + item)

        return cleaned if cleaned else [text]

    def migrate_modules(self):
        """Migrate modules.csv to normalized tables"""
        print("\n📚 Migrating modules...")

        try:
            df = pd.read_csv(self.MODULES_CSV)
            print(f"  Found {len(df)} modules")

            for idx, row in df.iterrows():
                # Insert module
                module_data = {
                    'course_code': row['course_code'],
                    'course_title': row['course_title'],
                    'credit_hours': int(row['credit_hours']) if pd.notna(row['credit_hours']) else None,
                    'course_desc': row['course_desc'] if pd.notna(row['course_desc']) else None,
                    'prerequisite': row['pre-requisite'] if pd.notna(row['pre-requisite']) else None
                }

                result = self.client.table('modules').upsert(module_data).execute()
                module_id = result.data[0]['module_id']

                # Insert objectives
                objectives = self.parse_objectives_outcomes(row.get('course_objectives', ''))
                for seq, objective in enumerate(objectives, 1):
                    self.client.table('course_objectives').insert({
                        'module_id': module_id,
                        'objective_text': objective,
                        'sequence_order': seq
                    }).execute()

                # Insert outcomes
                outcomes = self.parse_objectives_outcomes(row.get('course_outcomes', ''))
                for seq, outcome in enumerate(outcomes, 1):
                    self.client.table('course_outcomes').insert({
                        'module_id': module_id,
                        'outcome_text': outcome,
                        'sequence_order': seq
                    }).execute()

                print(f"  ✓ Migrated: {row['course_code']} - {row['course_title']}")

            print(f"✓ Modules migration complete: {len(df)} modules")

        except Exception as e:
            print(f"✗ Error migrating modules: {str(e)}")
            raise

    def migrate_tutors(self):
        """Migrate tutors.csv to normalized tables"""
        print("\n👨‍🏫 Migrating tutors...")

        try:
            df = pd.read_csv(self.TUTORS_CSV)
            print(f"  Found {len(df)} tutors")

            for idx, row in df.iterrows():
                # Insert tutor
                tutor_data = {
                    'name': row['name'],
                    'title': row['title'] if pd.notna(row['title']) else None,
                    'email': row['email'],
                    'specialization': row['specialization'] if pd.notna(row['specialization']) else None,
                    'phone': row['phone'] if pd.notna(row['phone']) else None,
                    'office': row['office'] if pd.notna(row['office']) else None,
                    'faculty': row['faculty'] if pd.notna(row['faculty']) else None,
                    'biography': row['biography'] if pd.notna(row['biography']) else None
                }

                result = self.client.table('tutors').upsert(tutor_data).execute()
                tutor_id = result.data[0]['tutor_id']

                # Insert URLs
                urls = [
                    ('google_scholar', row.get('google scholar url')),
                    ('research_gate', row.get('research gate url')),
                    ('profile', row.get('profile url'))
                ]

                for url_type, url in urls:
                    if pd.notna(url) and url:
                        self.client.table('tutor_urls').insert({
                            'tutor_id': tutor_id,
                            'url_type': url_type,
                            'url': url
                        }).execute()

                # Insert experience
                experiences = self.parse_experience(row.get('experience', ''))
                for seq, exp in enumerate(experiences, 1):
                    self.client.table('tutor_experience').insert({
                        'tutor_id': tutor_id,
                        'position': exp.get('position'),
                        'organization': exp.get('organization'),
                        'location': exp.get('location'),
                        'start_date': exp.get('start_date'),
                        'end_date': exp.get('end_date'),
                        'sequence_order': seq
                    }).execute()

                # Insert publications
                publications = self.parse_publications(row.get('publications', ''))
                for pub in publications:
                    self.client.table('tutor_publications').insert({
                        'tutor_id': tutor_id,
                        'publication_type': pub['publication_type'],
                        'title': pub['title'],
                        'year': pub['year'],
                        'venue': pub['venue'],
                        'details': pub['details']
                    }).execute()

                # Link tutor to modules they teach
                module_codes = self.parse_teaching_modules(row.get('teaching', ''))
                for code in module_codes:
                    try:
                        # Find module by code
                        module = self.client.table('modules').select('module_id').eq('course_code', code).execute()
                        if module.data:
                            self.client.table('tutor_teaching').insert({
                                'tutor_id': tutor_id,
                                'module_id': module.data[0]['module_id']
                            }).execute()
                    except Exception as e:
                        print(f"    Warning: Could not link {row['name']} to module {code}: {str(e)}")

                print(f"  ✓ Migrated: {row['name']}")

            print(f"✓ Tutors migration complete: {len(df)} tutors")

        except Exception as e:
            print(f"✗ Error migrating tutors: {str(e)}")
            raise

    def run_migration(self):
        """Run complete migration"""
        print("=" * 60)
        print("🚀 Starting Database Migration")
        print("=" * 60)

        try:
            # Migrate modules first (tutors reference modules)
            self.migrate_modules()

            # Then migrate tutors
            self.migrate_tutors()

            print("\n" + "=" * 60)
            print("✓ Migration completed successfully!")
            print("=" * 60)

        except Exception as e:
            print("\n" + "=" * 60)
            print(f"✗ Migration failed: {str(e)}")
            print("=" * 60)
            raise


def main():
    """Main execution"""
    migrator = DatabaseMigrator()
    migrator.run_migration()


if __name__ == "__main__":
    main()