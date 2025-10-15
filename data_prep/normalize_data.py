import json
import re
import unicodedata
from typing import List, Dict, Set


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    # normalizing unicode characters, removing accents using NFKD
    text = unicodedata.normalize('NFKD', text)

    # removing non-printables
    text = ''.join(char for char in text if char.isprintable() or char in ' \t\n\r')

    # replacing common problematic characters that might result from encoding issues
    replacements = {
        '\x96': '-',  # Smart dash –
        '\x97': '--',  # Long dash —
        '\x91': "'",  # Left single quotation mark ‘
        '\x92': "'",  # Right single quotation mark ’
        '\x93': '"',  # Left double quotation mark “
        '\x94': '"',  # Right double quotation mark ”
        '\x95': '•',  # Bullet •
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # removing control characters (except common ones like newline, tab)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

    # removing multiple consecutive whitespace characters and replace with single space
    text = re.sub(r'\s+', ' ', text)

    text = text.strip()

    return text


def deduplicate_data(data: List[Dict[str, str]]) -> List[Dict[str, str]]:

    seen: Set[str] = set()
    unique_data = []

    for item in data:
        # Create a unique key based on both prompt and completion
        key = f"{item['prompt']}|{item['completion']}"

        if key not in seen:
            seen.add(key)
            unique_data.append(item)

    return unique_data


def normalize_dataset(file_path: str, output_path: str = None) -> None:

    if output_path is None:
        output_path = file_path.replace('.json', '_cleaned.json')

    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Original data contains {len(data)} entries.")

    print("Cleaning text...")
    cleaned_data = []
    for item in data:
        cleaned_item = {
            'prompt': clean_text(item.get('prompt', '')),
            'completion': clean_text(item.get('completion', ''))
        }
        # only add items if both prompt and completion are not empty after cleaning
        if cleaned_item['prompt'] and cleaned_item['completion']:
            cleaned_data.append(cleaned_item)

    print(f"Data after cleaning contains {len(cleaned_data)} entries.")

    print("Removing duplicates...")
    deduplicated_data = deduplicate_data(cleaned_data)

    print(f"Data after deduplication contains {len(deduplicated_data)} entries.")

    print(f"Saving cleaned data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(deduplicated_data, f, ensure_ascii=False, indent=2)

    print(f"Normalization complete. Saved {len(deduplicated_data)} entries to {output_path}")


if __name__ == "__main__":
    normalize_dataset('../data/aou_training_dataset.json')
