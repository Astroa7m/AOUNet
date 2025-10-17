import io
import json
import os
import re
from typing import Dict, List
from typing import Set, Iterable

import tiktoken
import unicodedata
from tqdm import tqdm


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


def read_text_file(path: str, encoding: str = "utf-8") -> str:
    """
    Read a text file safely and return its content as a string.
    Handles common BOM markers and returns an empty string if file not found.
    """
    if not os.path.isfile(path):
        return ""
    with io.open(path, "r", encoding=encoding, errors="replace") as f:
        text = f.read()
    # strip Byte Order Mark if present
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    return text


def write_jsonl(items: Iterable[Dict], out_path: str, encoding: str = "utf-8") -> None:
    """
    Write an iterable of dicts to a JSONL file (one JSON object per line).
    Ensures atomic write by writing to a temp file then renaming.
    """
    tmp_path = out_path + ".tmp"
    with io.open(tmp_path, "w", encoding=encoding) as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(tmp_path, out_path)


def write_json(obj: object, out_path: str, indent: int = 2, encoding: str = "utf-8") -> None:
    """
    Write an object to a JSON file with atomic replace.
    """
    tmp_path = out_path + ".tmp"
    with io.open(tmp_path, "w", encoding=encoding) as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    os.replace(tmp_path, out_path)


def normalize_markdown(md_text: str, keep_code_blocks: bool = False) -> Dict[str, str]:
    """
    Normalize markdown text into a semantically clean version for embedding.

    Returns a dict with:
      - 'text': cleaned plain text
      - 'metadata': minimal info extracted (like code count, link count)
    """

    # ----------------------------
    # 1. Normalize line endings and strip BOM if any
    # ----------------------------
    text = md_text.replace('\r\n', '\n').replace('\r', '\n').lstrip('\ufeff')

    # ----------------------------
    # 2. Remove HTML tags (if markdown was generated from HTML)
    # ----------------------------
    text = re.sub(r'<[^>]+>', '', text)

    # ----------------------------
    # 3. Handle code blocks
    # ----------------------------
    code_blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
    code_block_count = len(code_blocks)

    if not keep_code_blocks:
        # Replace code blocks entirely with a placeholder (helps preserve context)
        text = re.sub(r'```.*?```', '[CODE BLOCK]', text, flags=re.DOTALL)
    else:
        # Or normalize inline (remove ```lang and keep code content)
        text = re.sub(r'```[a-zA-Z]*\n', '', text)
        text = text.replace('```', '\n')

    # ----------------------------
    # 4. Replace markdown headers with semantic labels
    # ----------------------------
    def header_replacer(match):
        hashes = match.group(1)
        header_text = match.group(2).strip()
        level = len(hashes)
        return f"\n{'#' * level} {header_text.upper()}\n"

    text = re.sub(r'^(#{1,6})\s*(.*)$', header_replacer, text, flags=re.MULTILINE)

    # ----------------------------
    # 5. Normalize bold, italic, and strikethrough
    # ----------------------------
    # bold (**text** or __text__)
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    # italic (*text* or _text_)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
    # strikethrough (~~text~~)
    text = re.sub(r'~~(.*?)~~', r'\1', text)

    # ----------------------------
    # 6. Normalize lists and blockquotes
    # ----------------------------
    # lists (-, *, +, or numbered)
    text = re.sub(r'^\s*[-*+]\s+', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '• ', text, flags=re.MULTILINE)
    # blockquotes (> ...)
    text = re.sub(r'^\s*>\s?', '', text, flags=re.MULTILINE)

    # ----------------------------
    # 7. Normalize links and images
    # ----------------------------
    # images ![alt](url)
    image_count = len(re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', text))
    text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'\1', text)

    # links [text](url) → text (url)
    link_count = len(re.findall(r'\[([^\]]+)\]\(([^)]+)\)', text))
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', text)

    # ----------------------------
    # 8. Remove residual markdown characters (like inline code `code`)
    # ----------------------------
    text = re.sub(r'`([^`]*)`', r'\1', text)

    # ----------------------------
    # 9. Cleanup whitespace
    # ----------------------------
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    # ----------------------------
    # 10. Return with metadata
    # ----------------------------
    metadata = {
        "code_blocks": code_block_count,
        "images": image_count,
        "links": link_count,
    }

    return {"text": text, "metadata": metadata}


def chunk_section_text(section: Dict[str, str], max_tokens: int = 500, overlap: int = 50) -> List[Dict[str, str]]:
    """
    Split a markdown section into smaller overlapping chunks.

    Args:
        section: A dict with keys: 'title', 'content', 'level'.
        max_tokens: Maximum tokens per chunk (approx).
        overlap: Number of tokens overlapping between chunks.

    Returns:
        List of chunk dicts, each containing title, chunk_id, and content.
    """

    enc = tiktoken.get_encoding("cl100k_base")  # same tokenizer used in OpenAI models
    tokens = enc.encode(section["content"])
    total_tokens = len(tokens)

    chunks = []
    start = 0
    chunk_index = 1

    while start < total_tokens:
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)

        chunk = {
            "title": section["title"],
            "level": section["level"],
            "chunk_id": f"{section['title']}_{chunk_index}",
            "content": chunk_text.strip()
        }
        chunks.append(chunk)

        start = end - overlap  # shift start by (max_tokens - overlap)
        chunk_index += 1

    return chunks


def segment_markdown_sections(normalized_text: str) -> List[Dict[str, str]]:
    """
    Segment normalized markdown into semantically meaningful sections.
    Splits based on header hierarchy (#, ##, ###, etc.).

    Returns a list of dicts:
    [
      {"title": "INSTALLATION", "level": 1, "content": "..."},
      {"title": "TROUBLESHOOTING", "level": 2, "content": "..."},
      ...
    ]
    """

    # ----------------------------
    # 1. Split by headers (capture headers too)
    # ----------------------------
    pattern = r'^(#{1,6})\s+(.*)$'
    matches = list(re.finditer(pattern, normalized_text, flags=re.MULTILINE))

    sections = []

    # If no headers found, treat entire text as one section
    if not matches:
        return [{"title": "ROOT", "level": 0, "content": normalized_text.strip()}]

    # ----------------------------
    # 2. Loop over headers and slice text between them
    # ----------------------------
    for i, match in enumerate(matches):
        level = len(match.group(1))  # number of '#' symbols
        title = match.group(2).strip()
        start = match.end()  # end position of header
        end = matches[i + 1].start() if i + 1 < len(matches) else len(normalized_text)
        content = normalized_text[start:end].strip()
        sections.append({
            "title": title,
            "level": level,
            "content": content
        })

    return sections

def chunk_all_sections(
    sections: List[Dict[str, str]],
    source_name: str = None,
    max_tokens: int = 500,
    overlap: int = 50
) -> List[Dict[str, str]]:
    """
    Chunk all markdown sections into final embedding-ready units.

    Args:
        sections: Output from segment_markdown_sections().
        source_name: Optional name of the file or document.
        max_tokens: Maximum tokens per chunk.
        overlap: Token overlap between chunks.

    Returns:
        A flat list of chunks with metadata for embedding.
    """
    all_chunks = []

    for section in sections:
        chunks = chunk_section_text(section, max_tokens=max_tokens, overlap=overlap)
        for chunk in chunks:
            chunk_entry = {
                "doc_id": os.path.splitext(os.path.basename(source_name))[0] if source_name else "unknown",
                "section_title": chunk["title"],
                "level": chunk["level"],
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"]
            }
            all_chunks.append(chunk_entry)

    return all_chunks

def process_markdown_folder(folder_path, output_path, max_tokens=500, overlap=50):
    """
    Process all .md files in a folder into embedding-ready chunks.
    """
    all_docs = []

    for filename in tqdm(os.listdir(folder_path), desc="Processing md files for chunking"):
        if not filename.endswith(".md"):
            continue

        file_path = os.path.join(folder_path, filename)

        # 1. Read & clean
        text = read_text_file(file_path)
        text = normalize_markdown(text)

        text = text["text"]

        # 2. Split into sections
        sections = segment_markdown_sections(text)

        # 3. Chunk sections into smaller pieces
        chunks = chunk_all_sections(
            sections,
            source_name=filename,
            max_tokens=max_tokens,
            overlap=overlap
        )

        all_docs.extend(chunks)

    # Save as JSONL (optional)
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"✅ Processed {len(all_docs)} chunks from folder: {folder_path}")
    return all_docs


if __name__ == "__main__":
    process_markdown_folder(r"../data/mds", r"../data/mds_chunked/chunks.jsonl")


