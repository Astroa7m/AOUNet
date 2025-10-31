import json
import time
import uuid
from typing import Dict, List, Any

import pandas as pd
from qdrant_client.models import PointStruct
from tqdm import tqdm

from config import *

general_data_input_file = "../../data/json/general_aou_rag_data_cleaned.json"
pdf_markdown_json_input_file_path = "../../data/mds_chunked/chunks.jsonl"
tutor_csv_file_path = "../../data/csv/tutors.csv"
modules_csv_file_path = "../../data/csv/modules.csv"

batch_size = 100

logger = get_logger(__name__)


def general_data_chunk(
        collection_name=q_a_collection_name,
        input_file_path=general_data_input_file
):
    """Chunks general data into, one record per chunk"""

    logger.debug(f"Started Q/A chunking")

    client = get_q_a_collection()
    embed_fn = get_embedding_function()

    logger.debug(f"Loading QA data from {input_file_path}")
    with open(input_file_path, encoding="utf-8") as f:
        qa_data = json.load(f)

    documents = []
    ids = []
    metadata = []
    for idx, record in enumerate(qa_data):
        document = f"Question: {record['prompt']} - Answer: {record['completion']}"
        documents.append(document)
        ids.append(str(uuid.uuid4()))
        metadata.append({"record_id": idx, "source": "aou_rag_dataset", "qa_type": "qa_pair"})

    # process by batches to avoid memory overload
    total_batches = range(0, len(documents), batch_size)

    for i in tqdm(total_batches, desc="Ingesting QA records", unit="batch"):
        batch_docs = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_metadata = metadata[i:i + batch_size]

        logger.debug(f"Adding batch {i}–{i + batch_size} to collection")

        # generate embeddings for batch
        embeddings = embed_fn(batch_docs)

        # create points
        points = [
            PointStruct(
                id=batch_ids[j],
                vector=embeddings[j],
                payload={
                    "document": batch_docs[j],
                    **batch_metadata[j]
                }
            )
            for j in range(len(batch_docs))
        ]

        # upsert to Qdrant
        for attempt in range(3):
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                break
            except Exception as e:
                logger.info(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)

    count = client.count(collection_name=collection_name).count
    logger.debug(f"Done added total {count} documents to collection")


def pdf_markdown_json_chunking(
        pdf_markdown_json_input_file_path=pdf_markdown_json_input_file_path
):
    """
    Load Markdown chunk JSONL and embed them in Qdrant.
    """
    logger.debug(f"Started Markdown/PDF chunking for {pdf_markdown_json_input_file_path}")

    client = get_pdf_collection()
    embed_fn = get_embedding_function()

    # Load chunks
    chunks = []
    with open(pdf_markdown_json_input_file_path, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            chunks.append(chunk)

    logger.debug(f"Loaded {len(chunks)} chunks from {pdf_markdown_json_input_file_path}")

    # Prepare documents and metadata
    documents = [c['section_title'] + ":\n" + c["content"] for c in chunks]
    ids = ids = [str(uuid.uuid4()) for _ in range(len(documents))]
    metadata = [
        {
            "source": c["doc_id"],
            "section_title": c["section_title"],
            "level": c["level"],
            "chunk_id": c['chunk_id']
        }
        for c in chunks
    ]

    # Add by batches
    total_batches = range(0, len(documents), batch_size)
    for i in tqdm(total_batches, desc="Ingesting Markdown chunks", unit="batch"):
        batch_docs = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_metadata = metadata[i:i + batch_size]

        # Generate embeddings for batch
        embeddings = embed_fn(batch_docs)

        # Create points
        points = [
            PointStruct(
                id=batch_ids[j],
                vector=embeddings[j],
                payload={
                    "document": batch_docs[j],
                    **batch_metadata[j]
                }
            )
            for j in range(len(batch_docs))
        ]

        # Upsert to Qdrant
        client.upsert(
            collection_name=pdf_collection_name,
            points=points
        )

    count = client.count(collection_name=pdf_collection_name).count
    logger.debug(f"Done added total {count} chunks to collection")



def query(query_text, n_results=5, debug=False):
    """Query the Q&A collection"""
    client = get_qdrant_client()
    embed_fn = get_embedding_function()

    # Generate query embedding
    query_vector = embed_fn([query_text])[0]

    # Search in Qdrant
    results = client.search(
        collection_name=q_a_collection_name,
        query_vector=query_vector,
        limit=n_results,
    )

    documents = [hit.payload.get("document", "") for hit in results]

    if debug:
        for doc in documents:
            print(doc)

    return documents


def create_modules_chunks(module_row: Dict) -> List[Dict[str, Any]]:
        """
        Create semantic chunks from a module record.
        """
        chunks = []

        # Chunk 1: Module Overview
        chunks.append({
            "id": f"{uuid.uuid4()}",
            "type": "overview",
            "text": f"""
            Module: {module_row['course_code']} - {module_row['course_title']}
            Credit Hours: {module_row['credit_hours']}
            Prerequisites: {module_row.get('pre-requisite', 'None')}

            Description:
            {module_row.get('course_desc', '')}
            """.strip(),
            "metadata": {
                "course_code": module_row['course_code'],
                "course_title": module_row['course_title'],
                "credit_hours": module_row['credit_hours'],
                "prerequisite": module_row.get('pre-requisite', ''),
                "chunk_type": "overview"
            }
        })

        # Chunk 2: Objectives
        if pd.notna(module_row.get('course_objectives')):
            chunks.append({
                "id": f"{uuid.uuid4()}",
                "type": "objectives",
                "text": f"""
                {module_row['course_code']} - {module_row['course_title']}

                Course Objectives:
                {module_row['course_objectives']}
                """.strip(),
                "metadata": {
                    "course_code": module_row['course_code'],
                    "course_title": module_row['course_title'],
                    "chunk_type": "objectives"
                }
            })

        # Chunk 3: Learning Outcomes (split by categories)
        if pd.notna(module_row.get('course_outcomes')):
            outcomes_text = module_row['course_outcomes']

            # Main outcomes chunk
            chunks.append({
                "id": f"{uuid.uuid4()}",
                "type": "outcomes",
                "text": f"""
                {module_row['course_code']} - Learning Outcomes

                {outcomes_text}
                """.strip(),
                "metadata": {
                    "course_code": module_row['course_code'],
                    "course_title": module_row['course_title'],
                    "chunk_type": "outcomes"
                }
            })

            # Extract specific outcome categories if present
            outcome_categories = ['Knowledge and understanding', 'Cognitive skills',
                                  'Practical and professional skills', 'Key transferable skills']

            for category in outcome_categories:
                if category.lower() in outcomes_text.lower():
                    chunks.append({
                        "id": f"{uuid.uuid4()}",
                        "type": f"outcomes_{category.lower().replace(' ', '_')}",
                        "text": f"""
                        {module_row['course_code']} - {category}:

                        {outcomes_text}
                        """.strip(),
                        "metadata": {
                            "course_code": module_row['course_code'],
                            "course_title": module_row['course_title'],
                            "outcome_category": category,
                            "chunk_type": "outcomes_category"
                        }
                    })

        # Chunk 4: Quick Reference (for exact matching)
        chunks.append({
            "id": f"{uuid.uuid4()}",
            "type": "reference",
            "text": f"""
            Course Code: {module_row['course_code']}
            Title: {module_row['course_title']}
            Credits: {module_row['credit_hours']}
            Prerequisites: {module_row.get('pre-requisite', 'None')}
            """.strip(),
            "metadata": {
                "course_code": module_row['course_code'],
                "course_title": module_row['course_title'],
                "credit_hours": module_row['credit_hours'],
                "prerequisite": module_row.get('pre-requisite', ''),
                "chunk_type": "reference"
            }
        })

        return chunks
def create_tutors_chunks(tutor_row: Dict) -> List[Dict[str, Any]]:
        """
        Create semantic chunks from a tutor record.
        Each chunk focuses on a specific aspect for better retrieval.
        """
        chunks = []

        # Chunk 1: Overview + Biography
        if pd.notna(tutor_row.get('biography')):
            chunks.append({
                "id": f"{uuid.uuid4()}",
                "type": "biography",
                "text": f"""
                Name: {tutor_row['name']}
                Title: {tutor_row['title']}
                Email: {tutor_row['email']}
                Specialization: {tutor_row['specialization']}
                Faculty: {tutor_row['faculty']}

                Biography:
                {tutor_row['biography']}
                """.strip(),
                "metadata": {
                    "tutor_name": tutor_row['name'],
                    "email": tutor_row['email'],
                    "title": tutor_row['title'],
                    "specialization": tutor_row['specialization'],
                    "faculty": tutor_row['faculty'],
                    "chunk_type": "biography"
                }
            })

        # Chunk 2: Teaching Experience
        if pd.notna(tutor_row.get('teaching')):
            chunks.append({
                "id": f"{uuid.uuid4()}",
                "type": "teaching",
                "text": f"""
                {tutor_row['name']} - Teaching Modules:

                {tutor_row['teaching']}

                Specialization: {tutor_row['specialization']}
                """.strip(),
                "metadata": {
                    "tutor_name": tutor_row['name'],
                    "email": tutor_row['email'],
                    "specialization": tutor_row['specialization'],
                    "chunk_type": "teaching"
                }
            })

        # Chunk 3: Professional Experience
        if pd.notna(tutor_row.get('experience')):
            chunks.append({
                "id": f"{uuid.uuid4()}",
                "type": "experience",
                "text": f"""
                {tutor_row['name']} - Professional Experience:

                {tutor_row['experience']}
                """.strip(),
                "metadata": {
                    "tutor_name": tutor_row['name'],
                    "email": tutor_row['email'],
                    "chunk_type": "experience"
                }
            })

        # Chunk 4: Publications & Research
        if pd.notna(tutor_row.get('publications')):
            chunks.append({
                "id": f"{uuid.uuid4()}",
                "type": "publications",
                "text": f"""
                {tutor_row['name']} - Research & Publications:

                Specialization: {tutor_row['specialization']}

                {tutor_row['publications']}
                """.strip(),
                "metadata": {
                    "tutor_name": tutor_row['name'],
                    "email": tutor_row['email'],
                    "specialization": tutor_row['specialization'],
                    "chunk_type": "publications"
                }
            })

        # Chunk 5: Contact & Quick Reference
        chunks.append({
            "id": f"{uuid.uuid4()}",
            "type": "contact",
            "text": f"""
            Contact Information:
            Name: {tutor_row['name']}
            Title: {tutor_row['title']}
            Email: {tutor_row['email']}
            Phone: {tutor_row.get('phone', 'N/A')}
            Office: {tutor_row.get('office', 'N/A')}
            Specialization: {tutor_row['specialization']}
            Faculty: {tutor_row['faculty']}
            Google Scholar: {tutor_row.get('google scholar url', 'N/A')}
            Research Gate: {tutor_row.get('research gate url', 'N/A')}
            Profile: {tutor_row.get('profile url', 'N/A')}
            """.strip(),
            "metadata": {
                "tutor_name": tutor_row['name'],
                "email": tutor_row['email'],
                "title": tutor_row['title'],
                "phone": tutor_row.get('phone', ''),
                "office": tutor_row.get('office', ''),
                "specialization": tutor_row['specialization'],
                "faculty": tutor_row['faculty'],
                "chunk_type": "contact"
            }
        })

        return chunks

def embed_csv_chunks(
        csv_path: str,
        collection_name: str,
        chunk_fn,
        client_getter,
        batch_size=batch_size
):
    """
    Reads CSV, creates chunks using the provided chunk function, embeds and upserts to Qdrant.
    Works for both modules and tutors datasets.
    """
    logger.debug(f"Started embedding CSV chunks for {csv_path}")

    client = client_getter
    embed_fn = get_embedding_function()

    # Load CSV
    df = pd.read_csv(csv_path)
    all_chunks = []

    logger.debug(f"Loaded {len(df)} rows from {csv_path}")

    # Create chunks for each row
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating chunks"):
        row_dict = row.to_dict()
        chunks = chunk_fn(row_dict)
        all_chunks.extend(chunks)

    logger.debug(f"Created total {len(all_chunks)} chunks from {csv_path}")

    # Prepare documents and metadata
    documents = [chunk["text"] for chunk in all_chunks]
    ids = [chunk["id"] for chunk in all_chunks]
    metadata = [chunk["metadata"] for chunk in all_chunks]

    # Process in batches
    total_batches = range(0, len(documents), batch_size)
    for i in tqdm(total_batches, desc=f"Ingesting {collection_name}", unit="batch"):
        batch_docs = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_metadata = metadata[i:i + batch_size]

        # Generate embeddings
        embeddings = embed_fn(batch_docs)

        # Create Qdrant points
        points = [
            PointStruct(
                id=batch_ids[j],
                vector=embeddings[j],
                payload={
                    "document": batch_docs[j],
                    **batch_metadata[j]
                }
            )
            for j in range(len(batch_docs))
        ]

        # Retry logic (same as general_data_chunk)
        for attempt in range(3):
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)

    count = client.count(collection_name=collection_name).count
    logger.debug(f"✅ Done embedding {count} total chunks to collection: {collection_name}")


if __name__ == "__main__":
    embed_csv_chunks(
        csv_path=modules_csv_file_path,
        collection_name=modules_collection_name,
        chunk_fn=create_modules_chunks,
        client_getter=get_csv_collection(modules_collection_name)
    )

    embed_csv_chunks(
        csv_path=tutor_csv_file_path,
        collection_name=tutors_collection_name,
        chunk_fn=create_tutors_chunks,
        client_getter=get_csv_collection(tutors_collection_name)
    )