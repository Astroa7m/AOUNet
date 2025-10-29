import json
import uuid

from tqdm import tqdm

from data_prep.legacy.config import *
from common.logger_config import get_logger

q_a_input_file_path = "data/json/general_aou_rag_data_cleaned.json"
pdf_markdown_json_input_file_path = "data/mds_chunked/chunks.jsonl"
batch_size = 100

logger = get_logger(__name__)


def qa_chunk(
        collection_name=q_a_collection_name,
        client_file_name=client_file_name,
        input_file_path=q_a_input_file_path
):
    """Chunks QA data into, one record per chunk"""

    logger.debug(f"Started Q/A chunking")

    collection = get_q_a_collection()

    logger.debug(f"Loading QA data from {input_file_path}")
    with open(input_file_path, encoding="utf-8") as f:
        qa_data = json.load(f)

    documents = []
    ids = []
    metadata = []
    for idx, record in enumerate(qa_data):
        document = f"Question: {record['prompt']} - Answer: {record['completion']}"
        documents.append(document)
        ids.append(f"qa_record_{idx}")
        metadata.append({"record_id": idx, "source": "aou_rag_dataset", "qa_type": "qa_pair"})

    # adding by batches to avoid memory overload when data is large
    total_batches = range(0, len(documents), batch_size)

    for i in tqdm(total_batches, desc="Ingesting QA records", unit="batch"):
        logger.debug(f"Adding batch {i}–{i + batch_size} to collection")
        collection.add(
            documents=documents[i:i + batch_size],
            ids=ids[i:i + batch_size],
            metadatas=metadata[i:i + batch_size]
        )

    count = collection.count()
    logger.debug(f"Done added total {count} documents to collection")


def pdf_markdown_json_chunking(
        client_file_name=client_file_name,
        pdf_markdown_json_input_file_path=pdf_markdown_json_input_file_path
):
    """
        Load Markdown chunk JSONL and embed them in Chroma.
        """
    logger.debug(f"Started Markdown/PDF chunking for {pdf_markdown_json_input_file_path}")

    collection = get_pdf_collection()

    # Load chunks
    chunks = []
    with open(pdf_markdown_json_input_file_path, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            chunks.append(chunk)

    logger.debug(f"Loaded {len(chunks)} chunks from {pdf_markdown_json_input_file_path}")

    # Prepare lists for documents, ids, metadata
    documents = [c['section_title'] + ":\n" + c["content"] for c in chunks]
    ids = [c["chunk_id"] + "-" + str(uuid.uuid4()) for c in chunks]
    metadata = [
        {"source": c["doc_id"], "section_title": c["section_title"], "level": c["level"], "chunk_id": c['chunk_id']} for
        c in chunks]

    # Add by batches
    total_batches = range(0, len(documents), batch_size)
    for i in tqdm(total_batches, desc="Ingesting Markdown chunks", unit="batch"):
        collection.add(
            documents=documents[i:i + batch_size],
            ids=ids[i:i + batch_size],
            metadatas=metadata[i:i + batch_size]
        )

    count = collection.count()
    logger.debug(f"Done added total {count} chunks to collection")


def query(query_text, n_results=5, debug=False):
    collection = get_q_a_collection()
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
    )

    if debug:
        for doc in results['documents']:
            print(doc)

    return results['documents']


def query_all_collections(query_text, n_results=5):

    results = []

    for collection in [get_q_a_collection(), get_pdf_collection()]:
        r = collection.query(query_texts=[query_text], n_results=n_results)
        results.extend(r['documents'][0])  # flatten the single-query list

    return results


if __name__ == "__main__":
    # creating = True
    # qa_chunk() if creating else query("ANy data about ahmed samir?", debug=True)
    # pdf_markdown_json_chunking()
    result = query_all_collections("Hello There")
    print(f"got {len(result)} results")
    for result in result:
        print(result)
