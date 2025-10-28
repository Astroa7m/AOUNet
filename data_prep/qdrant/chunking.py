import json
import time
import uuid

from qdrant_client.models import PointStruct
from tqdm import tqdm

from config import *

general_data_input_file = "../../data/json/general_aou_rag_data_cleaned.json"
pdf_markdown_json_input_file_path = "../../data/mds_chunked/chunks.jsonl"
tutors_modules_aou_convo_input_file_path = "../../data/json/tutors_modules_aou_convo_rag_data_cleaned.jsonl"

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

def convo_chunking(
        collection_name=conversation_collection_name,
        input_file_path=tutors_modules_aou_convo_input_file_path
):
    """
    Load a conversations JSONL file and embed each full conversation as one record.
    Each entry in the file should have a "conversation" field (list of {role, content} dicts).
    """

    logger.debug(f"Started conversation chunking for {input_file_path}")

    client = get_conversation_collection()
    embed_fn = get_embedding_function()

    # load conversation records
    conversations = []
    with open(input_file_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            conversations.append(record)

    logger.debug(f"Loaded {len(conversations)} conversation entries from {input_file_path}")

    # prepare documents and metadata
    documents = []
    ids = []
    metadata = []

    for idx, convo in enumerate(conversations):
        # combine full conversation into one text block
        convo_text = "\n".join([f"{m['role']}: {m['content']}" for m in convo])
        documents.append(convo_text)
        ids.append(str(uuid.uuid4()))
        metadata.append({"record_id": idx, "source": "conversation_dataset", "type": "dialogue"})

    total_batches = range(0, len(documents), batch_size)

    for i in tqdm(total_batches, desc="Ingesting conversation records", unit="batch"):
        batch_docs = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_metadata = metadata[i:i + batch_size]

        # generate embeddings for batch
        embeddings = embed_fn(batch_docs)

        # create Qdrant points
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

        # upsert batch
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
    logger.debug(f"✅ Done — added total {count} conversation documents to collection.")


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


if __name__ == "__main__":
    # general_data_chunk()
    # pdf_markdown_json_chunking()
    convo_chunking()
    # result = query_all_collections("who is Ahmed Samir?")
    # print(f"got {len(result)} results")
    # for r in result:
    #     print(r)
