import json
import os.path
from pathlib import Path

from data_prep.pdf_prep import pdfs_to_markdown
from logger_config import get_logger
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm



q_a_collection_name = "q_a_aou_data"
pdf_collection_name = "pdf_aou_data"
pdf_parent_file = "../data/pdfs"
client_file_name = "./chroma_db"
input_file_path = "../data/json/aou_rag_dataset.json"
model_name = "multi-qa-MiniLM-L6-cos-v1"
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

logger = get_logger(__name__)


def qa_chunk(
        collection_name=q_a_collection_name,
        client_file_name=client_file_name,
        input_file_path=input_file_path
):
    """Chunks QA data into, one record per chunk"""

    logger.debug(f"Started Q/A chunking")

    client = chromadb.PersistentClient(client_file_name)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        configuration={
            "hnsw": {
                "space": "cosine",
                "ef_construction": 200,
            }
        }
    )

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
    batch_size = 100
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

def pdf_chunking(
        collection_name=pdf_collection_name,
        client_file_name=client_file_name,
        pdf_parent_dir_path=pdf_parent_file
):
    path = Path(pdf_parent_dir_path)

    pdfs_path = []

    if not path.exists():
        raise Exception("dir not found")
    if not path.is_dir():
        raise Exception("Not a dir")

    def iterate_child_dir(fromdir: Path):
        global original_doc_count

        logger.debug(f"within {os.path.basename(fromdir)}")
        children = fromdir.iterdir()
        for dir in children:
            if dir.is_dir():
                iterate_child_dir(dir)
            else:
                # document detected
                pdfs_path.append(dir)

    iterate_child_dir(path)

    pdf_markdown = []

    for pdf_path in pdfs_path:
        pdf_markdown.append(pdfs_to_markdown(pdf_path))

    pass


def query(query_text, collection_name=q_a_collection_name,
          client_file_name=client_file_name, n_results=5, debug=False):
    client = chromadb.PersistentClient(path=client_file_name)
    collection = client.get_collection(name=collection_name, embedding_function=embedding_function)
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
    )

    if debug:
        for doc in results['documents']:
            print(doc)

    return results['documents']


if __name__ == "__main__":
    # creating = True
    # qa_chunk() if creating else query("ANy debugr about ahmed samir?", debug=True)
    pdf_chunking()
