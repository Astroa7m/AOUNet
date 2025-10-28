from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

q_a_collection_name = "q_a_aou_data"
pdf_collection_name = "pdf_aou_data"
client_file_name = Path(__file__).resolve().parents[0] / "./chroma_db"
model_name = "multi-qa-MiniLM-L6-cos-v1"


def get_chroma_client():
    client = chromadb.PersistentClient(client_file_name)
    return client

def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

def get_q_a_collection():
    return get_collection(q_a_collection_name)

def get_pdf_collection():
    return get_collection(pdf_collection_name)

def get_collection(collection_name):
    return get_chroma_client().get_or_create_collection(
        name=collection_name,
        embedding_function=get_embedding_function(),
        configuration={
            "hnsw": {
                "space": "cosine",
                "ef_construction": 200,
            }
        }
    )