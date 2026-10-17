import atexit
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchText, VectorParams

from common import *
from common.logger_config import get_logger
from data_prep.qdrant import get_embedding_model

logger = get_logger("QUADRANT_CONFIG")
load_dotenv()
# VECTOR_STORE=embedded (default): the index is plain files in vector_store/ shipped with the repo.
# Free, no server, nothing to wake up, works on Streamlit Cloud.
# VECTOR_STORE=server: use the Qdrant server at QDRANT_URL (local Docker or Qdrant Cloud).
VECTOR_STORE = os.getenv("VECTOR_STORE", "embedded").strip().lower()
QDRANT_URL = os.getenv("QDRANT_URL") or "http://localhost:6333"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
VECTOR_STORE_PATH = Path(__file__).resolve().parents[2] / "vector_store"

# initializing embedding model (singleton pattern)
_embedding_model = None
_qdrant_client = None

def is_embedded_qdrant():
    """True when the index lives in vector_store/ instead of a Qdrant server"""
    return VECTOR_STORE != "server"


def is_local_qdrant():
    """Check if we're using local Qdrant instance"""
    return is_embedded_qdrant() or "localhost" in QDRANT_URL or "127.0.0.1" in QDRANT_URL


# def get_embedding_model():
#     """Get or create the embedding model singleton"""
#     global _embedding_model
#     if _embedding_model is None:
#         logger.debug("creating new embedding model instance")
#         _embedding_model = SentenceTransformer(model_name)
#     return _embedding_model


def get_qdrant_client():
    """Get Qdrant client instance (singleton)"""
    global _qdrant_client
    if _qdrant_client is None:
        if is_embedded_qdrant():
            logger.debug(f"Using embedded Qdrant at {VECTOR_STORE_PATH}")
            _qdrant_client = QdrantClient(path=str(VECTOR_STORE_PATH))
            # release the folder lock before Python tears modules down, otherwise it prints a traceback on exit
            atexit.register(_qdrant_client.close)
        elif is_local_qdrant():
            # Local mode - no API key needed
            logger.debug("Using Qdrant locally")
            _qdrant_client = QdrantClient(url=QDRANT_URL)
        else:
            # Cloud mode, requires API key
            logger.debug("Using Qdrant remotely")
            if not QDRANT_API_KEY:
                raise ValueError(
                    "QDRANT_API_KEY environment variable is required for cloud deployment. "
                )
            _qdrant_client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                timeout=20
            )
    return _qdrant_client


def get_embedding_function():
    """
    Get embedding function that returns a list of embeddings.
    """
    model = get_embedding_model()
    return lambda texts: [vector.tolist() for vector in model.embed(texts)]


def ensure_collection(collection_name):
    """Create collection if it doesn't exist and return client"""
    client = get_qdrant_client()

    collections = client.get_collections().collections
    collection_exists = any(col.name == collection_name for col in collections)

    if not collection_exists:
        # get vector size from model
        vector_size = len(get_embedding_function()(["dimension probe"])[0])

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        print(f"Created collection '{collection_name}' in {'local' if is_local_qdrant() else 'cloud'} Qdrant")

    return client


def get_q_a_collection():
    """Get Q&A collection client"""
    return ensure_collection(q_a_collection_name)


def get_pdf_collection():
    """Get PDF collection client"""
    return ensure_collection(pdf_collection_name)


def get_csv_collection(collection_name):
    """Get csv collection based on name client"""
    return ensure_collection(collection_name)


_kb_available = None


def knowledge_base_available() -> bool:
    """True when Qdrant answers and has the collections the agent searches. Checked once per process."""
    global _kb_available
    if _kb_available is None and is_embedded_qdrant() and not VECTOR_STORE_PATH.exists():
        logger.info("No vector_store/ folder, knowledge base disabled, using web search only")
        _kb_available = False
    if _kb_available is None:
        try:
            existing = {c.name for c in get_qdrant_client().get_collections().collections}
            needed = {tutors_collection_name, modules_collection_name, q_a_collection_name}
            _kb_available = needed <= existing
            if not _kb_available:
                logger.warning(f"Qdrant is missing collections {needed - existing}, knowledge base disabled")
        except Exception as e:
            logger.warning(f"Qdrant not reachable, knowledge base disabled: {e}")
            _kb_available = False
    return _kb_available


def query_all_collections(query_text, n_results=5):
    """Query both Q&A and PDF collections"""
    client = get_qdrant_client()
    embed_fn = get_embedding_function()

    # Generate query embedding
    query_vector = embed_fn([query_text])[0]

    # embeddings match meaning, not exact codes, so documents that literally contain a module code go first
    module_codes = sorted(set(re.findall(r"\b[A-Z]{1,3}\d{3}[A-Z]?\b", query_text.upper())))

    results = []
    collections = [tutors_collection_name, modules_collection_name, q_a_collection_name]
    for collection_name in collections:
            documents = []
            for code in module_codes:
                exact_hits, _ = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=Filter(must=[FieldCondition(key="document", match=MatchText(text=code))]),
                    limit=3,
                )
                documents.extend(hit.payload.get("document", "") for hit in exact_hits)

            # client.search() was removed in qdrant-client 1.16, query_points() replaces it
            search_results = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=n_results,
            ).points
            documents.extend(hit.payload.get("document", "") for hit in search_results)
            # drop duplicates, keep order, keep the same total size as before
            results.extend(list(dict.fromkeys(documents))[:n_results])

    return results


if __name__ == '__main__':
    # query = "Who is Abrar?"
    # result = query_all_collections(query)
    # print("="*30, query, "="*30)
    # for row in result:
    #     print(row)
    # print("="*60)
    # query = "Who teaches TM354??"
    # result = query_all_collections(query)
    # print("=" * 30, query, "=" * 30)
    # for row in result:
    #     print(row)
    # query = "how can I apply at aou?"
    # result = query_all_collections(query)
    # print("=" * 30, query, "=" * 30)
    # for row in result:
    #     print(row)
    # query = "who teaches tm354?"
    # print("query modules")
    # res = query_all_collections(query)
    # for doc in res:
    #     print(doc)
    # print("query tutors")
    # res = query_all_collections(query)
    # for doc in res:
    #     print(doc)

    query = input("query: ")

    while query != "exit":
        answers = query_all_collections(query)
        for r in answers:
            print(r)
            print("/"*300)
        query = input("query: ")