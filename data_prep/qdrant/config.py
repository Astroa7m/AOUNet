import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from common.logger_config import get_logger
q_a_collection_name = "q_a_aou_data"
pdf_collection_name = "pdf_aou_data"
model_name = "multi-qa-MiniLM-L6-cos-v1"

logger = get_logger("QUADRANT_CONFIG")
load_dotenv()
# automatically detects local vs cloud
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

# initializing embedding model (singleton pattern)
_embedding_model = None
_qdrant_client = None

def is_local_qdrant():
    """Check if we're using local Qdrant instance"""
    return "localhost" in QDRANT_URL or "127.0.0.1" in QDRANT_URL


def get_embedding_model():
    """Get or create the embedding model singleton"""
    global _embedding_model
    if _embedding_model is None:
        logger.debug("creating new embedding model instance")
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


def get_qdrant_client():
    """Get Qdrant client instance (singleton)"""
    global _qdrant_client
    if _qdrant_client is None:
        if is_local_qdrant():
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
                timeout=120
            )
    return _qdrant_client


def get_embedding_function():
    """
    Get embedding function that returns a list of embeddings.
    """
    model = get_embedding_model()
    return lambda texts: model.encode(texts, convert_to_numpy=True).tolist()


def ensure_collection(collection_name):
    """Create collection if it doesn't exist and return client"""
    client = get_qdrant_client()

    collections = client.get_collections().collections
    collection_exists = any(col.name == collection_name for col in collections)

    if not collection_exists:
        # get vector size from model
        model = get_embedding_model()
        vector_size = model.get_sentence_embedding_dimension()

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


def get_collection(collection_name):
    """Get collection by name"""
    return ensure_collection(collection_name)


def query_all_collections(query_text, n_results=5):
    """Query both Q&A and PDF collections"""
    client = get_qdrant_client()
    embed_fn = get_embedding_function()

    # Generate query embedding
    query_vector = embed_fn([query_text])[0]

    results = []

    for collection_name in [q_a_collection_name, pdf_collection_name]:
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=n_results,
        )
        documents = [hit.payload.get("document", "") for hit in search_results]
        results.extend(documents)

    return results
