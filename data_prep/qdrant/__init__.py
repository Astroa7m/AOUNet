from sentence_transformers import SentenceTransformer

from data_prep.legacy.chunking import logger
model_name = "multi-qa-MiniLM-L6-cos-v1"
_embedding_model = None

def get_embedding_model():
    """Get or create the embedding model singleton"""
    global _embedding_model
    if _embedding_model is None:
        logger.debug("creating new embedding model instance")
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model

get_embedding_model()