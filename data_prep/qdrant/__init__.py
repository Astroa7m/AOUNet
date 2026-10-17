from fastembed import TextEmbedding

from common.logger_config import get_logger

logger = get_logger(__name__)
# fastembed runs on ONNX Runtime, so no PyTorch: the install stays small enough for Streamlit Cloud.
# Multilingual, so Arabic questions find English documents too. Changing it means rebuilding vector_store/.
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_embedding_model = None

def get_embedding_model() -> TextEmbedding:
    """Get or create the embedding model singleton"""
    global _embedding_model
    if _embedding_model is None:
        logger.debug("creating new embedding model instance")
        _embedding_model = TextEmbedding(model_name)
    return _embedding_model
