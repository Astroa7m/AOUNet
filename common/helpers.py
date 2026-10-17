import io
import os
from PIL import Image as PILImage

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph.state import CompiledStateGraph

load_dotenv()

# Groq retires models from time to time (llama-3.1-8b-instant is gone), so every name can be changed from .env
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
# the free tier limits tokens per day per model, each fallback model has its own quota
FALLBACK_MODELS = [m.strip() for m in os.getenv("FALLBACK_MODELS", "openai/gpt-oss-120b,qwen/qwen3.8-27b").split(",") if m.strip()]
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "openai/gpt-oss-120b")

api_key1 = os.getenv("GROQ_API_KEY1")
# max_retries=1: a daily quota error says "try again in 5 minutes", waiting is pointless, the fallback model answers instead
llm = ChatGroq(model=LLM_MODEL, api_key=api_key1, max_retries=1)
fallback_llms = [ChatGroq(model=model, api_key=api_key1, max_retries=1) for model in FALLBACK_MODELS]
# a second key spreads the rate limit, one free key is enough to run the app
api_key2 = os.getenv("GROQ_API_KEY2") or api_key1
summarization_model = ChatGroq(model=SUMMARY_MODEL, api_key=api_key2, max_retries=1)


def visualize_graph(graph: CompiledStateGraph):
    bytes = graph.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)
    img_stream = io.BytesIO(bytes)

    img = PILImage.open(img_stream)
    img.show()