import io
import os
import sys
from PIL import Image as PILImage

import groq
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph.state import CompiledStateGraph

load_dotenv()

# Groq retires models from time to time (llama-3.1-8b-instant is gone), so every name can be changed from .env
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
# the free tier limits tokens per day per model, each fallback model has its own quota.
# qwen/qwen3.8-27b is not a default: with no tools left it typed fake tool calls as its answer
FALLBACK_MODELS = [m.strip() for m in os.getenv("FALLBACK_MODELS", "openai/gpt-oss-120b").split(",") if m.strip()]
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "openai/gpt-oss-120b")

api_key1 = os.getenv("GROQ_API_KEY1")
# max_retries=0: on a 429 the groq library waits up to 60 s before retrying by itself. The fallback chain
# and ModelRetryMiddleware already handle that, so hand over to the next model at once
llm = ChatGroq(model=LLM_MODEL, api_key=api_key1, max_retries=0)
fallback_llms = [ChatGroq(model=model, api_key=api_key1, max_retries=0) for model in FALLBACK_MODELS]


def _usable_second_key() -> str:
    """GROQ_API_KEY2 spreads the load over a second account. A revoked one used to crash every long chat
    (only the summarizer uses it), so check it once with a free call and fall back to key 1."""
    key2 = os.getenv("GROQ_API_KEY2")
    if not key2 or key2 == api_key1:
        return api_key1
    try:
        groq.Groq(api_key=key2, max_retries=0, timeout=10).models.list()  # lists models, costs no tokens
        return key2
    except groq.AuthenticationError:
        print("GROQ_API_KEY2 is invalid or revoked, using only GROQ_API_KEY1", file=sys.stderr, flush=True)
        return api_key1
    except Exception:
        return key2  # network hiccup at startup: keep the key, it is probably fine


api_key2 = _usable_second_key()
if api_key2 != api_key1:
    # Groq counts quotas per account, so a key from a second account doubles the free daily budget.
    # Its models come after key 1's, and only help if the key really is from another account.
    fallback_llms += [ChatGroq(model=model, api_key=api_key2, max_retries=0) for model in [LLM_MODEL, *FALLBACK_MODELS]]
    print(f"GROQ_API_KEY2 is valid, model chain has {1 + len(fallback_llms)} entries", file=sys.stderr, flush=True)
summarization_model = ChatGroq(model=SUMMARY_MODEL, api_key=api_key2, max_retries=0)


def visualize_graph(graph: CompiledStateGraph):
    bytes = graph.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)
    img_stream = io.BytesIO(bytes)

    img = PILImage.open(img_stream)
    img.show()