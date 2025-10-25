import io
import os
from PIL import Image as PILImage

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph.state import CompiledStateGraph

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="openai/gpt-oss-20b", api_key=api_key)


def visualize_graph(graph: CompiledStateGraph):
    bytes = graph.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)
    img_stream = io.BytesIO(bytes)

    img = PILImage.open(img_stream)
    img.show()