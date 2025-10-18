import io
import os
from PIL import Image as PILImage

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="openai/gpt-oss-120b", api_key=api_key)


def visualize_graph(graph):
    bytes = graph.get_graph().draw_mermaid_png()
    img_stream = io.BytesIO(bytes)

    img = PILImage.open(img_stream)
    img.show()