from typing import List, Dict, Any, Literal

from ddgs import DDGS
from langchain_core.tools import tool

from data_prep.qdrant.config import query_all_collections


@tool
def searching_aou_site(query: str) -> list[dict[str, Any]] | str:
    """
    Search for information related to the user query about AOU Oman.
    This function search the university website for relative information
    :param query: user query
    :return: list of dict with the keys [title, href, body] containing most relevant web searches
    """
    try:
        query1 = f"{query} - site:https://www.arabou.edu.kw/"
        query2 = f"{query} - site:https://www.aou.edu.om/"
        with DDGS() as ddgs:
            results1 = ddgs.text(query1, max_results=5)
            results2 = ddgs.text(query2, max_results=5)
            results = results1 + results2
            return results
    except Exception as e:
        return f"Error using the searching tool: {str(e)}"


@tool
def retrieve_aou_knowledge_base(query: str) -> str:
    """
    Retrieve relevant documents from AOU knowledge base.

    Args:
        query: The question to search for.
    Returns:
        Retrieved documents as formatted text
    """

    try:
        results = query_all_collections(query)
        if not results:
            return "No relevant documents found."

        formatted = "=== Retrieved Documents ===\n"
        formatted += "\n".join(results)
        return formatted
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"

if __name__ == "__main__":
    for record in searching_aou_site("Ahmed Samir"):

        print(record)