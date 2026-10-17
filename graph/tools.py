from typing import List, Dict, Any, Literal

from ddgs import DDGS
from langchain_core.tools import tool

from data_prep.qdrant.config import query_all_collections


@tool
def searching_aou_site(query: str) -> list[dict[str, Any]] | str:
    """
    Searches the official AOU websites (aou.edu.om and arabou.edu.kw) for facts about AOU Oman:
    admissions, programs, fees, contacts, policies and news. Never answer these from memory, search instead.
    If retrieve_aou_knowledge_base is available, try it first.
    Make one focused search, at most two. If the results are enough, answer immediately and include the links.
    :param query: user query
    :return: list of dict with the keys [title, href, body] containing most relevant web searches
    """
    try:
        # site: takes a bare domain, a scheme or a leading "-" breaks the filter
        query1 = f"{query} site:arabou.edu.kw"
        query2 = f"{query} site:aou.edu.om"
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
    If the information gotten from this function are enough return them immediately.
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
        # no Qdrant configured or reachable, the app still works through web search
        print(f"Knowledge base unavailable: {e}")
        return "The knowledge base is not available right now. Use searching_aou_site to answer instead."

if __name__ == "__main__":
    for record in searching_aou_site("Ahmed Samir"):

        print(record)