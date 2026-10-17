from typing import List, Dict, Any, Literal

from ddgs import DDGS
from langchain_core.tools import tool

from data_prep.qdrant.config import query_all_collections

# The free Groq tier allows about 7,000 to 8,000 tokens per request and per minute, so tool results get a budget.
# A full module page is about 1,500 characters; without these caps one knowledge base call was 5,000 tokens.
MAX_DOC_CHARS = 900
MAX_KB_CHARS = 4000       # about 1,000 tokens
MAX_SEARCH_RESULTS = 5
MAX_SEARCH_BODY_CHARS = 250


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
            # the Oman site first, it is the most relevant for students here
            results = ddgs.text(query2, max_results=3) + ddgs.text(query1, max_results=3)
        return [
            {"title": r.get("title", ""), "href": r.get("href", ""), "body": r.get("body", "")[:MAX_SEARCH_BODY_CHARS]}
            for r in results[:MAX_SEARCH_RESULTS]
        ]
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
        results = query_all_collections(query, n_results=3)
        if not results:
            return "No relevant documents found."

        # best match of each collection first (see query_all_collections), stop at the budget
        kept, used = [], 0
        for document in results:
            document = " ".join(document.split())  # the CSV chunks are full of padding spaces
            if len(document) > MAX_DOC_CHARS:
                document = document[:MAX_DOC_CHARS] + "..."
            if used + len(document) > MAX_KB_CHARS and kept:
                break
            kept.append(document)
            used += len(document)
        return "=== Retrieved Documents ===\n" + "\n---\n".join(kept)
    except Exception as e:
        # no Qdrant configured or reachable, the app still works through web search
        print(f"Knowledge base unavailable: {e}")
        return "The knowledge base is not available right now. Use searching_aou_site to answer instead."

if __name__ == "__main__":
    for record in searching_aou_site("Ahmed Samir"):

        print(record)