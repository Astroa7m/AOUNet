from ddgs import DDGS
from langchain_core.tools import tool


@tool
def search_aou_site(query: str) -> list[dict[str, str]]:
    """
    Search for information related to the user query about AOU Oman.
    This function search the university website for relative information
    :param query: user query
    :return: list of dict with the keys [title, href, body] containing most relevant web searches
    """

    query1 = f"{query} - site:https://www.arabou.edu.kw/"
    query2 = f"{query} - site:https://www.aou.edu.om/"
    with DDGS() as ddgs:
        results1 = ddgs.text(query1, max_results=5)
        results2 = ddgs.text(query2, max_results=5)
        results = results1 + results2
        return results


if __name__ == "__main__":
    for record in search_aou_site("Ahmed Samir"):

        print(record)