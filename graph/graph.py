from typing import List

from langchain_core.messages import HumanMessage
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from data_prep.config import get_pdf_collection, get_q_a_collection
from helpers import llm


class RetrivalSate(MessagesState):
    retrieval_result: List[str]
    query: str
def retrieval(state: MessagesState) -> RetrivalSate:
    query = state["messages"][0].content
    results = []

    for collection in [get_q_a_collection(), get_pdf_collection()]:
        r = collection.query(query_texts=[query], n_results=5)
        results.extend(r['documents'][0]) # returns [[data here]], where the outer contains a single list

    return {
        "retrieval_result": results,
        "query": query
    }


def call_llm(state: RetrivalSate) -> MessagesState:
    result = state["retrieval_result"]
    query = state['query']

    return {
        "messages": [llm.invoke(f"You are a helpful assistant. Answer the followig query based on the following context:\nquery:{query}\ncontent{result}")]
    }

builder = StateGraph(MessagesState)
builder.add_node("retrieval", retrieval)
builder.add_node("another", call_llm)
builder.add_edge(START, "retrieval")
builder.add_edge("retrieval", "another")
builder.add_edge("another", END)
graph = builder.compile()


result = graph.invoke({"messages": [HumanMessage("Do you have the study plan of ITC?")]})

for message in result['messages']:
    message.pretty_print()
