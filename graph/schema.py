from typing import List, Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class RetrivalState(MessagesState):
    retrieval_result: List[str]
    query: str


class RouterSchema(BaseModel):
    """Analyzes the context and the fulfillment of the query and route the llm to act accordingly"""
    reasoning: str = Field(
        description="Step by step reasoning behind the classification"
    )

    classification: Literal["fulfilled", "not_fulfilled"] = Field(
        description="The classification of the query fulfillment based on the context. 'fulfilled' Means the context is enough for the llm to answer the query. 'not_fulfilled' means a websearch needs to be fired to answer the query"
    )
