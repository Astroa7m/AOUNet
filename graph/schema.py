from typing import List, Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class AgentState(MessagesState):
    """Enhanced state that tracks retrieval results and iterations"""
    retrieval_result: List[str]
    query: str


class RouterSchema(BaseModel):
    """Analyzes the context and the fulfillment of the query and route the llm to act accordingly"""
    reasoning: str = Field(
        description="Step by step reasoning behind the classification"
    )

    classification: Literal["none", "only_context", "only_websearch", "both"] = Field(
        description="""
        The classification of the query fulfillment based on the provided context.
        'none' Means the user is just greeting and asking about general things.
        'only_context' Means the context is enough for the llm to answer the query.
        'only_websearch' Means a websearch needs to be fired to answer the query because the context is irrelevant
        'both' Means the context is relevant but might not be enough and therefore more information could be fetched with a websearch 
        """
    )


class AgentRouterSchema(BaseModel):
    """Analyzes the intent of the query and router accordingly"""
    reasoning: str = Field(
        description="Step by step reasoning behind the classification"
    )

    classification: Literal["info", "normal"] = Field(
        description="""
        Classification of the user's query intent:

        'info' - Use this for ANY query that could be answered using AOU (Arab Open University) information. 
        Always prefer this classification unless it is absolutely clear that the query is unrelated to AOU.
        This includes:
            - Questions about AOU faculty members, staff, or students (e.g., "Who is Dr. Ahmed?", "Tell me about Dawood")
            - Questions about AOU programs, courses, schedules, or policies
            - Questions about AOU facilities, locations, or services
            - Questions about AOU events, news, or announcements
            - Questions about admission, registration, or academic procedures
            - ANY person's name mentioned, assume they might be affiliated with AOU unless explicitly stated otherwise
            - Questions about departments, faculties, or organizational structure
            - Historical or factual information about AOU
        When unsure, choose 'info' and seek clarification from the user if necessary.

        'normal' - Use this only for queries that are clearly outside AOU’s domain or are general conversation, such as:
            - Greetings or casual talk (e.g., "Hello", "Thanks", "How are you?")
            - Personal questions about the AI itself (e.g., "Who made you?", "What can you do?")
            - General knowledge questions clearly unrelated to AOU (e.g., "What's the weather?", "Tell me a joke")

        Behavioral note:
            - The assistant should always consider asking for clarification even after responding — 
              especially if there’s any doubt that the user’s intent was fully addressed or could relate to AOU.
            - Default mindset: “The user probably means something about AOU — let’s clarify to be sure.”

        In short:
            - 'info': The default and preferred route for anything possibly AOU-related.
            - 'normal': Clearly non-AOU or social/general conversation.
        """
    )
