from typing import List, Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class AgentState(MessagesState):
    """Enhanced state that tracks retrieval results and iterations"""
    # retrieval_result: List[str]
    # query: str


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

class RetrieveMessageReranked(BaseModel):
    """Selects, refines, and organizes the most relevant information from retrieved documents."""
    messages: List[str] = Field(
        description="""
        A clean, ordered list of the most relevant and clear information extracted from the retrieved items.

        Each message should:
            - Directly answer or strongly support the user's query.
            - Be paraphrased or summarized for clarity and conciseness.
            - Merge similar or repeated information into a single coherent statement.
            - Exclude any irrelevant, vague, or redundant content.
            - Preserve factual accuracy and be easy for users to read.

        Output behavior:
            - The list should be sorted from most to least relevant.
            - Each item should be short, factual, and phrased naturally.
            - Avoid including metadata, file names, or collection identifiers.

        In short:
            - Provide only the most useful, paraphrased information that helps the assistant generate an accurate final response.
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
        - 'tutors_modules': When the query is about tutors or modules
        - 'normal': normal chatting, might be faq, policies, study plans, general queries. Might also includes individuals that are not tutors.
        """
    )



class RerankedContext(BaseModel):
    """Structured output for reranked and optimized context"""
    relevant_passages: list[str] = Field(
        description="Top relevant passages reranked by importance to the query"
    )
    summary: str = Field(
        description="Brief summary of the key information found"
    )
    confidence: str = Field(
        description="Confidence level: high, medium, or low"
    )