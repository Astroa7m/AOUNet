import sys
from copy import copy
from functools import lru_cache
from typing import Any, Dict, Literal

from langchain.agents import create_agent
import groq
from langchain.agents.middleware import (
    AgentMiddleware,
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    HumanInTheLoopMiddleware,
    ModelFallbackMiddleware,
    ModelRequest,
    ModelRetryMiddleware,
    SummarizationMiddleware,
    dynamic_prompt,
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from common.helpers import fallback_llms, llm, summarization_model
from common.pretty_print import pretty_print_messages
from data_prep.qdrant.config import knowledge_base_available, query_all_collections
# absolute imports work everywhere once the project is installed with `pip install -e .`
from graph.prompt import AOU_NET_SYSTEM_PROMPT, RERANK_PROMPT, AGENT_ROUTER_PROMPT, ROLE_PROMPTS
from graph.schema import AgentState, AgentRouterSchema, RetrieveMessageReranked, UserContext
from graph.tools import searching_aou_site, retrieve_aou_knowledge_base
from graph.user_tools import STUDENT_TOOLS, TUTOR_TOOLS

# todo: fix conversation dataset format user/assistant, might confuse llm
# todo: add debugging prints or loggers for each step to monitor state change and stateless llm calls
# todo: implement reranker top-k
# todo: fix tool loop when asked + monitor retrieve context

# ============================================================================
# TOOLS DEFINITION
# ============================================================================


tools = [searching_aou_site, retrieve_aou_knowledge_base]
llm_with_tools = llm.bind_tools(tools)
llm_with_agent_router = llm.with_structured_output(AgentRouterSchema, method="json_schema")
llm_with_reranker = llm.with_structured_output(RetrieveMessageReranked, method="json_schema")


# ============================================================================
# NODE FUNCTIONS
# ============================================================================

# not using it anymore, might remove later
def retrieval(state: AgentState) -> Dict[str, Any]:
    """
    Retrieves relevant documents from collections based on the user's query.

    Args:
        state: Current agent state containing messages

    Returns:
        Dict with retrieval_result, query, and messages
    """
    # extract query from the last human message
    last_message = state["messages"][-1]
    query = last_message.content

    # handle different query formats (for LangChain Studio compatibility)
    if isinstance(query, list):
        # extract text from structured format: [{'type': 'text', 'text': '<query>'}]
        query = next((item.get('text', '') for item in query if item.get('type') == 'text'), '')

    if not query:
        return {
            "retrieval_result": [],
            "query": "",
            "messages": state['messages']
        }

    results = []

    try:
        results = query_all_collections(query)
        # for debugging
        print(f"collections result: {results}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"Error querying collection: {e}")

    return {
        "retrieval_result": results,
        "query": query,
        "messages": state['messages']
    }


def router(state: AgentState) -> Command[Literal["sql_subgraph", "call_llm"]]:
    message = state["messages"][-1]
    res = llm_with_agent_router.invoke(
        [
            {"role": "system", "content": AGENT_ROUTER_PROMPT},
            {"role": "user", "content": message},
        ]
    )
    update = AIMessage(res.reasoning)

    if res.classification == "tutors_modules":
        goto = "sql_subgraph"
    elif res.classification == "normal":
        goto = "call_llm"
    else:
        raise ValueError(f"Invalid classification {res.classification}")
    return Command(goto=goto, update=update)

def add_system_message_if_needed(state: MessagesState) -> dict:
    """Add system message only if this is the first message in the conversation."""

    # check if we already have a system message
    has_system_message = any(isinstance(msg, SystemMessage) for msg in state['messages'])

    if not has_system_message:
        system_message = SystemMessage(AOU_NET_SYSTEM_PROMPT)
        # getting user message
        user_message = state["messages"][0]
        # preparing to remove user message from state
        to_remove = [RemoveMessage(user_message.id)]
        # making a copy of the user message and changing its id to not interfer wiht deletion
        to_add_user_message = copy(user_message)
        to_add_user_message.id = user_message.id + "first"
        # adding system first then user message
        to_add = [system_message, to_add_user_message]
        return {"messages": to_remove + to_add}

    return {"messages": []}


def rerank_and_optimize_retrieved(query: str, data: str) -> RetrieveMessageReranked | str:
    try:
        return llm_with_reranker.invoke(RERANK_PROMPT.substitute(query=query, retrieved_data=data))
    except Exception as e:
        print("Could not call reranker")
        return data


def call_llm(state: AgentState) -> Dict[str, Any]:
    response = llm_with_tools.invoke(state['messages'])

    return {"messages": [response]}


def tool_handler(state: dict):
    "Performs tool call"

    # list tool for tool message
    result = []
    # iterate through tool calls

    for tool_call in state['messages'][-1].tool_calls:
        try:
            # get the tool
            tool = next((t for t in tools if t.name.lower() == tool_call["name"].lower()), None)

            if tool is None:
                result.append(ToolMessage(
                    content=f"Error: Tool '{tool_call['name']}' not found",
                    tool_call_id=tool_call['id']
                ))
                continue

            # run the tool
            tool_res = tool.invoke(tool_call['args'])

            # ensure content is a string
            if not isinstance(tool_res, str):
                tool_res = str(tool_res)

            print(50 * "=", "before reranked\n", tool_res)
            # format, rerank, optimize, and return top-k for tool result (search & retrieval only) to reduce token usage
            # only for these two tools
            if tool.name in [searching_aou_site.name, retrieve_aou_knowledge_base.name]:
                reranked_info = rerank_and_optimize_retrieved(tool_call['args']['query'], tool_res)
                print(50 * "=", "query is\n", tool_call['args']['query'])
                # in case of error this will be a string returning the original data without reranking
                if isinstance(reranked_info, RetrieveMessageReranked):
                    tool_res = str(reranked_info.messages)
                    print(50 * "=", "after reranked\n", tool_res)
            # create a ToolMessage
            result.append(
                ToolMessage(
                    content=tool_res,
                    tool_call_id=tool_call['id']
                )
            )
        except Exception as e:
            result.extend(
                [
                    ToolMessage(
                        content=f"Error executing tool '{tool_call['name']}': {str(e)}. Please try again with different parameters.",
                        tool_call_id=tool_call['id']
                    )
                ]
            )

    return {
        "messages": result
    }


def cleanup_state(state: AgentState) -> MessagesState:
    """Clean up temporary state after query completion, keep only conversation."""

    # keep only user query and final AI response for conversation history
    messages_to_keep = []

    # find the last user message (current query)
    last_human_msg = None
    for msg in reversed(state['messages']):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg
            break

    # find the final AI response (last AIMessage with content)
    final_ai_msg = None
    for msg in reversed(state['messages']):
        if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
            # Skip reasoning messages
            if not msg.content.startswith("reasoning:"):
                final_ai_msg = msg
                break

    remove_message = None
    for msg in reversed(state['messages']):
        if isinstance(msg, ToolMessage):
            # removing redundant tool result from chat history after the model has answered to reduce chat history size
            remove_message = RemoveMessage(msg.id)
            break

    if last_human_msg:
        messages_to_keep.append(last_human_msg)
    if final_ai_msg:
        messages_to_keep.append(final_ai_msg)

    return {
        # to avoid "NotImplementedError: Unsupported message type: <class 'NoneType'>" if no tool was found
        "messages": [remove_message] + messages_to_keep if remove_message else messages_to_keep
    }


# ============================================================================
# CONDITIONAL EDGE FUNCTIONS
# ============================================================================

def should_continue(state: AgentState) -> Literal["tool_handler", "cleanup_state"]:
    last_message = state['messages'][-1]

    # If no tool calls, we have the answer
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return "cleanup_state"

    # Execute tools
    return "tool_handler"


# ============================================================================
# CONSTRUCTING AOU MULTI-RETRIEVAL SUPGRAPH
# ============================================================================
memory = MemorySaver()


def build_assistant() -> CompiledStateGraph[Any, Any, Any, Any]:
    """
    Constructs the retrieval state graph that either gets data via RAG or via websearch

    Returns:
        Compiled StateGraph ready for execution
    """
    builder = StateGraph(AgentState)

    # nodes
    builder.add_node("add_system_message", add_system_message_if_needed)
    builder.add_node("call_llm", call_llm)
    builder.add_node("tool_handler", tool_handler)
    builder.add_node("cleanup_state", cleanup_state)
    # flow
    builder.add_edge(START, "add_system_message")
    builder.add_edge("add_system_message", "call_llm")

    # Conditional edge: continue to tools or end
    builder.add_conditional_edges(
        "call_llm",
        should_continue,
        {
            "tool_handler": "tool_handler",
            "cleanup_state": "cleanup_state"
        }
    )
    # after tool execution, loop back to LLM
    builder.add_edge("tool_handler", "call_llm")
    builder.add_edge("cleanup_state", END)

    return builder.compile(checkpointer=memory)


# ============================================================================
# TEST
# ============================================================================
ROLE_TOOLS = {
    "guest": [],
    "student": STUDENT_TOOLS,
    "tutor": TUTOR_TOOLS,
}


@dynamic_prompt
def role_aware_prompt(request: ModelRequest) -> str:
    """Base AOUNet prompt plus a section for the logged in user (name and role only, no records)"""
    context: UserContext = request.runtime.context or UserContext()
    return AOU_NET_SYSTEM_PROMPT + ROLE_PROMPTS[context.role].substitute(name=context.name)


def is_retryable(error: Exception) -> bool:
    """gpt-oss on Groq sometimes emits a broken tool name (e.g. 'search<|channel|>commentary'), a retry fixes it"""
    if isinstance(error, (groq.RateLimitError, groq.APIConnectionError, groq.InternalServerError)):
        return True
    message = str(error).lower()
    return isinstance(error, groq.BadRequestError) and (
        "tool call validation failed" in message or "tool choice is none" in message
    )


def model_failure_message(error: Exception) -> str:
    """Shown to the user when every retry failed. The real error goes to the logs (Streamlit Cloud: Manage app)."""
    print(f"Model call failed after retries: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    if isinstance(error, groq.RateLimitError):
        return "The free AI quota is busy right now. Please wait a minute and ask again."
    return "Sorry, I could not reach the language model right now. Please try again in a moment."


# every tool round re-sends the whole context, and the free tier allows about 8,000 tokens per minute,
# so more than 2 tool rounds per answer runs out of budget inside a single answer
MAX_TOOL_RESULTS_PER_ANSWER = 2


class AnswerAfterEnoughTools(AgentMiddleware):
    """After 2 tool results for the current question, the model must answer.

    Blocking extra calls (ToolCallLimitMiddleware) let gpt-oss keep asking until LangGraph's 25 step
    recursion limit crashed the run. Removing the tools alone was not enough either: with tool calls in
    its history gpt-oss still called one and Groq answered 400 "Tool choice is none". So the model gets
    a clean conversation instead: earlier questions and answers, plus one message with what the tools found.

    A class with both methods, because the Streamlit page streams with astream_events (async) while
    scripts and tests use invoke (sync). A @wrap_model_call function only has the sync one.
    """

    def _prepare(self, request: ModelRequest) -> ModelRequest:
        messages = request.messages
        last_question = max((i for i, m in enumerate(messages) if m.type == "human"), default=-1)
        findings = [m for m in messages[last_question + 1:] if m.type == "tool"]
        if last_question < 0 or len(findings) < MAX_TOOL_RESULTS_PER_ANSWER:
            return request

        earlier = [
            m for m in messages[:last_question]
            if m.type == "human" or (m.type == "ai" and m.content and not m.tool_calls)
        ]
        found = "\n\n".join(f"[{m.name}]\n{m.content}" for m in findings)
        final_question = HumanMessage(
            f"{messages[last_question].content}\n\n"
            f"Here is what your tools found:\n{found}\n\n"
            "Answer now using this information. Do not call any tools."
        )
        return request.override(messages=earlier + [final_question], tools=[])

    def wrap_model_call(self, request, handler):
        return handler(self._prepare(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._prepare(request))


def general_tools() -> list:
    """Web search always works with no key. The knowledge base only when Qdrant is up, otherwise the model loops on it."""
    if knowledge_base_available():
        return [searching_aou_site, retrieve_aou_knowledge_base]
    return [searching_aou_site]


@lru_cache(maxsize=None)
def get_agent(role: str = "guest"):
    """One agent per role. All share the same memory, each chat has its own thread_id."""
    return create_agent(
        model=llm,
        tools=general_tools() + ROLE_TOOLS[role],
        context_schema=UserContext,
        checkpointer=memory,
        middleware=build_middleware(),
    )


def build_middleware() -> list:
    """Order matters: the first one wraps all the others. Every one must work sync and async,
    because the UI streams (async) and scripts invoke (sync); tests/test_middleware.py checks it."""
    return [
            role_aware_prompt,
            # before retry and fallback, so every model in the chain gets the tool-free request
            AnswerAfterEnoughTools(),
            # waits 5s, 10s, 20s: long enough for the free tier's per-minute token window to free up
            ModelRetryMiddleware(
                max_retries=3,
                retry_on=is_retryable,
                on_failure=model_failure_message,
                initial_delay=5.0,
                backoff_factor=2.0,
                max_delay=20.0,
            ),
            # daily quota used up on the main model: answer with the next model, it has its own quota
            *([ModelFallbackMiddleware(*fallback_llms)] if fallback_llms else []),
            # old tool results were the bulk of every request (one knowledge base call stayed in all later calls),
            # so once the request passes 2,500 tokens only the 2 newest tool results are sent in full
            ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=2500, keep=2)]),
            SummarizationMiddleware(
                model=summarization_model,
                trigger=("tokens", 3000),
                keep=("messages", 8),
            ),
            # pauses the run before any email goes out, the UI shows Approve / Reject
            HumanInTheLoopMiddleware(
                interrupt_on={"send_email": {"allowed_decisions": ["approve", "reject"]}},
                description_prefix="Email waiting for your approval",
            ),
    ]


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "memory_test"}}  # persistent thread_id
    # agent = build_assistant()
    agent = get_agent()

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        result = agent.invoke(
            input={"messages": [HumanMessage(user_input)]},
            config=config,
            context=UserContext(),
        )

        pretty_print_messages(result["messages"])
        # time.sleep(0.2)  # small delay to make console easier to read
