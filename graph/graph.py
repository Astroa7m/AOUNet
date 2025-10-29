import sys
from copy import copy
from typing import Literal

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph.state import CompiledStateGraph

from common.helpers import llm
from common.pretty_print import pretty_print_messages
from graph.prompt import AOU_NET_SYSTEM_PROMPT
from graph.schema import AgentState, AgentRouterSchema
from graph.tools import *

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


# ============================================================================
# NODE FUNCTIONS
# ============================================================================

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


# will use this later to rerank and optimize context
def extract_tool_contexts(messages: list) -> dict[str, Any]:
    """
    Extract query and results for each tool type from recent messages.
    Only gets the LAST occurrence of each tool to avoid mixing old results.
    """

    contexts = {
        "knowledge_base_query": None,
        "knowledge_base_results": [],
        "web_search_query": None,
        "web_search_results": [],
    }

    # find the last AIMessage with tool calls and subsequent ToolMessages
    last_tool_call_index = None
    for i in reversed(range(len(messages))):
        if isinstance(messages[i], AIMessage) and messages[i].tool_calls:
            last_tool_call_index = i
            break

    if last_tool_call_index is None:
        return contexts

    # get the AI message with tool calls
    ai_msg_with_tools = messages[last_tool_call_index]

    # map tool call IDs to their queries
    tool_call_map = {}
    for tool_call in ai_msg_with_tools.tool_calls:
        tool_name = tool_call.get("name", "").lower()
        tool_call_id = tool_call.get("id")
        query = tool_call.get("args", {}).get("query", "")

        tool_call_map[tool_call_id] = {
            "name": tool_name,
            "query": query
        }

    # collect ToolMessages that come after this AI message
    for i in range(last_tool_call_index + 1, len(messages)):
        msg = messages[i]
        if isinstance(msg, ToolMessage):
            tool_call_id = msg.tool_call_id
            if tool_call_id in tool_call_map:
                tool_info = tool_call_map[tool_call_id]
                tool_name = tool_info["name"]

                # categorize by tool type
                if "retrieve_aou_knowledge_base" in tool_name:
                    contexts["knowledge_base_query"] = tool_info["query"]
                    contexts["knowledge_base_results"].append(msg.content)
                elif "searching_aou_site" in tool_name:
                    contexts["web_search_query"] = tool_info["query"]
                    contexts["web_search_results"].append(msg.content)

    return contexts


def call_llm(state: AgentState) -> Dict[str, Any]:
    response = llm_with_tools.invoke(state['messages'])

    return {"messages": [response]}


def tool_handler(state: dict):
    "Performs tool call"

    # list tool for tool message
    result = []
    tool_call_count = state.get('tool_call_count', {}).copy()
    max_calls = state.get('max_tool_calls_per_tool', 2)
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

            # Check if tool has been called too many times
            current_count = tool_call_count.get(tool_call["name"], 0)

            if current_count >= max_calls:
                result.append(ToolMessage(
                    content=f"⚠️ Tool '{tool_call["name"]}' has already been called {current_count} times. Maximum calls reached. Please provide an answer with the information you have.",
                    tool_call_id=tool_call['id']
                ))
                continue

            # run the tool
            tool_res = tool.invoke(tool_call['args'])

            # Increment counter
            tool_call_count[tool_call["name"]] = current_count + 1

            # ensure content is a string
            if not isinstance(tool_res, str):
                tool_res = str(tool_res)

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
        "messages": result,
        "tool_call_count": tool_call_count
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

def build_assistant() -> CompiledStateGraph[Any, Any, Any, Any]:
    """
    Constructs the retrieval state graph that either gets data via RAG or via websearch

    Returns:
        Compiled StateGraph ready for execution
    """
    builder = StateGraph(AgentState)

    # nodes
    builder.add_node("add_system_message", add_system_message_if_needed)
    builder.add_node("retrieval", retrieval)
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
    builder.add_edge("cleanup_state", END)

    # after tool execution, loop back to LLM
    builder.add_edge("tool_handler", "call_llm")
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


def normal_llm(state: MessagesState):
    response = llm.invoke(state['messages'])

    return {"messages": [response]}


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


# def build_assistant():
#     overall_workflow = (
#         StateGraph(AgentState)
#         .add_node("add_system_message", add_system_message_if_needed)
#         .add_node(agent_router)
#         .add_node("aou_retrieval_graph", build_aou_retrieval_graph())
#         .add_node("normal_llm", normal_llm)
#         .add_edge(START, "add_system_message")
#         .add_edge("add_system_message", "agent_router")
#         .add_edge("normal_llm", END)
#     )
#     memory = MemorySaver()
#     aou_assistant = overall_workflow.compile(checkpointer=memory)
#     return aou_assistant


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "memory_test"}}  # persistent thread_id
    agent = build_assistant()

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        result = agent.invoke(
            input={"messages": [HumanMessage(user_input)]},
            config=config
        )

        pretty_print_messages(result["messages"])
        # time.sleep(0.2)  # small delay to make console easier to read
