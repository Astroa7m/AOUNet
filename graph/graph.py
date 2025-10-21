from typing import Literal, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from common.helpers import llm
from data_prep.config import get_pdf_collection, get_q_a_collection
from prompt import RETRIEVAL_PROMPT, DECISION_MAKING_PROMPT, WEBSEARCH_PROMPT
from schema import RouterSchema, AgentState, AgentRouterSchema
from tools import search_aou_site

# ============================================================================
# TOOLS DEFINITION
# ============================================================================

# @tool
# def done_tool() -> str:
#     """Call this tool when you have completed the task and provided a final answer to the user"""
#     return "Task completed successfully"


tools = [search_aou_site]
llm_with_tools = llm.bind_tools(tools)
llm_with_agent_router = llm.with_structured_output(AgentRouterSchema, method="json_schema")


# llm_with_data_router = llm.with_structured_output(RouterSchema, method='json_schema')


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

    # query both collections
    results = []
    collections = [get_q_a_collection(), get_pdf_collection()]

    for collection in collections:
        try:
            r = collection.query(query_texts=query,
                                 n_results=3)  # reducing results to make testing easier mgiht increase later
            # ChromaDB returns nested lists: {'documents': [[doc1, doc2, ...]]}
            if r.get('documents') and len(r['documents']) > 0:
                results.extend(r['documents'][0])
        except Exception as e:
            print(f"Error querying collection: {e}")
            continue

    return {
        "retrieval_result": results,
        "query": query,
        "messages": state['messages']
    }


# def data_router(state: AgentState) -> Dict[str, Any]:
#     """
#     Analyzes the query and retrieval results to determine if we have enough context
#     or need to search for more information.
#
#     Args:
#         state: Current agent state
#
#     Returns:
#         Dict with updated messages and potentially modified query
#     """
#     query = state['query']
#     context = state['retrieval_result']
#
#     # use LLM to classify if retrieval results are sufficient
#     result = llm_with_data_router.invoke([
#         {
#             "role": "system",
#             "content": DECISION_MAKING_PROMPT.substitute(query=query, context=context)
#         }
#     ])
#
#     reasoning = result.reasoning
#     classification = result.classification
#
#     # create AI message with reasoning
#     ai_msg = AIMessage(content=f"reasoning: {reasoning}", metadata={"reasoning": reasoning})
#
#     if classification == "only_context":
#         # We have enough context, proceed to generate answer
#         return {
#             "messages": [ai_msg]
#         }
#     elif classification == "only_websearch":
#         # Context not relevant, instruct LLM to use search tool
#         return {
#             "messages": [ai_msg],
#             "retrieval_result": []
#         }
#     elif classification == "both":
#         # Context not enough, instruct LLM to use search tool
#         return {
#             "messages": [ai_msg],
#             "retrieval_result": context
#
#         }
#     elif classification == "none":
#         return {
#             "messages": [ai_msg],
#             "retrieval_result": []
#
#         }
#     else:
#         raise ValueError(f"Invalid classification from router: {classification}")


def call_llm(state: AgentState) -> Dict[str, Any]:
    context = state["retrieval_result"]
    query = state['query']

    # gather ALL context including tool results
    all_context_parts = []
    all_context_parts.append("=== Retrieved Documents ===")
    all_context_parts.extend(context)

    # Add tool results from message history
    tool_results = [
        msg.content for msg in state['messages']
        if isinstance(msg, ToolMessage)
    ]
    if tool_results:
        all_context_parts.append("\n=== Additional Search Results ===")
        all_context_parts.extend(tool_results)

    combined_context = "\n".join(all_context_parts)
    prompt = RETRIEVAL_PROMPT.substitute(query=query, context=combined_context)


    system_message = SystemMessage(content=prompt)
    response = llm_with_tools.invoke(state['messages'] + [system_message])

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

    # DEBUG
    print(f"📝 Content: '{last_message.content}'")
    print(f"📝 Type: {type(last_message)}")
    print(f"📝 Additional kwargs: {last_message.additional_kwargs}")
    print(f"📝 Response metadata: {last_message.response_metadata}")
    if hasattr(last_message, 'reasoning_content'):
        print(f"💭 Reasoning content: {last_message.reasoning_content}")

    # If no tool calls, we have the answer
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return "cleanup_state"

    # Execute tools
    return "tool_handler"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

# def finalize(state: AgentState):
#     """Generate final response using ALL gathered information."""
#     context = state["retrieval_result"]
#     query = state['query']
#
#     # gather ALL context including tool results (same as call_llm)
#     all_context_parts = []
#
#     if context:
#         all_context_parts.append("=== Retrieved Documents ===")
#         all_context_parts.extend(context)
#
#     # Add tool results
#     tool_results = [
#         msg.content for msg in state['messages']
#         if isinstance(msg, ToolMessage)
#     ]
#     if tool_results:
#         all_context_parts.append("\n=== Additional Search Results ===")
#         all_context_parts.extend(tool_results)
#
#     # Build comprehensive prompt
#     if all_context_parts:
#         combined_context = "\n".join(all_context_parts)
#         prompt = RETRIEVAL_PROMPT.substitute(query=query, context=combined_context)
#     else:
#         prompt = WEBSEARCH_PROMPT.substitute(query=query)
#
#     response = llm.invoke(state['messages'] + [SystemMessage(content=prompt)])
#
#     return {"messages": [response]}


def build_graph() -> CompiledStateGraph[Any, Any, Any, Any]:
    """
    Constructs the LangGraph state graph with all nodes and edges.

    Returns:
        Compiled StateGraph ready for execution
    """
    builder = StateGraph(AgentState)

    # nodes
    builder.add_node("retrieval", retrieval)
    # builder.add_node("data_router", data_router)
    builder.add_node("call_llm", call_llm)
    builder.add_node("tool_handler", tool_handler)
    builder.add_node("cleanup_state",cleanup_state)
    # builder.add_node("finalize", finalize)
    # flow
    builder.add_edge(START, "retrieval")
    # builder.add_edge("retrieval", "data_router")
    builder.add_edge("retrieval", "call_llm")

    # Conditional edge: continue to tools or end
    builder.add_conditional_edges(
        "call_llm",
        should_continue,
        {
            "tool_handler": "tool_handler",
            "cleanup_state": "cleanup_state"
        }
    )
    builder.add_edge("cleanup_state",END)


    # after tool execution, increment counter and loop back to LLM
    builder.add_edge("tool_handler", "call_llm")

    return builder.compile()


def agent_router(state: MessagesState) -> Command[Literal["graph", 'normal_llm']]:
    print("Got state in router", state)
    res = llm_with_agent_router.invoke(
        [
            {"role": "system", "content": f"Classify the following query:\n{state['messages'][-1].content}"},
        ]
    )

    try:
        if res.classification == 'info':
            goto = 'graph'
        else:
            goto = 'normal_llm'
        print(f"{res.classification}: reasoning from agent router: {res.reasoning}")

        return Command(goto=goto, update=state)

    except Exception as e:
        print("Exception occurred in agent_router: Fix it later mate", e)


def normal_llm(state: MessagesState):
    response = llm.invoke(state['messages'])

    return {"messages": [response]}


def cleanup_state(state: AgentState) -> Dict[str, Any]:
    """Clean up temporary state after query completion, keep only conversation."""

    # keep only user query and final AI response for conversation history
    messages_to_keep = []

    # find the last user message (current query)
    last_human_msg = None
    for msg in state['messages']:
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

    if last_human_msg:
        messages_to_keep.append(last_human_msg)
    if final_ai_msg:
        messages_to_keep.append(final_ai_msg)

    return {
        "messages": messages_to_keep,
        "retrieval_result": [],  # Clear retrieval
        "query": "",  # Clear query
    }


def build_assistant():
    overall_workflow = (
        StateGraph(AgentState)
        .add_node(agent_router)
        .add_node("graph", build_graph())
        .add_node("normal_llm", normal_llm)
        .add_edge(START, "agent_router")
        .add_edge("normal_llm", END)
    )

    aou_assistant = overall_workflow.compile()
    return aou_assistant


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    result = build_assistant().invoke({
        "messages": [HumanMessage("Who is Dawood Sulima")],
    })

    print("\n" + "=" * 80)
    print("CONVERSATION HISTORY")
    print("=" * 80 + "\n")

    for message in result['messages']:
        print("=" * 80 + "\n")
        print(type(message))

        # try:
        #     if message.metadata['reasoning']:
        #         print("reasoning:", message.metadata['reasoning'])
        # except:
        #     pass
        #
        # try:
        #     if message.additional_kwargs['reasoning_content']:
        #         print("reasoning:", message.additional_kwargs['reasoning_content'])
        # except:
        #     pass

        print("content:", message)
        print("\n" + "=" * 80)
