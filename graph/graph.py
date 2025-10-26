from typing import Literal, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from common.helpers import llm
from common.pretty_print import pretty_print_messages
from data_prep.qdrant.config import query_all_collections
from graph.prompt import RETRIEVAL_PROMPT
from graph.schema import AgentState, AgentRouterSchema
from graph.tools import search_aou_site

# ============================================================================
# TOOLS DEFINITION
# ============================================================================


tools = [search_aou_site]
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

    results=[]

    try:
        results = query_all_collections(query)
    except Exception as e:
        print(f"Error querying collection: {e}")

    return {
        "retrieval_result": results,
        "query": query,
        "messages": state['messages']
    }


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
# CONSTRUCTING AOU MULTI-RETRIEVAL SUPGRAPH
# ============================================================================

def build_aou_retrieval_graph() -> CompiledStateGraph[Any, Any, Any, Any]:
    """
    Constructs the retrieval state graph that either gets data via RAG or via websearch

    Returns:
        Compiled StateGraph ready for execution
    """
    builder = StateGraph(AgentState)

    # nodes
    builder.add_node("retrieval", retrieval)
    builder.add_node("call_llm", call_llm)
    builder.add_node("tool_handler", tool_handler)
    builder.add_node("cleanup_state", cleanup_state)
    # flow
    builder.add_edge(START, "retrieval")
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
    builder.add_edge("cleanup_state", END)

    # after tool execution, increment counter and loop back to LLM
    builder.add_edge("tool_handler", "call_llm")

    return builder.compile()


# ============================================================================
# CONSTRUCTING OVERALL GRAPH
# ============================================================================

def agent_router(state: MessagesState) -> Command[Literal["aou_retrieval_graph", 'normal_llm']]:
    print("Got state in router", state)
    res = llm_with_agent_router.invoke(
        [
            {"role": "system", "content": f"Classify the following query:\n{state['messages'][-1].content}"},
        ]
    )

    try:
        if res.classification == 'info':
            goto = 'aou_retrieval_graph'
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
        .add_node("aou_retrieval_graph", build_aou_retrieval_graph())
        .add_node("normal_llm", normal_llm)
        .add_edge(START, "agent_router")
        .add_edge("normal_llm", END)
    )
    memory = MemorySaver()
    aou_assistant = overall_workflow.compile(checkpointer=memory)
    return aou_assistant


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
