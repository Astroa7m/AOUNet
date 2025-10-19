from typing import List, Literal, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import Field
from common.helpers import llm
from data_prep.config import get_pdf_collection, get_q_a_collection
from prompt import RETRIEVAL_PROMPT, DECISION_MAKING_PROMPT
from schema import RouterSchema, AgentState
from tools import search_aou_site

# ============================================================================
# TOOLS DEFINITION
# ============================================================================

@tool
def done_tool() -> str:
    """Call this tool when you have completed the task and provided a final answer to the user"""
    return "Task completed successfully"


tools = [search_aou_site, done_tool]

llm_with_tools = llm.bind_tools(tools, include_reasoning=True)
llm_with_router = llm.with_structured_output(RouterSchema, method='json_schema')


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
            r = collection.query(query_texts=query, n_results=5)
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


def router(state: AgentState) -> Dict[str, Any]:
    """
    Analyzes the query and retrieval results to determine if we have enough context
    or need to search for more information.

    Args:
        state: Current agent state

    Returns:
        Dict with updated messages and potentially modified query
    """
    query = state['query']
    context = state['retrieval_result']

    # use LLM to classify if retrieval results are sufficient
    result = llm_with_router.invoke([
        {
            "role": "system",
            "content": DECISION_MAKING_PROMPT.substitute(query=query, context=context)
        }
    ])

    reasoning = result.reasoning
    classification = result.classification

    # create AI message with reasoning
    ai_msg = AIMessage(content=f"reasoning: {reasoning}", metadata={"reasoning": reasoning})

    if classification == "fulfilled":
        # We have enough context, proceed to generate answer
        return {
            "messages": [ai_msg]
        }
    elif classification == "not_fulfilled":
        # need more information, instruct LLM to use search tool
        search_instruction = (
            f"The retrieved context is insufficient. Use the 'search_aou_site' tool "
            f"to search for information about: {query}"
        )
        return {
            "messages": [ai_msg],
            "query": search_instruction
        }
    else:
        raise ValueError(f"Invalid classification from router: {classification}")


def call_llm(state: AgentState) -> Dict[str, Any]:
    """
    Calls the LLM with tools to generate a response or make tool calls.

    Args:
        state: Current agent state

    Returns:
        Dict with the LLM's response message
    """
    result = state["retrieval_result"]
    query = state['query']

    # create system message with context and instructions
    system_message = SystemMessage(
        content=(
                RETRIEVAL_PROMPT.substitute(query=query, context=result)
        )
    )

    # invoke LLM with tools
    response = llm_with_tools.invoke(state['messages'] + [system_message])

    return {"messages": [response]}


def tool_handler(state: AgentState) -> Dict[str, Any]:
    """
    Executes tool calls requested by the LLM.

    Args:
        state: Current agent state with tool calls in last message

    Returns:
        Dict with tool result messages
    """
    last_message = state['messages'][-1]

    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {"messages": []}

    results = []

    for tool_call in last_message.tool_calls:
        try:
            # find the requested tool (case-insensitive)
            tool = next(
                (t for t in tools if t.name.lower() == tool_call["name"].lower()),
                None
            )

            if tool is None:
                error_msg = f"Error: Tool '{tool_call['name']}' not found. Available tools: {[t.name for t in tools]}"
                results.append(ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_call['id']
                ))
                continue

            # execute the tool
            tool_result = tool.invoke(tool_call['args'])

            # ensure result is a string
            if not isinstance(tool_result, str):
                tool_result = str(tool_result)

            results.append(ToolMessage(
                content=tool_result,
                tool_call_id=tool_call['id']
            ))

        except Exception as e:
            # handle any errors during tool execution
            error_msg = f"Error executing tool '{tool_call['name']}': {str(e)}"
            print(error_msg)
            results.append(ToolMessage(
                content=error_msg,
                tool_call_id=tool_call['id']
            ))

    return {"messages": results}


def increment_iteration(state: AgentState) -> Dict[str, Any]:
    """
    Increments the iteration counter to prevent infinite loops.

    Args:
        state: Current agent state

    Returns:
        Dict with incremented iteration_count
    """
    return {"iteration_count": state.get("iteration_count", 0) + 1}


# ============================================================================
# CONDITIONAL EDGE FUNCTIONS
# ============================================================================

def should_continue(state: AgentState) -> Literal["tool_handler", "__end__"]:
    """
    Determines whether to continue with tool execution or end the graph.

    Args:
        state: Current agent state

    Returns:
        "tool_handler" if tools need to be executed, "end" otherwise
    """
    # check iteration limit to prevent infinite loops
    current_iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 5)

    if current_iteration >= max_iterations:
        print(f"Max iterations ({max_iterations}) reached. Ending conversation.")
        return END

    # check last message for tool calls
    last_message = state['messages'][-1]

    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return END

    # check if done_tool was called
    has_done = any(
        tc["name"].lower() == "done_tool"
        for tc in last_message.tool_calls
    )

    if has_done:
        print("Done tool called. Ending conversation.")
        return END

    # continue with tool execution
    print(f"Tool calls detected: {[tc['name'] for tc in last_message.tool_calls]}")
    return "tool_handler"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_graph() -> CompiledStateGraph[Any, Any, Any, Any]:
    """
    Constructs the LangGraph state graph with all nodes and edges.

    Returns:
        Compiled StateGraph ready for execution
    """
    builder = StateGraph(AgentState)

    # nodes
    builder.add_node("retrieval", retrieval)
    # builder.add_node("router", router)
    builder.add_node("call_llm", call_llm)
    builder.add_node("tool_handler", tool_handler)
    builder.add_node("increment", increment_iteration)

    # flow
    builder.add_edge(START, "retrieval")
    builder.add_edge("retrieval", "call_llm")
    # builder.add_edge("router", "call_llm")

    # Conditional edge: continue to tools or end
    builder.add_conditional_edges(
        "call_llm",
        should_continue,
       {
           "tool_handler": "tool_handler",
           END: END
       }
    )

    # after tool execution, increment counter and loop back to LLM
    builder.add_edge("tool_handler", "increment")
    builder.add_edge("increment", "call_llm")

    return builder.compile()


graph = build_graph()

# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    result = graph.invoke({
        "messages": [HumanMessage("Do a deep research about Ahmed Samir")]
    })

    print("\n" + "=" * 80)
    print("CONVERSATION HISTORY")
    print("=" * 80 + "\n")

    for message in result['messages']:
        message.pretty_print()

    print("\n" + "=" * 80)
    print(f"Total iterations: {result.get('iteration_count', 0)}")
    print("=" * 80)