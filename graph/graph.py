from typing import Literal, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from common.helpers import llm
from data_prep.config import get_pdf_collection, get_q_a_collection
from prompt import RETRIEVAL_PROMPT, DECISION_MAKING_PROMPT, WEBSEARCH_PROMPT
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
MAX_ITERATION_COUNT = 2
llm_with_tools = llm.bind_tools(tools)
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

    if classification == "only_context":
        # We have enough context, proceed to generate answer
        return {
            "messages": [ai_msg]
        }
    elif classification == "only_websearch":
        # Context not relevant, instruct LLM to use search tool
        return {
            "messages": [ai_msg],
            "retrieval_result": []
        }
    elif classification == "both":
        # Context not enough, instruct LLM to use search tool
        return {
            "messages": [ai_msg],
            "retrieval_result": context

        }
    else:
        raise ValueError(f"Invalid classification from router: {classification}")


def call_llm(state: AgentState) -> Dict[str, Any]:
    context = state["retrieval_result"]
    query = state['query']

    # Gather ALL context including tool results
    all_context_parts = []

    if context:
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

    # Build prompt
    if all_context_parts:
        combined_context = "\n".join(all_context_parts)
        prompt = RETRIEVAL_PROMPT.substitute(query=query, context=combined_context)
    else:
        prompt = WEBSEARCH_PROMPT.substitute(query=query)

    # Add done_tool instruction
    instruction = (
        "\n\nIMPORTANT: When you have fully answered the query, "
        "call ONLY the 'done_tool' (do not call it with other tools). "
        "You can call tools multiple times across iterations if needed."
    )

    system_message = SystemMessage(content=prompt + instruction)
    response = llm_with_tools.invoke(state['messages'] + [system_message])

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

            # create a ToolMessage
            result.append(
                ToolMessage(
                    content=tool_res,
                    tool_call_id=tool_call['id']
                )
            )
        except Exception as e:
            result.append(ToolMessage(
                content=f"Error executing tool: {str(e)}",
                tool_call_id=tool_call['id']
            ))

    return {"messages": result}


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

def should_continue(state: AgentState) -> Literal["tool_handler", END]:
    """Route to tool handler"""

    # prevent infinite loops
    current_iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", MAX_ITERATION_COUNT)

    if current_iteration >= max_iterations:
        print(f"Max iterations ({max_iterations}) reached.")
        return END

    # get last message
    last_message = state['messages'][-1]

    # if the last message is a tool call
    if last_message.tool_calls:
        print(f"We got a call here {last_message.tool_calls}")
        for tool_call in last_message.tool_calls:
            print(f"current tool {tool_call}")
            if tool_call["name"] == "done_tool":
                print("We are doning")
                return END
            else:
                print("we are handling tool")
                return "tool_handler"
    return END


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
    builder.add_node("router", router)
    builder.add_node("call_llm", call_llm)
    builder.add_node("tool_handler", tool_handler)
    builder.add_node("increment", increment_iteration)

    # flow
    builder.add_edge(START, "retrieval")
    builder.add_edge("retrieval", "router")
    builder.add_edge("router", "call_llm")

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
        "messages": [HumanMessage("I am looking for information about and individual called Dawood Suliman, but I think there're plenty with the same name so idk his last name")],
        "iteration_count": 0,
        "max_iterations": MAX_ITERATION_COUNT
    })

    print("\n" + "=" * 80)
    print("CONVERSATION HISTORY")
    print("=" * 80 + "\n")

    for message in result['messages']:
        message.pretty_print()

    print("\n" + "=" * 80)
    print(f"Total iterations: {result.get('iteration_count', 0)}")
    print("=" * 80)
