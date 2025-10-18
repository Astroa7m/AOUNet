from typing import List, Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.types import Command

from common.helpers import llm
from data_prep.config import get_pdf_collection, get_q_a_collection
from prompt import RETRIEVAL_PROMPT, DECISION_MAKING_PROMPT
from schema import RetrivalState, RouterSchema
from tools import search_aou_site





tools = [search_aou_site]

llm_with_tools = llm.bind_tools(tools)

llm_with_router = llm.with_structured_output(RouterSchema, method='json_schema')


def retrieval(state: MessagesState) -> RetrivalState:
    # get content of the last message which is the query of HumanMessage
    query = state["messages"][0].content

    # just for debugging
    if isinstance(query, List):
        # messages in the chat of langchain studio gives
        # [{'type': 'text', 'text': '<query here>'}]
        # so we extract it
        query = query[0]['text']

    results = []

    for collection in [get_q_a_collection(), get_pdf_collection()]:
        r = collection.query(query_texts=query, n_results=5)
        results.extend(r['documents'][0])  # returns [[data here]], where the outer contains a single list

    return {
        "retrieval_result": results,
        "query": query,
        "messages": state['messages']
    }


def call_llm(state: RetrivalState) -> MessagesState:
    result = state["retrieval_result"]
    query = state['query']

    return {
        "messages": [llm_with_tools.invoke(RETRIEVAL_PROMPT.substitute(query=query, context=result))]
    }


def router(state: RetrivalState) -> Command[Literal["tool_handler", call_llm.__name__]]:
    """Analyzes query and the context and determine the next step for the graph"""
    query = state['messages'][0].content
    print("making sure of the query: ", query)
    context = state['retrieval_result']

    result = llm_with_router.invoke(
        [
            {"role": "system", "content": DECISION_MAKING_PROMPT.substitute(query=query, context=context)}
        ]
    )

    reasoning = result.reasoning
    classification = result.classification
    goto = "call_llm"
    if classification == "fulfilled":
        # go to call llm with the same state
        update = state
    elif classification == "not_fulfilled":
        update = {
            "messages": [
                {
                    "role": "system",
                    "content": f"Use the search_aou_site tool to fulfill the following query, you should search for the thing that the user is asking about:\n{query}"
                }
            ]
        }
    else:
        raise ValueError(f"Invalid classification {classification}")

    return Command(goto=goto, update=update)


def tool_handler(state: MessagesState):
    "Performs tool call"
    print("current state from tool handler", state)
    result = []

    for tool_call in state['messages'][-1].tool_calls:
        # get the tool
        tool = list(filter(lambda x: x.name.lower() == tool_call["name"].lower(), tools))[0]
        # call the tool
        tool_res = tool.invoke(tool_call['args'])
        result.append(
            {
                "role": "tool",
                "content": tool_res,
                "tool_call_id": tool_call['id']
            }
        )
    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["tool_handler", END]:
    """Route to tool handler"""
    # get last message
    last_message = state['messages'][-1]

    # if the last message is a tool call
    if last_message.tool_calls:
        print(f"We got a call here {last_message.tool_calls}")
        for tool_call in last_message.tool_calls:
            print(f"current tool {tool_call}")
            return "tool_handler"

    return END


builder = StateGraph(MessagesState)

builder.add_node(retrieval)
builder.add_node(router)
builder.add_node(tool_handler)
builder.add_node(call_llm)

builder.add_edge(START, "retrieval")
builder.add_edge("retrieval", "router")
builder.add_edge("router", "call_llm")
builder.add_conditional_edges("call_llm", should_continue)
graph = builder.compile()

result = graph.invoke({"messages": [HumanMessage("Any information about dawood sulaiman?")]})

for message in result['messages']:
    message.pretty_print()
