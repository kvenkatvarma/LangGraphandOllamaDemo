from typing_extensions import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
import my_tools

model = "llama3.2"

llm = ChatOllama(model=model)

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

all_tools = [my_tools.get_weather, my_tools.calculate]

# Agent node
def agent_node(state: AgentState):
    llm_with_tools = llm.bind_tools(all_tools)
    messages = state['messages']  # This is now a list of messages
    response = llm_with_tools.invoke(messages)
    # LangGraph nodes should return a dictionary updating the state
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tc in response.tool_calls:
            print(f"[AGENT] called Tool {tc.get('name', '?')} with args {tc.get('args', '?')}")
    else:
        print(f"[AGENT] Responding...")

    return {"messages": [response]}

# Fix 1: Pass the message inside a list
state = {'messages': [HumanMessage("Hi, What is 2+2")]}

result = agent_node(state)

def should_continue(state: AgentState):
    last = state['messages'][-1]

    if hasattr(last, 'tool_calls') and last.tool_calls:
        return 'tools'
    else:
        return END
