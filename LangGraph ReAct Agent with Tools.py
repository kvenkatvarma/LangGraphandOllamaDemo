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
    return {"messages": [response]}

# Fix 1: Pass the message inside a list
state = {'messages': [HumanMessage("Hi")]}

result = agent_node(state)
print(result)