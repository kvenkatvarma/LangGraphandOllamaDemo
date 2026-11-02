from typing_extensions import TypedDict,Annotated
import operator
from langgraph.graph import StateGraph,START,END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage,SystemMessage
from langgraph.prebuilt import ToolNode

model = "llama3.2"

llm = ChatOllama(model=model)

class AgentState(TypedDict):
    messages: Annotated[list,operator.add]