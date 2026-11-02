import my_tools

from typing_extensions import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode

# store conversation in memory checkpoints. It uses RAM by default
from langgraph.checkpoint.memory import MemorySaver

#my_tools.get_weather.invoke({'location': "Mumbai"})

#my_tools.calculate.invoke({'expression': '2+2*1.4/23-34'})

MODEL_NAME = "llama3.2"

llm = ChatOllama(model=MODEL_NAME)

all_tools = [my_tools.get_weather, my_tools.calculate]

# Create Agent State
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


## Agent Node

def agent_node(state: AgentState):
    llm_with_tools = llm.bind_tools(all_tools)

    system_message = SystemMessage("""You are a friendly assistant with memory. 
                                   Use the availale tools to help the user when needed.

                                   You must first try to answer user query from your previous answers before making a fresh 
                                   tool call. Do not make answers by yourself if you are not sure.""")

    messages = [system_message] + state['messages']

    response = llm_with_tools.invoke(messages)

    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tc in response.tool_calls:
            print(f"[AGENT] called Tool {tc.get('name', '?')} with args {tc.get('args', '?')}")
    else:
        print(f"[AGENT] Responding...")

    return {'messages': [response]}


# Routing
def should_continue(state: AgentState):
    last = state['messages'][-1]

    if hasattr(last, 'tool_calls') and last.tool_calls:
        return "tools"
    else:
        return END

    # =============================================================================
    # Graph
    # =============================================================================


def create_agent():
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(all_tools))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["tools", END])

    builder.add_edge("tools", "agent")

    # add checkpoint for memory across the sessions
    checkpointer = MemorySaver()

    graph = builder.compile(checkpointer=checkpointer)

    return graph

agent = create_agent()

config = {"configurable": {"thread_id": "Varma"}}

query = "What is the current weather in Mumbai?"
result = agent.invoke({'messages': [HumanMessage(query)]}, config=config)

query = "What is the 2 + 3?"
result = agent.invoke({'messages': [HumanMessage(query)]}, config=config)
print(result)