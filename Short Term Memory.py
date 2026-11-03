from typing_extensions import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
import os
import sqlite3
import my_tools

load_dotenv()
MODEL_NAME = "llama3.2"

llm = ChatOllama(model=MODEL_NAME)


os.makedirs('db', exist_ok=True)

db_path = "db/checkpoints.db"

# my_tools.get_weather.invoke({'location': "Mumbai"})

#my_tools.calculate.invoke({'expression': '2+2*1.4/23-34'})

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


state = {"messages": [HumanMessage("Hi")]}
#result = agent_node(state)


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
def create_agent(checkpointer):

    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(all_tools))


    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["tools", END])

    builder.add_edge("tools", "agent")

    graph = builder.compile(checkpointer=checkpointer)

    return graph

#checkpointer = MemorySaver()
#agent = create_agent(checkpointer)

conn = sqlite3.connect(db_path,check_same_thread=False)
checkpointer = SqliteSaver(conn)

def chat(agent, query, thread_id):

    config = {"configurable": {"thread_id": thread_id}}

    for chunk in agent.stream({'messages': [query]}, config=config):

        if 'agent' in chunk:
            chunk = chunk.get('agent')
        else:
            chunk = chunk.get('tools')

        if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
            for tc in chunk.tool_calls:
                print(f"[AGENT] called Tool {tc.get('name', '?')} with args {tc.get('args', '?')}")
        else:
            print(f"[AGENT/ToolMessage] Responding.\n{chunk['messages'][0].content}")

        # print(chunk)

agent = create_agent(checkpointer)

query = "What is 2+3"
chat(agent, query, "Varma")

#https://console.neon.tech/

conn = psycopg.connect(os.getenv("POSTGRESQL_URL"),autocommit=True,prepare_threshold=0)
checkpointer = PostgresSaver(conn)
checkpointer.setup()

agent = create_agent(checkpointer)

query = "What is 2+3"
chat(agent, query, "Varma")
