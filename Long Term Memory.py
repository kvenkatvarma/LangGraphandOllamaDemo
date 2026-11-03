from dotenv import load_dotenv

load_dotenv()
from typing_extensions import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END

# short term memory persistence
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver

# long term memory persistence
from langgraph.store.postgres import PostgresStore
from langgraph.store.sqlite import SqliteStore

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
import psycopg
import os
import my_tools

# Configuration

MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"

llm = ChatOllama(model=MODEL_NAME)

# =============================================================================
# Store and Saver Setup
# =============================================================================

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

def embed_texts(texts: list[str]) -> list[list[float]]:

    return embeddings.embed_documents(texts)

db_url = os.getenv("POSTGRESQL_URL")

checkpointer_conn = psycopg.connect(db_url, autocommit=True, prepare_threshold=0)
checkpointer = PostgresSaver(checkpointer_conn)

store_conn = psycopg.connect(db_url, autocommit=True, prepare_threshold=0)
store = PostgresStore(store_conn, index = {'embed': embed_texts, 'dims': 768})


# first time setup
checkpointer.setup()
store.setup()

# =============================================================================
# Demo 1: Store and Retrieve Memories
# =============================================================================

# Memory Operations: put(), get(), search(), delete()
"""
user_id = "VenkatVarma"
namespace = (user_id, "preferences")

store.put(namespace, "food", {"diet": "veg",
                              "likes": ["pasta", "pizza", "veggies"]})


store.put(namespace, "color", {"favorite": "blue",
                               "dislike": "brown"})

store.put(namespace, "work", {
    "role": "Data Scientist",
    "interests": ["machine learning", "ai", "gen ai", "agents"]
})

#print(store.get(namespace, "color"))
#store.delete(namespace, "color")

# =============================================================================
# Demo 2: Semantic Search
# =============================================================================

query = "What does Venkat Varma like to eat?"
results = store.search(namespace, query=query, limit=1)
print(results)

query = "What does Venkat Varma like in colors?"
results = store.search(namespace, query=query, limit=1)
print(results)
"""

# State Definition

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    user_id: str

# =============================================================================
# Tools for Memory Management
# =============================================================================

# save user memory and get user memory

@tool
def save_user_memory(user_id:str, category:str, information:dict) -> str:
    """
    Save user preference or information to long-term memory.

    Args:
        user_id: User identifier
        category: Category of information (e.g., 'food', 'work', 'hobbies', 'schedule', 'location')
        information: Dictionary containing the information to save
    """

    namespace = (user_id, "preferences")

    store.put(namespace, category, information)

    return f"Saved {category} preferences."


# if you want to use store.search
# @tool
# def get_user_memory(user_id:str, query:str) -> str:
# store.search(namespace, query)
@tool
def get_user_memory(user_id: str, category: str) -> str:
    """
        Retrieve user preference or information from long-term memory.

        Args:
            user_id: User identifier
            category: Category of information to retrieve (e.g., 'food', 'work', 'hobbies')
        """

    namespace = (user_id, "preferences")

    item = store.get(namespace, category)

    if item:
        return f"{category}: {item.value}"
    else:
        return f"No '{category}' information found!"


#print(get_user_memory.invoke({"user_id": 'VenkatVarma', "category": 'color'}))
all_tools = [my_tools.get_weather, my_tools.calculate]


def agent_node(state: AgentState):
    store_conn = psycopg.connect(db_url, autocommit=True, prepare_threshold=0)
    store = PostgresStore(store_conn, index={'embed': embed_texts, 'dims': 768})

    user_id = state.get("user_id", "unknown")
    namespace = (user_id, "preferences")

    last_message = state['messages'][-1].content
    memories = store.search(namespace, query=last_message, limit=3)

    # build context memory for personalized answer
    context_line = []
    for mem in memories:
        text = f" -{mem.key}: {mem.value}"
        context_line.append(text)

    memory_text = "\n\n".join(context_line) if context_line else "No user preferences stored yet!"

    print(f"User Memory Retrieved: \n{memory_text}\n")

    tools = all_tools + [save_user_memory, get_user_memory]

    llm_with_tools = llm.bind_tools(tools)

    system_prompt = SystemMessage(f"""
                        You are a helpful assistant with long-term memory capabilities and access to utility tools.

                            User ID: {user_id}
                            Current User Memories:
                            {memory_text}

                            MEMORY TOOLS USAGE:

                            1. save_user_memory: Use when user shares NEW information
                            - Always pass user_id: "{user_id}"
                            - Food preferences (diet, likes, dislikes, allergies)
                            - Work information (role, company, interests)
                            - Hobbies and activities
                            - Schedule and availability
                            - Location and timezone

                            2. get_user_memory: Use when you need to recall specific category
                            - Always pass user_id: "{user_id}"
                            - When answering questions about past preferences
                            - When user asks "what do you know about me?"
                            - When making recommendations based on preferences

                            UTILITY TOOLS USAGE:

                            3. get_weather: Use to retrieve current weather information
                            - Pass location as parameter (city name, zip code, or coordinates)
                            - Use when user asks about weather conditions
                            - Use when planning activities that depend on weather
                            - Examples: "What's the weather in London?", "Will it rain today?"

                            4. calculate: Use to perform mathematical calculations
                            - Pass mathematical expression as string parameter
                            - Supports basic arithmetic (+, -, *, /)
                            - Supports advanced operations (powers, roots, trigonometry)
                            - Use when user needs numerical computations
                            - Examples: "What's 15% of 250?", "Calculate the area of a circle with radius 5"

                            GUIDELINES:
                            - Always save when user shares personal information
                            - Retrieve specific categories when needed for context
                            - Use semantic search results shown above for general context
                            - Use get_weather when location-based weather info is needed
                            - Use calculate for any mathematical operations or conversions
                            - Be conversational and natural when using all tools
                            - Combine tools when appropriate (e.g., weather + saved location preference)
                        """)

    messages = [system_prompt] + state['messages']

    response = llm_with_tools.invoke(messages)

    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tc in response.tool_calls:
            print(f"[AGENT] called Tool {tc.get('name', '?')} with args {tc.get('args', '?')}")
    else:
        print(f"[AGENT] Responding...")

    return {'messages': [response]}