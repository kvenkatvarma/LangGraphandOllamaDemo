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
