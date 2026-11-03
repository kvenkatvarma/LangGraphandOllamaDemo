from typing_extensions import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.types import Command, interrupt

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
import os
import re

# Configuration
MODEL_NAME = "llama3.2"
llm = ChatOllama(model=MODEL_NAME)


# =============================================================================
# State Definition
# =============================================================================
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# =============================================================================
# Transfer Money Tool
# =============================================================================
@tool
def transfer_money(amount:int, recipient:str):
    """
    Transfer money. Large transfers require approval.

    Args:
        amount: Amount in dollars
        recipient: Recipient name
    """

    # let's ask user approval if money is more than 1000
    if amount > 1000:
        approval = interrupt(
            {
                "type": "approval_required",
                "amount": amount,
                "recipient": recipient
            }
        )

        if approval.get("decision") != "approve":
            return "Transfer is cancelled!"

    return f"Transferred {amount} to {recipient}"

#print(transfer_money.invoke({'amount':100000,'recipient':'Venkat'}))