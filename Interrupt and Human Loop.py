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

# =============================================================================
# Guardrail Node - PII Detection
# =============================================================================
# PII Pattern Definitions
patterns = {
    "SSN": r'\b\d{3}-\d{2}-\d{4}\b',  # SSN: 123-45-6789
    "Credit Card": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit Card: 1234-5678-9012-3456
    "Mobile Number": r'\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',  # Mobile: +1-234-567-8900
    "Email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email: user@example.com
    "URL/Link": r'https?://[^\s]+|www\.[^\s]+'  # URL: http://example.com or www.example.com
}


def guardrail_node(state: AgentState):
    last_message = state['messages'][-1].content

    for pii_type, pattern in patterns.items():
        if re.search(pattern, last_message):
            return {
                "messages": [SystemMessage(
                    f"Request Blocked: Contains {pii_type}. Please don't share sensitive personal information")]
            }

    return {"messages": []}

# =============================================================================
# Agent Node
# =============================================================================

def agent_node(state:AgentState):

    tools = [transfer_money]

    llm_with_tools = llm.bind_tools(tools)

    system_message = SystemMessage("You are a financial assistant. use transfer_money tool to transfer the money.")

    messages = [system_message] + state['messages']

    response = llm_with_tools.invoke(messages)

    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tc in response.tool_calls:
            print(f"[AGENT] called Tool {tc.get('name', '?')} with args {tc.get('args', '?')}")
    else:
        print(f"[AGENT] Responding...")
