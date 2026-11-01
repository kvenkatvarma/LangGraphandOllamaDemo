from typing_extensions import TypedDict,Literal
from langgraph.graph import StateGraph,START,END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage,SystemMessage
from pydantic import BaseModel,Field

model = "deepseek-r1"

class SentimentAnalysis(BaseModel):
    sentiment :Literal["positive","negative"] = Field(description="Sentiment Classification either positive or negative")
    confidence: float= Field(ge=0,le=1.0, description="Confidence score from 0.0 to 1.0")
    reason:str =Field(description="Brief Explanation")