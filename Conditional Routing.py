from typing_extensions import TypedDict,Literal
from langgraph.graph import StateGraph,START,END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage,SystemMessage
from pydantic import BaseModel,Field

model = "deepseek-r1"

llm = ChatOllama(model)
class SentimentAnalysis(BaseModel):
    sentiment :Literal["positive","negative"] = Field(description="Sentiment Classification either positive or negative")
    confidence: float= Field(ge=0,le=1.0, description="Confidence score from 0.0 to 1.0")
    reason:str =Field(description="Brief Explanation")

#Created a state

class SentimentState(TypedDict):
    original_tweet :str
    sentiment:str
    confidence:float
    response_tweet:str

#Adding Nodes

def analyze_sentiment(state:SentimentState):
    tweet = state["original_tweet"]
    print(f"analyzing customer tweet: {tweet}")
    structured_llm = llm.with_structured_output(SentimentAnalysis)
    messages=[
        SystemMessage("Analyze the sentiment and provide the structured output. use 0 to 1 scale for "
                      "confidence.lower is negative and higher is positive"),
        HumanMessage(tweet)
    ]
    analysis = structured_llm.invoke(messages)
    print(f"Sentiment Analysis is done : \n {analysis}")
    return {'sentiment': analysis.sentiment,
            'confidence': analysis.confidence}