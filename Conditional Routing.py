from typing_extensions import TypedDict,Literal
from langgraph.graph import StateGraph,START,END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage,SystemMessage
from pydantic import BaseModel,Field

model = "deepseek-r1"

llm = ChatOllama(model=model)
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


#print(analyze_sentiment(state))

def generate_positive_response(state:SentimentState):
    print(f"Current state in Positive response node : {state}")
    messages = [
        SystemMessage(""" Generate a warm response in this positive tweet under 230 chars.
        confidence : {state['confidence']}. High confidence means be enthusiastic otherwise be friendly."""),
        HumanMessage(state['original_tweet'])
    ]
    response = llm.invoke(messages)
    return {'response_tweet': response.content.strip()}

def generate_negative_response(state:SentimentState):
    print(f"Current state in negative response node : {state}")
    messages = [
        SystemMessage(""" Generate and empathetic response to this negative tweet under 250 chars.
        if confidence : {state['confidence']} is ery low then be empathetic otherwise be understanding"""),
        HumanMessage(state['original_tweet'])
    ]
    response = llm.invoke(messages)
    return {'response_tweet': response.content.strip()}

def route_by_sentiment(state:SentimentState):
    if state['sentiment'] == 'positive':
        return "positive_response"
    else:
        return "negative_response"

def create_router_graph():
    builder = StateGraph(SentimentState)

    #add nodes
    builder.add_node("analyze",analyze_sentiment)
    builder.add_node("positive_response",generate_positive_response)
    builder.add_node("negative_response",generate_negative_response)

    #add edges
    builder.add_edge(START,"analyze")
    builder.add_conditional_edges("analyze",route_by_sentiment,["positive_response","negative_response"])

    builder.add_edge("positive_response",END)
    builder.add_edge("negative_response",END)

    graph = builder.compile()
    return graph

graph = create_router_graph()

state = {'original_tweet': 'I am very Happy as I earned my AI Certificate'}
graph.invoke(state)