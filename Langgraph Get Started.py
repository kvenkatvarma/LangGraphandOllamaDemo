from typing_extensions import TypedDict
from langgraph.graph import START,END,StateGraph

#Creating custom state

class SimpleState(TypedDict):
    input_text: str
    output_text: str

print(SimpleState(input_text="Hello World!", output_text="Hello World!"))