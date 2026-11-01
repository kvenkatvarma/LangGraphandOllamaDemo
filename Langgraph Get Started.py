from typing_extensions import TypedDict
from langgraph.graph import START,END,StateGraph

#Creating custom state

class SimpleState(TypedDict):
    input_text: str
    output_text: str

#Creating custom node and we have to pass state class as a parameter as below

def process_input(state:SimpleState) -> SimpleState:
    output_text = state['input_text'].upper()
    return {'output_text': output_text}

state = {'input_text':'Venkat','output_text':''}

print(process_input(state))