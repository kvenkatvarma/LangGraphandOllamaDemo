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

state = {'input_text':'hello','output_text':''}

#print(process_input(state))
def add_prefix(state:SimpleState) -> SimpleState:
    print("Current State {Prefix} ", state)
    output = " Hey I have added something here. " +  state['output_text']
    return {'output_text': output}

def add_suffix(state:SimpleState) -> SimpleState:
    print("Current State {Suffic} ", state)
    output =  state['output_text'] + " . I have added Suffix"
    return {'output_text': output}

print(add_suffix(add_prefix(process_input(state))))