from langchain_core.tools import tool
from langchain_core.messages import AnyMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.prebuilt import ToolNode, tools_condition
from OCRrecognition import OCRProcessor
from transcriber import AudioTranscriber

from dotenv import load_dotenv
import os
from langgraph.graph.message import add_messages

load_dotenv()
os.environ["OPENAI-API-KEY"] = os.getenv("OPENAI_API_KEY")


@tool
def audio_to_text(audio_bytes) -> str:
    """
    Transcribe audio bytes into text.
    
    Args:
        audio_bytes: The audio file content in bytes.

    Returns:
        A string containing the transcribed text.
    """
    transcriber = AudioTranscriber()
    return transcriber.transcribe(audio_bytes) 

@tool 
def image_to_text(image_bytes) -> str:
    """
    Extract text from image bytes using OCR.
    
    Args:
        image_bytes: The image file content in bytes.

    Returns:
        A string containing the extracted text.
    """
    ocr_processor = OCRProcessor()
    return ocr_processor.ocr(image_bytes)    

TOOLS = [audio_to_text, image_to_text]

llm = ChatOpenAI(
    model = 'gpt-3.5-turbo'
).bind_tools(tools=TOOLS)



class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent(state: State):
    return {"messages":[llm.invoke(state["messages"])]}


graph_builder = StateGraph(State)
graph_builder.add_node("agent", agent)
graph_builder.add_edge(START, "agent")
tools_node = ToolNode(TOOLS)
graph_builder.add_node("tools", tools_node)
graph_builder.add_conditional_edges(
    "agent",
    tools_condition
)
graph_builder.add_edge("tools", "agent")
graph_builder.add_edge("agent", END)
graph = graph_builder.compile()
graph_representation = graph.get_graph().draw_mermaid()

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "q"]:
        print("Goodbye!")
        break
    initial_state = {"messages": [{"role": "user", "content": user_input}]}
    for event in graph.stream(initial_state):
        print(event.values())
        for value in event.values():
            print(value['messages'])
            print("Assitant:", value["messages"].content)

