from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, BaseMessage, ToolMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from typing import Annotated, Sequence, TypedDict, Literal
from typing_extensions import TypedDict
from langgraph.prebuilt import ToolNode, tools_condition
from OCRrecognition import OCRProcessor
from transcriber import AudioTranscriber
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os
from langgraph.graph.message import add_messages
import functools

load_dotenv()
os.environ["OPENAI-API-KEY"] = os.getenv("OPENAI_API_KEY")


@tool
def audio_to_text(file_path) -> str:
    """
    Process an audio file from its path and transcribe its content.
    
    Args:
        file_path: The audio file.

    Returns:
        A string containing the transcribed text.
    """
    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    transcriber = AudioTranscriber()
    return transcriber.transcribe_audio(audio_bytes) 

@tool 
def image_to_text(file_path: str) -> str:
    """
    Process an image file from its path and extract its text.
    
    Args:
        file_path: The image file.

    Returns:
        A string containing the extracted text.

    """
    with open(file_path, "rb") as f:
        image_bytes = f.read()

    ocr_processor = OCRProcessor()
    return ocr_processor.ocr_process(image_bytes)    

TOOLS = [audio_to_text, image_to_text]

llm = ChatOpenAI(
    model = 'gpt-3.5-turbo'
).bind_tools(tools=TOOLS)

def create_tool_selector_agent():
    tools = [audio_to_text, image_to_text]
    llm = ChatOpenAI(model="gpt-3.5-turbo").bind_tools(tools)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a tool selector AI agent. Your job is to decide whether to use the 'audio_to_text' or 'image_to_text' tool "
                "to process the given file. Once processed, return the extracted text."
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    return prompt | llm


# Segundo agente: Extrae información personal
def create_personal_data_extractor_agent():
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a personal data extractor AI agent. Your job is to extract Name, Surname, and DNI from the given text. "
                "The DNI is an identifier with 8 digits followed by a letter."
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    return prompt | llm


# Entrada principal
def process_input(file_bytes: bytes) -> dict:
    """
    Process the file bytes using two agents: one for selecting tools and another for extracting personal data.

    Args:
        file_bytes: The file content in bytes.

    Returns:
        A dictionary with extracted personal data.
    """
    # Inicializar los agentes
    tool_selector_agent = create_tool_selector_agent()
    personal_data_extractor_agent = create_personal_data_extractor_agent()

    # Agente 1: Selección de herramienta y transcripción
    tool_selector_input = {
        "messages": [
            {
                "role": "user",
                "content": "Process these bytes to extract the text."
            },
            {"role": "user", "content": str(file_bytes)},
        ]
    }
    extracted_text = tool_selector_agent.invoke(tool_selector_input)["content"]

    # Agente 2: Extracción de datos personales
    personal_data_input = {
        "messages": [
            {
                "role": "user",
                "content": f"Extract personal data from the following text:\n\n{extracted_text}"
            }
        ]
    }
    personal_data = personal_data_extractor_agent.invoke(personal_data_input)["content"]

    return personal_data


## Estado Agente
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    sender: str

def agent_node(state, agent, name):
    result = agent.invoke(state)
    if isinstance(result, ToolMessage):
        # Si el agente invoca una herramienta, devuelve el resultado como está
        pass
    else:
        # Si el resultado es un mensaje AI, lo ajustamos con el nombre del agente
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        "sender": name,
    }

# Nodos para los dos agentes
tool_selector_node = functools.partial(agent_node, agent=create_tool_selector_agent(), name="ToolSelector")
personal_data_node = functools.partial(agent_node, agent=create_personal_data_extractor_agent(), name="PersonalDataExtractor")
tool_node = ToolNode(tools=TOOLS)

def router(state) -> Literal["call_tool", "__end__", "continue"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        # Si el último mensaje incluye una invocación de herramienta, llama a la herramienta
        return "call_tool"
    return "continue"


workflow = StateGraph(State)

# Agregar nodos
workflow.add_node("ToolSelector", tool_selector_node)
workflow.add_node("PersonalDataExtractor", personal_data_node)
workflow.add_node("call_tool", tool_node)

# Agregar edges condicionales
workflow.add_conditional_edges(
    "ToolSelector",
    router,
    {"continue": "PersonalDataExtractor", "call_tool": "call_tool"}
)
workflow.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],
    {"ToolSelector": "ToolSelector"}
)

# Agregar transiciones estándar
workflow.add_edge(START, "ToolSelector")
workflow.add_edge("PersonalDataExtractor", END)

# Compilar el grafo
graph = workflow.compile()
graph_representation = graph.get_graph().draw_mermaid()


file_path = "temp/DNI_Enrique_Delgado.png"
# Estado inicial
initial_state = {
    "messages": [
        HumanMessage(content=f"This is the file path: {file_path}")
    ],
    "sender": "user",
}

# Ejecutar grafo
events = graph.stream(initial_state, {"recursion_limit": 150})

# Imprimir cada evento
for event in events:
    print(event)
    print("----")

