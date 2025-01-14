from langchain_core.tools import Tool
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import Annotated, Sequence, TypedDict, Literal
from langgraph.graph.message import add_messages
from OCRrecognition import OCRProcessor
from transcriber import AudioTranscriber
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
import functools
import os
from dotenv import load_dotenv


class Agent:
    def __init__(self):
        self.tools = [
            Tool(
                name="audio_to_text",
                func=self.audio_to_text,
                description="Transcribe audio files into text."
            ),
            Tool(
                name="image_to_text",
                func=self.image_to_text,
                description="Extracts text from image files."
            )
        ]
        self.tool_selector_agent = self.create_tool_selector_agent()
        self.personal_data_extractor_agent = self.create_personal_data_extractor_agent()
        self.workflow = self.create_workflow()

    def audio_to_text(self, file_path: str) -> str:
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

    def image_to_text(self, file_path: str) -> str:
        """
        Process an image file from its path and extract its text.

        Args:
        file_path: The image file.

    Returns:
        A string containing the transcribed text.
        """
        with open(file_path, "rb") as f:
            image_bytes = f.read()
        ocr_processor = OCRProcessor()
        return ocr_processor.ocr_process(image_bytes)

    def create_tool_selector_agent(self):
        llm = ChatOpenAI(model="gpt-3.5-turbo").bind_tools(self.tools)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a tool selector AI agent. Your job is to decide whether to use 'audio_to_text' or 'image_to_text'. "
                    "Once processed, return the extracted text."
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        return prompt | llm

    def create_personal_data_extractor_agent(self):
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a personal data extractor AI agent. Extract Name, Surname, and DNI. "
                    "DNI is an identifier with 8 digits followed by a letter."
                    "If detected language is Spanish Surname has two words."
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        return prompt | llm

    def create_workflow(self):
        class State(TypedDict):
            messages: Annotated[Sequence[BaseMessage], add_messages]
            sender: str

        def agent_node(state, agent, name):
            result = agent.invoke(state)
            if isinstance(result, ToolMessage):
                pass
            else:
                result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
            return {"messages": [result], "sender": name}

        tool_selector_node = functools.partial(
            agent_node, agent=self.tool_selector_agent, name="ToolSelector"
        )
        personal_data_node = functools.partial(
            agent_node, agent=self.personal_data_extractor_agent, name="PersonalDataExtractor"
        )
        tool_node = ToolNode(self.tools)

        def router(state) -> Literal["call_tool", "__end__", "continue"]:
            messages = state["messages"]
            last_message = messages[-1]
            if last_message.tool_calls:
                # Si el ultimo mensaje incluye una invocacion de herramienta, llama a la herramienta
                return "call_tool"
            return "continue"

        workflow = StateGraph(State)
        workflow.add_node("ToolSelector", tool_selector_node)
        workflow.add_node("PersonalDataExtractor", personal_data_node)
        workflow.add_node("call_tool", tool_node)

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

        workflow.add_edge(START, "ToolSelector")
        workflow.add_edge("PersonalDataExtractor", END)

        return workflow.compile()

    def process(self, file_path: str) -> dict:
        initial_state = {
            "messages": [
                HumanMessage(content=f"Process this file at path: {file_path}.")
            ],
            "sender": "user",
        }

        response = {}
        events = self.workflow.stream(initial_state, {"recursion_limit": 150})
        for event in events:
            for node, data in event.items():
                if "messages" in data:
                    for message in data["messages"]:
                        if hasattr(message, "content"):
                            response[node] = message.content
        return response

if __name__ == '__main__':
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    file_path = "temp/tmpnlomzk0e.wav"
    agent = Agent()
    
    response = agent.process(file_path)
    print(response)