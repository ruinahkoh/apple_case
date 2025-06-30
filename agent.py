from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_community.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_core.tools import Tool
import sys
import os
import operator
import numpy
# from dotenv import load_dotenv
import json
# _ = load_dotenv()
from src.utils import *


with open('src/config.json', 'r') as f:
    config = json.load(f)



class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:
    def __init__(self, model, checkpointer):
        self.system = system_prompt
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(checkpointer=checkpointer)
        self.model = model

        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        persist_directory = config["vectorstore"]["persist_directory"]

        # Load vectorstores
        education_vectordb = Chroma(
            collection_name=config["vectorstore"]["education_collection_name"],
            embedding_function=embeddings,
            persist_directory=persist_directory
        )

        inventory_vectordb = Chroma(
            collection_name=config["vectorstore"]["inventory_collection_name"],
            embedding_function=embeddings,
            persist_directory=persist_directory
        )

        # LLM & Prompts
        llm = ChatOpenAI(model_name="gpt-4o-mini")
        prompt2 = PromptTemplate(template=education_template, input_variables=["context", "question"])
        prompt3 = PromptTemplate(template=inventory_template, input_variables=["context", "question"])

        # RAG Chains
        self.education_chain = RetrievalQA.from_chain_type(
            self.model, retriever=education_vectordb.as_retriever(), chain_type_kwargs={"prompt": prompt2}
        )

        self.inventory_chain = RetrievalQA.from_chain_type(
            self.model, retriever=inventory_vectordb.as_retriever(), chain_type_kwargs={"prompt": prompt3
            }
        )

        # Define tools properly
        tools = [
            Tool(
                name="apple_education_assistant",
                func=self.education_chain.run,
                description="useful for answering questions about how and where to buy Apple products."
            ),
            Tool(
                name="apple_inventory_assistant",
                func=self.inventory_chain.run,
                description="useful for checking on the inventory of products."
            ),
        ]

       
        self.tools = {tool.name: tool for tool in tools}
        self.model = model.bind_tools(tools)

   
       

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        tool_messages = []
    
        for t in tool_calls:
            tool_name = t['name']
            tool_args = t['args']
            tool_call_id = t['id']
    
            print(f"Calling tool: {tool_name} with args: {tool_args}")
    
            result = self.tools[tool_name].invoke(tool_args)
    
            tool_messages.append(ToolMessage(tool_call_id=tool_call_id, name=tool_name, content=str(result)))
    
        return {'messages': tool_messages}