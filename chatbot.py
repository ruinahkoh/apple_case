
from dotenv import load_dotenv
import os
import json
# from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from functools import partial
import concurrent.futures
# from langchain_pinecone import PineconeVectorStore
# import ollama
from src.utils import *
from openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import AgentType, Tool, initialize_agent, AgentExecutor
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory



with open('src/config.json', 'r') as f:
    config = json.load(f)

batch_size = config["vectorstore"]["batch_size"]   

class ChatBot:
    load_dotenv()
    def __init__(self):

        
        # define embeddings
        embeddings = HuggingFaceEmbeddings()

          # #Place vectorDB under /tmp. It can be anywhere else
        # # from langchain.vectorstores import Chroma
        persist_directory = config["vectorstore"]["persist_directory"]
        products_collection_name = config["vectorstore"]["products_collection_name"]  
        vectordb = Chroma(collection_name=products_collection_name, embedding_function=embeddings,
                                    persist_directory=persist_directory)


        print(vectordb._collection.count())

        education_collection_name = config["vectorstore"]["education_collection_name"]  
        education_vectordb = Chroma(collection_name=education_collection_name, embedding_function=embeddings,
                                    persist_directory=persist_directory)

        inventory_collection_name = config["vectorstore"]["inventory_collection_name"]  
        inventory_vectordb = Chroma(collection_name=inventory_collection_name, embedding_function=embeddings,
                                    persist_directory=persist_directory)


        # Initialize ChatOpenAI
        model_name = "gpt-4o-mini"
        llm = ChatOpenAI(model_name=model_name)

        # define prompts
        prompt = PromptTemplate(template=education_template, input_variables=["context", "question"])
        prompt2 = PromptTemplate(template=inventory_template, input_variables=["context", "question"])
        
     
      

        self.education_chain = RetrievalQA.from_chain_type(
            llm, retriever=education_vectordb.as_retriever(), chain_type_kwargs={"prompt": prompt}
        )

        self.inventory_chain = RetrievalQA.from_chain_type(
            llm, retriever=inventory_vectordb.as_retriever(), chain_type_kwargs={"prompt": prompt2
            }
        )


        # Define tools properly
        tools = [
            Tool(
                name="Apple Education Assistant",
                func=self.education_chain.run,
                description="useful for answering questions about how and where to buy Apple products."
            ),
            Tool(
                name="Apple Inventory Assistant",
                func=self.inventory_chain.run,
                description="useful for checking on the inventory of products."
            ),
        ]

        agent_memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,    # important for agent
            input_key="input"        # set input key properly
        )
       
        self.agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, memory=agent_memory)
    
        
        
# Usage example:
if __name__ == "__main__":
    chatbot = ChatBot()