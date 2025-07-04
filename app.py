import streamlit as st
from langgraphchatbot import Chatbot
from agent import Agent  # Your previously defined Agent class
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import os

working_dir = os.getcwd()
# you can you other types of database like postgres rather than in memeory
conn = sqlite3.connect("chat_memory.db", check_same_thread=False)  
memory = SqliteSaver(conn)


st.title("🍎 Apple Sales Assistant")

# Initialize OpenAI model and agent
if "chatbot" not in st.session_state:
    model = ChatOpenAI(model="gpt-4o-mini")
    agent = Agent(model=model, checkpointer=memory)  # make sure memory is defined or replace it
    st.session_state.chatbot = Chatbot(agent=agent)
    st.session_state.messages = []  # stores {"role": ..., "content": ...}

# Display past messages from chatbot history
for msg in st.session_state.chatbot.history:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Handle new user input
if prompt := st.chat_input("Ask me about Apple products or inventory..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send user input through chatbot (adds both user and assistant to history)
    st.session_state.chatbot.send(prompt)

   # Get the latest assistant response
    response_msg = st.session_state.chatbot.history[-1]
    if response_msg.type == "ai":
        with st.chat_message("assistant"):
            with st.spinner("Getting your answer from mystery stuff.."):
                st.markdown(response_msg.content)