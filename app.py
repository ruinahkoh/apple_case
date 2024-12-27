import json
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from build_vectorstore import generate_data_store
from rag import query_rag
from src.utils import PROMPT_TEMPLATE

st.set_page_config(page_title="LangChain & Streamlit RAG")
st.title("LangChain & Streamlit RAG for News Articles")



vectordb = generate_data_store()

with open('src/config.json', 'r') as f:
    config = json.load(f)

user_input = st.text_input("Please enter your query", "")


if user_input:
    with st.spinner("Generating LLM response"):
        response_text = query_rag(user_input, vectordb)
        st.write(response_text)