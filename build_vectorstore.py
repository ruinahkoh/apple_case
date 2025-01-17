import pandas as pd
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import streamlit as st
from src.utils import *

##
from functools import partial


with open('src/config.json', 'r') as f:
    config = json.load(f)


# def generate_datastore():

file_path = config["vectorstore"]["file_path"]
batch_size = config["vectorstore"]["batch_size"]   



def add_to_chroma_database(batch, vectordb):
    vectordb.add_documents(documents=batch)

# file = 'data/tech_news_articles.csv'

def generate_data_store():
    """Function to generate chroma vector store uing documents"""


    documents = load_data(file_path)
    print(len(documents))
    # define embeddings
    embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small")

    # #Place vectorDB under /tmp. It can be anywhere else
    # # from langchain.vectorstores import Chroma
    persist_directory = config["vectorstore"]["persist_directory"]
    collection_name = config["vectorstore"]["collection_name"]  
    vectordb = Chroma(collection_name=collection_name, embedding_function=embeddings,
                                 persist_directory=persist_directory)
    # vectordb = Chroma.from_documents(documents=list(documents[0:100]), embedding=embeddings,
    #                              persist_directory=persist_directory)

    data_list = form_batch(documents, batch_size)
    add_to_chroma_database_with_param = partial(add_to_chroma_database, vectordb=vectordb)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            executor.map(add_to_chroma_database_with_param,  data_list)

    # save_to_chroma(data_list)

    print(vectordb._collection.count())

    return vectordb

if __name__ == "__main__":
    generate_data_store()

