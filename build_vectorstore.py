import pandas as pd
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from functools import partial
import concurrent.futures
import streamlit as st
from src.utils import *

##
from functools import partial


with open('src/config.json', 'r') as f:
    config = json.load(f)


# def generate_datastore():

apple_products = config["vectorstore"]["product_file"]
batch_size = config["vectorstore"]["batch_size"]   
apple_education = config["vectorstore"]["education_file"]
apple_inventory = config["vectorstore"]["inventory_file"]
products_collection_name = config["vectorstore"]["products_collection_name"]  
education_collection_name = config["vectorstore"]["education_collection_name"]  
inventory_collection_name = config["vectorstore"]["inventory_collection_name"]  


def add_to_chroma_database(batch, vectordb):
    vectordb.add_documents(documents=batch)


def generate_data_store(file_path, collection_name):
    """Function to generate chroma vector store uing documents"""
    documents = load_data(file_path)
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=4)
    docs = text_splitter.split_documents(documents)
    # define embeddings

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs)
    # embeddings = HuggingFaceEmbeddings()

    # #Place vectorDB under /tmp. It can be anywhere else
    # # from langchain.vectorstores import Chroma
    persist_directory = config["vectorstore"]["persist_directory"]
    vectordb = Chroma(collection_name=collection_name, embedding_function=embeddings,
                                 persist_directory=persist_directory)
    
    if vectordb._collection.count() == 0:
        data_list = form_batch(docs, batch_size)
        add_to_chroma_database_with_param = partial(add_to_chroma_database, vectordb=vectordb)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                executor.map(add_to_chroma_database_with_param,  data_list)

    print(vectordb._collection.count())

    return vectordb

if __name__ == "__main__":
    # generate_data_store(apple_products, products_collection_name)
    generate_data_store(apple_education, education_collection_name)
    generate_data_store(apple_inventory, inventory_collection_name)

