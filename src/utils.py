
import concurrent.futures
import logging
import json
from langchain_community.document_loaders import CSVLoader


LOGGER = logging.getLogger('logger')


with open('src/config.json', 'r') as f:
    config = json.load(f)

collection_name = config["vectorstore"]["collection_name"]  


def load_data(file_path):
    LOGGER.info('Loading data into documents')
    loader = CSVLoader(file_path=file_path)
    documents = loader.load()
    return documents


def batch_process(documents_arr, batch_size,):
    for i in range(1, len(documents_arr), batch_size):
        batch = documents_arr[i:i + batch_size]
        add_to_chroma_database(batch)


def add_to_chroma_database(batch):
    vectordb.add_documents(documents=batch)


def save_to_chroma(documents):
    #this allows parallel processing and faster processing for inserting the articles into chroma
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(add_to_chroma_database, documents)


def form_batch(documents_arr, batch_size):
    data_list = []
    for i in range(1, len(documents_arr), batch_size):
        data_list.append(documents_arr[i:i + batch_size])
    return data_list
    

PROMPT_TEMPLATE = """
You are a helpful research analyst, based on the context: 
{context}
 - -
Answer the question:{question} 
"""
