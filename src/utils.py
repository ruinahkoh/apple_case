
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


def form_batch(documents_arr, batch_size):
    data_list = []
    for i in range(0, len(documents_arr), batch_size):
        data_list.append(documents_arr[i:i + batch_size])
    return data_list
    

PROMPT_TEMPLATE = """
You are a helpful research analyst, based on the context: 
{context}
 - -
Answer the question:{question} 
"""
