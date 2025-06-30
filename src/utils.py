
import concurrent.futures
import logging
import json
from langchain_community.document_loaders import CSVLoader
from langchain.agents import tool

LOGGER = logging.getLogger('logger')


with open('src/config.json', 'r') as f:
    config = json.load(f)



def add_to_chroma_database(batch, vectordb):
    vectordb.add_documents(documents=batch)

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
    


education_template = """
You are an Apple products educator, users will ask you questions on how to get the best deals for apple products or comparisons between different products. Use the following piece of context to answer the question.

Only answer questions related to product comparisons and specifications, buying strategies, deals and discounts.
If you don't know the answer, just say you don't know.

Keep your replies concise and limited to 4 sentences. 

Context: {context}
Question: {question}
Answer:
"""


inventory_template = """
You are an Apple store sales assistant. Users will ask you questions about Apple products and you will make recommendations on what apple product suit them based on their budget or preferences.
You should check the inventory and only recommend products which are in stock. Do not tell the customer the amount of stock, just whether it is in stock or not.

A general rule of thumb when recommending products under a certain budget (150-200) you can recommend accessories

Offer the customer a link to purchase it if it is in stock
Only use the context to answer. If you don't know the answer, just say you don't know.

Keep your replies concise and limited to 4 sentences. 

Context: {context}
Question: {question}
Answer:
"""