This repo demonstrates the ability to use Retrival Augmented Generation using a chatbot to search for apple products, learn about buying strategies for apple products

1. Mock data of apple products/inventory and apple educational content
2. Data Processing and feature selection (notebooks)
3. Constructing a vectordb using Chroma
4. Using ChatGPT to query vectordb and return relevant products/inventory or educational content
4. Streamlit Front end to add user input 

### How to build_vectorstore
python3 build_vectorstore.py

### How to install packages 
pip install -r requirements.txt

### How to run the app
streamlit run app.py

### Required for calling OPENAI: set your ‘OPENAI_API_KEY’ Environment Variable
https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety



