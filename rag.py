from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from src.utils import *

def query_rag(query, vectordb):
    """
    Query a Retrieval-Augmented Generation (RAG) system using Chroma database and OpenAI.
    Args:
    - query_text (str): The text to query the RAG system with.
    Returns:
    - formatted_response (str): Formatted response including the generated text and sources.
    - response_text (str): The generated response text.
    """
    results = vectordb.similarity_search_with_relevance_scores(query, k=3)

    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")

    # Combine context from matching documents
    context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])

    # Create prompt template using context and query text
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    # Initialize OpenAI chat model
    model = ChatOpenAI()

    # Generate response text based on the prompt
    response_text = model.predict(prompt)

    # Get sources of the matching documents
    # sources = [doc.metadata.get("source", None) for doc, _score in results]

    # Format and return response including generated text and sources
    # formatted_response = f"Response: {response_text}\nSources: {sources}"
    
    return response_text