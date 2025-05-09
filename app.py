import json
import streamlit as st
# from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from build_vectorstore import generate_data_store
from chatbot import ChatBot
from langchain.schema import HumanMessage, AIMessage

st.set_page_config(page_title="Apple Products Chatbot")
st.title("Apple Products Chatbot")



# with open('src/config.json', 'r') as f:
#     config = json.load(f)

# apple_education = config["vectorstore"]["education_file"]
# education_collection_name = config["vectorstore"]["education_collection_name"] 




# vectordb = generate_data_store(apple_education, education_collection_name)



# user_input = st.text_input("Please enter your query", "")
# if user_input:
#     with st.spinner("Generating LLM response"):
#         response_text = query_rag(user_input, vectordb)
#         st.write(response_text)


bot = ChatBot()
def convert_session_to_messages(session_messages):
    converted = []
    for m in session_messages:
        if m["role"] == "user":
            converted.append(HumanMessage(content=m["content"]))
        else:
            converted.append(AIMessage(content=m["content"]))
    return converted


#Function for generating LLM response
def generate_response(input):
    # result = bot.rag_chain.invoke(input)
  

    result = bot.agent.invoke({"input": input})

   
    return result


#Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm Apple Store Chatbot. How can I assist you today?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)


# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Getting your answer from mystery stuff.."):
            response = generate_response(input) 
            result_text = response['output']
            st.write(result_text) 
    message = {"role": "assistant", "content": response["output"]}
    st.session_state.messages.append(message)