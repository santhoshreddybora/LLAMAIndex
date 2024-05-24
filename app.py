from llama_index.core import SimpleDirectoryReader

from llama_index.core import VectorStoreIndex
import os
from langchain.chains import ConversationalRetrievalChain 
from langchain.llms.openai import OpenAI
from dotenv import load_dotenv 
import streamlit as st

load_dotenv()
#get API key
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


# load the data 
reader=SimpleDirectoryReader('data')
documents=reader.load_data()

# Create the index
index=VectorStoreIndex(documents)
index.save_to_disk('index.json')


# Load the index
index = VectorStoreIndex.load_from_disk('index.json')
llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

chatbot = ConversationalRetrievalChain(llm=llm, retriever=index.as_retriever())

def get_response(question):
    return chatbot.run(question)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Streamlit UI
st.title("PDF Knowledge Chatbot")
st.write("Ask questions based on the content of the provided PDFs.")

user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input:
        response = get_response(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))
    else:
        st.warning("Please enter a question.")

for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.write(f"**{sender}:** {message}")
    else:
        st.write(f"**{sender}:** {message}")