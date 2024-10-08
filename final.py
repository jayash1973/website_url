import os
import requests
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv("apikey.env")
together_api_key = os.getenv("TOGETHER_API_KEY")

if not together_api_key:
    st.error("Together API key is missing. Please check your apikey.env file.")

# Fetch content from URL
def fetch_web_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            return soup.get_text()
        else:
            st.error(f"Failed to fetch content from URL. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"An error occurred while fetching the URL: {e}")
        return None

# Fetch embeddings from Together API
def get_together_embeddings(documents):
    headers = {
        "Authorization": f"Bearer {together_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "input": [doc.page_content for doc in documents],  # input expects text, check API documentation
        "model": "embedding-ada-002"  # Example of using an embedding model
    }

    response = requests.post("https://api.together.xyz/v1/embeddings", headers=headers, json=data)

    if response.status_code == 200:
        embeddings = response.json().get("data")
        if embeddings:
            return embeddings
        else:
            st.error("No embeddings returned from Together API.")
    else:
        st.error(f"Error fetching embeddings from Together API. Status Code: {response.status_code}")
    return None

# Create vector store from web content
def get_vectorstore_from_url(url):
    web_content = fetch_web_content(url)
    if web_content is None:
        return None

    documents = [Document(page_content=web_content)]
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(documents)

    embeddings = get_together_embeddings(document_chunks)
    if embeddings is None:
        st.error("Failed to fetch embeddings, cannot proceed with FAISS vector store.")
        return None

    vector_store = FAISS.from_documents(document_chunks, embeddings)
    return vector_store

# Function to handle Together chat API responses
def together_chat_api(prompt):
    headers = {
        "Authorization": f"Bearer {together_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt
    }
    response = requests.post("https://api.together.xyz/v1/chat", headers=headers, json=data)

    if response.status_code == 200:
        return response.json().get("reply")
    else:
        st.error(f"Error getting response from Together API. Status code: {response.status_code}")
        return None

# Context retriever chain
def get_context_retriever_chain(vector_store):
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Generate a search query based on the conversation.")
    ])
    retriever_chain = create_history_aware_retriever(together_chat_api, retriever, prompt)
    return retriever_chain

# Conversational RAG chain
def get_conversational_rag_chain(retriever_chain):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(together_chat_api, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# Generate response
def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    return response['answer']

# App config
st.set_page_config(page_title="Chat with Websites", page_icon="🤖")
st.title("Chat with Websites")

# Sidebar input
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if website_url:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

    user_query = st.chat_input("Type your message here...")
    if user_query:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            st.chat_message("AI", content=message.content)
        elif isinstance(message, HumanMessage):
            st.chat_message("Human", content=message.content)
