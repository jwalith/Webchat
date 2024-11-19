
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st

load_dotenv()

def create_vector_store_from_url(url):
    # Load the webpage content as documents
    webpage_loader = WebBaseLoader(url)
    loaded_document = webpage_loader.load()
    
    # Split the document into smaller chunks
    splitter = RecursiveCharacterTextSplitter()
    document_chunks = splitter.split_documents(loaded_document)
    
    # Create a vector store from the document chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    return vector_store

def initialize_retriever_chain(vector_store):
    llm_model = ChatOpenAI()
    retriever = vector_store.as_retriever()
    
    prompt_template = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="conversation_history"),
        ("user", "{query}"),
        ("user", "Based on the conversation above, generate a query to find information relevant to the discussion.")
    ])
    
    retriever_chain = create_history_aware_retriever(llm_model, retriever, prompt_template)
    return retriever_chain

def setup_conversational_chain(retriever_chain):
    llm_model = ChatOpenAI()
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Provide answers to the user's questions based on the following context:\n\n{context}"),
        MessagesPlaceholder(variable_name="conversation_history"),
        ("user", "{query}"),
    ])
    
    combine_documents_chain = create_stuff_documents_chain(llm_model, prompt_template)
    return create_retrieval_chain(retriever_chain, combine_documents_chain)

def generate_bot_response(user_query):
    retriever_chain = initialize_retriever_chain(st.session_state.doc_vector_store)
    conversational_chain = setup_conversational_chain(retriever_chain)
    
    response = conversational_chain.invoke({
        "conversation_history": st.session_state.conversation_log,
        "query": user_query
    })
    
    return response['answer']

# Streamlit app configuration
st.set_page_config(page_title="Website Chat Assistant", page_icon="🤖")
st.title("Chat with a Website")

# Sidebar for input settings
with st.sidebar:
    st.header("Settings")
    input_website_url = st.text_input("Enter Website URL")

# Ensure URL is provided
if not input_website_url:
    st.info("Please enter a website URL to start chatting.")
else:
    # Initialize session state variables
    if "conversation_log" not in st.session_state:
        st.session_state.conversation_log = [
            AIMessage(content="Hello! How can I assist you?")
        ]
    if "doc_vector_store" not in st.session_state:
        st.session_state.doc_vector_store = create_vector_store_from_url(input_website_url)

    # User query input
    user_input = st.chat_input("Type your message...")
    if user_input:
        bot_reply = generate_bot_response(user_input)
        st.session_state.conversation_log.append(HumanMessage(content=user_input))
        st.session_state.conversation_log.append(AIMessage(content=bot_reply))

    # Display chat conversation
    for chat_message in st.session_state.conversation_log:
        if isinstance(chat_message, AIMessage):
            with st.chat_message("AI"):
                st.write(chat_message.content)
        elif isinstance(chat_message, HumanMessage):
            with st.chat_message("User"):
                st.write(chat_message.content)
