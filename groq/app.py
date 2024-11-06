import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()

# Loading Groq API
groqApi = os.environ["GROQ_API_KEY"]

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="llama3.1")
    st.session_state.loader = WebBaseLoader(
        "https://www.nationalchurchillmuseum.org/be-ye-men-of-valour.html"
    )
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.textSplitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    st.session_state.finalDocument = st.session_state.textSplitter.split_documents(
        st.session_state.docs
    )
    st.session_state.vector = FAISS.from_documents(
        st.session_state.finalDocument, st.session_state.embeddings
    )

st.title("Sir Winston Churchill Demo")
llm = ChatGroq(groq_api_key=groqApi, model="llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question asked by the user only in the provided context.
    Think step by step before answering questions and provide a detailed answer.
    I will tip you $1000 if the user finds the answer helpful
    Character: Sir Winston Churchill, former Prime Minister of the United Kingdom
    Task: Respond to the user's query in a manner that is characteristic of Winston Churchill's speaking style, drawing upon his vast knowledge and unique perspective. 
    Guidelines:
    Formal and Eloquent: Employ a formal and eloquent tone, using complex sentence structures and a wide vocabulary.
    Strong Opinions: Express strong opinions and convictions, often with a sense of urgency and determination.
    Historical References: Reference historical events and figures to support your arguments and illustrate your points.
    Rhetorical Devices: Utilize rhetorical devices such as metaphors, similes, and alliteration to enhance the impact of your message.
    British English: Adhere to British English grammar and spelling conventions.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

documentChain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vector.as_retriever()
retrievalChain = create_retrieval_chain(retriever, documentChain)

prompt = st.text_input("Ask Sir Winston Churchill anything")

if prompt:
    response = retrievalChain.invoke({"input": prompt})
    st.write(response["answer"])
