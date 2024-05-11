import streamlit as st
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough

st.title("PDF Chatbot")

uploaded_file = st.file_uploader(
        label='Upload a PDF', type=['pdf', 'PDF'],
        accept_multiple_files=False
    )

if uploaded_file:
    with open(os.path.join("./data/",uploaded_file.name),"wb") as f: 
        f.write(uploaded_file.getbuffer())
    local_path = "./data/" + uploaded_file.name
        
    loader = PyPDFLoader(local_path)
    data = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)

    persist_directory = 'db'
    embeddings = OllamaEmbeddings(model="nomic-embed-text",show_progress=True)

    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        collection_name="local-rag",
        persist_directory=persist_directory
    )

    local_model = "llama3"

    llm = ChatOllama(
            model=local_model,
            streaming=True)
    
    template = """
    ### System:
    You are an respectful and honest assistant. You have to answer the user's questions using only the context provided to you. If you don't know the answer just say you don't know. Don't try to make up an answer.
    ### Context: {context}
    ### Question: {question}
    ### Response: """

    retriever = vector_db.as_retriever()

    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    input = st.text_input('Input your prompt here')
    if input:
        response = chain.stream(input)
        st.write_stream(response)