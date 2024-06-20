import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pandas as pd
import tempfile
import time

# Load environment variables

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model Document Q&A")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

def vector_embedding(file_path):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        documents = []

        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(documents)
        st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
        st.session_state.documents = final_documents

# Specify the file path here

# specific_file_path ="./EDA_report.pdf"
specific_file_path = "./us_census./thinkstats2-21-22_compressed.pdf"

# Call the vector embedding function with the specific file
vector_embedding(specific_file_path)
st.write("Vector Store DB Is Ready")

prompt1 = st.text_input("Enter Your Question From Documents")

if prompt1 and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write(f"Response time: {time.process_time() - start} seconds")
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for doc in response.get("context", []):
            st.write(doc.page_content)
            st.write("--------------------------------")
