import streamlit as st
import os

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

@st.cache_resource
def init_bot():

    import os

    CORPUS_PATH = "hr_docs"

    if not os.path.exists(CORPUS_PATH):
        st.error(
            f"Folder '{CORPUS_PATH}' not found.\n\n"
            "Create an hr_docs folder in your GitHub repository and upload your HR PDF files."
        )
        st.stop()

    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    st.sidebar.success(f"Documents Loaded: {len(documents)}")

    if len(documents) == 0:
        st.error(
            "No PDF documents found in hr_docs folder."
        )
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    st.sidebar.success(f"Chunks Created: {len(chunks)}")

    if len(chunks) == 0:
        st.error(
            "Documents loaded but no chunks were created."
        )
        st.stop()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 30,
            "lambda_mult": 0.5
        }
    )

    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0,
        max_tokens=1024
    )

    return retriever, llm
