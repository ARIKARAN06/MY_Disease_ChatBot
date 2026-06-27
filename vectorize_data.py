import os
import json

from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain

working_dir = os.path.dirname(os.path.abspath(__file__))

def Vectorize_document():

    embeddings = HuggingFaceEmbeddings()

    loader = DirectoryLoader(
        "Disease_data",
        glob="./*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,
                                                   chunk_overlap=500)

    text_chunks = text_splitter.split_documents(documents)

    vector_db = Chroma.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        persist_directory="Vector_DB_dir"
    )

    print("Documents vectorized")






