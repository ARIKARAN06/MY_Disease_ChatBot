import os
import json
import warnings
warnings.filterwarnings('ignore')

from vectorize_data import Vectorize_document
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain

working_dir = os.path.dirname(os.path.abspath(__file__))

config_path = f"{working_dir}/config.json"
config_data = json.load(open(config_path))

GROQ_API_KEY = config_data["GROQ_API_KEY"]

os.environ["GROQ_API_KEY"]  = GROQ_API_KEY

def setup_vectorstore():
    Vectorize_document()
    persist_directory = f"{working_dir}/Vector_DB_dir"
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma(persist_directory = persist_directory,
                         embedding_function=embeddings)

    return vectorstore

def Create_ChatBot(vectorstore):

    llm = ChatGroq(model="llama-3.3-70b-versatile",
                   temperature=0)

    retriever = vectorstore.as_retriever(
        search_search_kwargs={"k": 2}
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever = retriever,
        chain_type = "stuff",
        memory = memory,
        verbose=True,
        return_source_documents = True
    )

    return chain