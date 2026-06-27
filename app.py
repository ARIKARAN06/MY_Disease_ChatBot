import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from vectorize_data import Vectorize_document
from Predict import find_disease
from RAG_utility import (setup_vectorstore,
                         Create_ChatBot)

st.set_page_config(
    page_title="Disease-ChatBot",
    page_icon="🏥",
    layout="centered"
)


st.title("🏥 AI Dermatology Assistant")

st.subheader(
    "Deep Learning-Based Skin Disease Classification with (RAG) and Groq Llama 3.3"
)

st.divider()



if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "disease" not in st.session_state:
    st.session_state.disease = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = Create_ChatBot(st.session_state.vectorstore)


input_image = st.file_uploader("Upload a image..",type=['jpg','jpeg','img'])

if input_image is not None:
    st.image(input_image)

    disease = find_disease(input_image)

    st.session_state.disease = disease

    st.chat_message('assistant').write(
        f"Predicted Disease: **{disease}**\n\nYou can ask me question about this disease."
    )

for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


user_question = st.chat_input("Ask anything about the disease?")

if user_question:
    st.session_state.chat_history.append({"role": "user", "content": user_question})

    with st.chat_message('user'):
        st.markdown(user_question)


    with st.chat_message('assistant'):
        query = f"""
        Predicted Disease: {st.session_state.disease}

        User Question:
        {user_question}
        """
        response = st.session_state.conversational_chain.invoke({"question":query})
        assistant_response = response["answer"]
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
        st.markdown(assistant_response)
