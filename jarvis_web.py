import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from vector import retriever  # from your existing code
from typing import Dict

# Setup model and prompt
model = OllamaLLM(model="llama3")
template = """
You are Jarvis, a smart assistant that answers based on user data and reviews.

Relevant info:
{reviews}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain: RunnableSerializable[Dict[str, str], str] = prompt | model

# Streamlit UI
st.set_page_config(page_title="Jarvis AI", layout="centered")
st.title("🤖 Jarvis — Your Local AI Assistant")

question = st.text_input("Ask Jarvis something...")

if question:
    with st.spinner("Thinking..."):
        docs = retriever.invoke(question)
        reviews = "\n".join([doc.page_content for doc in docs])
        response = chain.invoke({"reviews": reviews, "question": question})
        st.success(response)
