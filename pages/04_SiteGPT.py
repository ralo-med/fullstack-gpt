import streamlit as st
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser, output_parser
import os

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🤔",
    layout="wide",
)

html2text_transformer = Html2TextTransformer()

st.title("SiteGPT")
st.write("Ask questions about the content of a website.")
st.write("Enter the URL of the website you want to analyze.")\

with st.sidebar:
    url = st.text_input("Enter the URL.",placeholder="https://www.google.com")
    if st.button("Analyze"):
        st.write("Analyzing the website...")
        st.write("Website URL: ", url)
        st.write("Website content: ", url)
        


if url:
    loader = AsyncChromiumLoader([url])
    docs = loader.load()
    transformed_docs = html2text_transformer.transform_documents(docs)
    #





