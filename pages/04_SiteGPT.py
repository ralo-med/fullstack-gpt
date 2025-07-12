import streamlit as st
from langchain.document_loaders import SitemapLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser, output_parser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import ssl
import urllib3
import certifi


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🤔",
    layout="wide",
)

html2text_transformer = Html2TextTransformer()

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    elif footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ").replace("\xa0", " ")

@st.cache_data(show_spinner="Loading sitemap...", ttl=3600)
def load_sitemap(url):
    import urllib3
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # SSL 검증을 완전히 비활성화하는 세션 생성
    session = requests.Session()
    session.verify = False
    
    # 어댑터 설정
    adapter = HTTPAdapter()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=200)

    
    loader = SitemapLoader(
        url,
        filter_urls=[
          r"^(.*\/blog\/).*"
        ],
        parsing_function=parse_page,
        requests_kwargs={"verify": False},
        session=session
    )
    loader.requests_per_second = 20  # 더 빠른 속도
    docs = loader.load()
    split_docs = text_splitter.split_documents(docs)
    return split_docs

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
    if url.endswith(".xml"):
        if st.button("🔄 다시 로드", key="reload_button"):
            st.cache_data.clear()
        
        try:
            docs = load_sitemap(url)
            st.write(f"📄 로드된 문서: {len(docs)}개")
            st.write(docs)
        except Exception as e:
            st.error(f"❌ 사이트맵 로딩 중 오류가 발생했습니다: {str(e)}")
    else:
        with st.sidebar:
            st.error("Please enter a Sitemap URL.")





