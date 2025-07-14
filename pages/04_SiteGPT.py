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
from langchain.schema.runnable import RunnablePassthrough
from langchain.embeddings import OpenAIEmbeddings
import os
import ssl
import urllib3
import certifi
from langchain.schema.runnable import RunnableLambda

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.1)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5 (ex:4.7).

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 4.8
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0.0
                                                  
    Your turn!

    Question: {question}
"""
)

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

def get_answer(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answer_chain = answers_prompt|llm
    answer = []
    for doc in docs:
        result = answer_chain.invoke({"context":doc.page_content, "question":question})
        answer.append(result)
    return answer

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
    vectorstore = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    return vectorstore.as_retriever()

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
            retriever = load_sitemap(url)
            chain = {"docs":retriever, "question":RunnablePassthrough()}|RunnableLambda(get_answer)
            st.write(f"📄 로드된 문서: {len(retriever)}개")
            st.write(retriever)
        except Exception as e:
            st.error(f"❌ 사이트맵 로딩 중 오류가 발생했습니다: {str(e)}")
    else:
        with st.sidebar:
            st.error("Please enter a Sitemap URL.")





