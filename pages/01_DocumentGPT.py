import time
import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

st.set_page_config(
    page_title="Document GPT",
    page_icon="📄",
    layout="wide",
)


class ChatCallbackHandler(BaseCallbackHandler):

    def __init__(self):
        self.message = ""
        self.message_box = None

    def on_llm_start(self, *args, **kwargs):
        self.message = ""
        self.message_box = st.empty()
        
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token: str, **kwargs):
        self.message += token
        if self.message_box is not None:
            self.message_box.markdown(self.message)

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.1, streaming=True, callbacks=[ChatCallbackHandler()])


@st.cache_data(show_spinner="Embedding file..." )
def embed_file(file):
    try:
        # OpenAI API 키 확인
        if not os.getenv("OPENAI_API_KEY"):
            st.error("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다!")
            st.info("환경변수를 설정하거나 .env 파일에 API 키를 추가해주세요.")
            return None
        
        # 디렉토리 생성
        os.makedirs("./.cache/files", exist_ok=True)
        os.makedirs("./.cache/embeddings", exist_ok=True)
        
        file_content = file.read()
        file_path = f"./.cache/files/{file.name}"
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
        
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        
        with st.spinner("📄 문서를 로딩하고 분할하는 중..."):
            loader = UnstructuredFileLoader(file_path)
            docs = loader.load_and_split(text_splitter=splitter)
            st.success(f"✅ {len(docs)}개의 청크로 분할 완료!")
        
        with st.spinner("🔤 임베딩을 생성하는 중..."):
            embeddings = OpenAIEmbeddings()
            cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
            vectorstore = FAISS.from_documents(docs, cached_embeddings)
            retriever = vectorstore.as_retriever()
            st.success("✅ 임베딩 생성 완료!")
        
        return retriever
        
    except Exception as e:
        st.error(f"❌ 임베딩 생성 중 오류가 발생했습니다: {str(e)}")
        st.info("다음 사항을 확인해주세요:")
        st.info("1. OPENAI_API_KEY가 올바르게 설정되었는지")
        st.info("2. 인터넷 연결이 안정적인지")
        st.info("3. 업로드한 파일이 손상되지 않았는지")
        return None


st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload a file on the sidebar to get started!
"""
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
)

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def save_message(message, role):
    st.session_state["messages"].append({"role": role, "message": message})

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that can answer questions about the documents you are given. If you don't know the answer, just say that you don't know. Do not make up an answer. Always answer in Korean.
    Context: {context}""",),

    ("user", "{question}"),
])

if file:
    retriever = embed_file(file)
    
    if retriever:
        st.success("🎉 파일이 성공적으로 처리되었습니다!")
        
        send_message("I'm ready to answer your questions!", "ai", save=False)
        paint_history()
        message = st.chat_input("Ask me anything!")
        if message:
            send_message(message, "human")
            chain = {
                "context": retriever|RunnableLambda(lambda x: "\n\n".join([doc.page_content for doc in x])),
                "question": RunnablePassthrough()
            } | prompt | llm 
            with st.chat_message("ai"):
                response = chain.invoke( message)
                
           
    else:
        st.warning("파일 처리를 완료할 수 없습니다. 위의 오류 메시지를 확인해주세요.")

else:
    st.session_state["messages"] = []
