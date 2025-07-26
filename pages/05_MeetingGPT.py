import streamlit as st
import subprocess
import math
from pydub import AudioSegment
import glob
import openai
import os
from openai import OpenAI     # openai‑python ≥ 1.0.0
import glob, os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()   

has_transcript = os.path.exists("./.cache/transcript.txt")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4.1-nano",
)

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=100,
)


@st.cache_resource
def embed_file(file_path):
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=100,
    )
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    st.warning("⚠️ 임베딩 기능이 비활성화되었습니다.")
    return None


@st.cache_data()
def transcribe_chunks(chunk_folder: str, destination_folder: str) -> str:
    if has_transcript:
        return
    # 1) mp3 파일 목록을 정렬(숫자순)해서 가져오기 ------------------------------
    files = sorted(
        glob.glob(f"{chunk_folder}/*.mp3"),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split('_')[-1])
    )

    # 2) 각 chunk 전사 --------------------------------------------------------
    transcript_parts = []
    for path in files:
        with open(path, "rb") as f:
            text = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="ko",        # 한국어
                response_format="text"  # 순수 문자열
            )
            transcript_parts.append(text)  # 그대로 문자열이므로 바로 추가

    final_text = "\n".join(transcript_parts)

    # 3) 출력 폴더 준비 후 저장 -----------------------------------------------
    os.makedirs(destination_folder, exist_ok=True)
    out_path = os.path.join(destination_folder, "transcript.txt")
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(final_text)

    return final_text


@st.cache_data()
def extract_audio_from_video(video_path):
    if has_transcript:
        return
    audio_path = video_path.replace("mp4", "mp3")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        audio_path,
    ]
    subprocess.run(command)


@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    if has_transcript:
        return
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        chunk = track[start_time:end_time]
        chunk.export(
            f"./{chunks_folder}/chunk_{i}.mp3",
            format="mp3",
        )


st.set_page_config(
    page_title="MeetingGPT",
    page_icon="💼",
)

st.markdown(
    """
# MeetingGPT
            
Welcome to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any questions about it.

Get started by uploading a video file in the sidebar.
"""
)

with st.sidebar:
    video = st.file_uploader(
        "Video",
        type=["mp4", "avi", "mkv", "mov"],
    )

if video:
    chunks_folder = "./.cache/chunks"
    transcript_dir = "./.cache/transcripts"
    transcript_path = os.path.join(transcript_dir, "transcript.txt")
    with st.status("Loading video...") as status:
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        audio_path = video_path.replace("mp4", "mp3")
        with open(video_path, "wb") as f:
            f.write(video_content)
        status.update(label="Extracting audio...")
        extract_audio_from_video(video_path)
        status.update(label="Cutting audio segments...")
        cut_audio_in_chunks(audio_path, 10, chunks_folder)
        status.update(label="Transcribing audio...")
        transcribe_chunks(chunks_folder, transcript_dir)
    transcript_tab, summary_tab, qa_tab = st.tabs(
        [
            "Transcript",
            "Summary",
            "Q&A",
        ]
    )

    with transcript_tab:
        with open(transcript_path, "r") as file:
            st.write(file.read())
    with summary_tab:
        start = st.button("Generate summary")

        if start:
            loader = TextLoader(transcript_path)
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=800,
                chunk_overlap=100,
            )
            docs = loader.load_and_split(text_splitter=splitter)

            first_summary_prompt = ChatPromptTemplate.from_template(
                """
                Write a concise summary of the following:
                "{text}"
                CONCISE SUMMARY:                
            """
            )

            first_summary_chain = first_summary_prompt | llm | StrOutputParser()

            summary = first_summary_chain.invoke(
                {"text": docs[0].page_content},
            )

            refine_prompt = ChatPromptTemplate.from_template(
                """
                Your job is to produce a final summary.
                We have provided an existing summary up to a certain point: {existing_summary}
                We have the opportunity to refine the existing summary (only if needed) with some more context below.
                ------------
                {context}
                ------------
                Given the new context, refine the original summary.
                If the context isn't useful, RETURN the original summary.
                """
            )

            refine_chain = refine_prompt | llm | StrOutputParser()

            with st.status("Summarizing...") as status:
                for i, doc in enumerate(docs[1:]):
                    status.update(label=f"Processing document {i+1}/{len(docs)-1} ")
                    summary = refine_chain.invoke(
                        {
                            "existing_summary": summary,
                            "context": doc.page_content,
                        }
                    )
                    st.write(summary)
            st.write(summary)

        with qa_tab:
            retriever = embed_file(transcript_path)

            docs = retriever.invoke("do they talk about marcus aurelius?")

            st.write(docs)
