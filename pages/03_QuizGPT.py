import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser, output_parser
import json

class JsonOutputParser(BaseOutputParser):
    def parse(self, text: str):
        text=text.replace("```json","").replace("```","")
        return json.loads(text)
    
output_parser = JsonOutputParser()

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.1, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Use (o) to signal the correct answer.
         
    Question examples:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
         
    Your turn!
         
    Context: {context}
""",
        )
    ]
)

questions_chain = {"context": format_docs} | questions_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.
     
    You format exam questions into JSON format.
    Answers with (o) are the correct ones.
     
    Example Input:

    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
     
    Example Output:
     
    ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }},
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }},
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }},
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }},
                ]
            }}
        ]
     }}
    ```
    Your turn!

    Questions: {context}

""",
        )
    ]
)

formatting_chain = formatting_prompt | llm

st.set_page_config(
    page_title="QuizGPT",
    page_icon="🤔",
    layout="wide",
)

st.title("QuizGPT")
st.write("This is a quiz application built with Streamlit and OpenAI.")


@st.cache_data(show_spinner="Splitting file..." )
def split_file(file):
    
        if not os.getenv("OPENAI_API_KEY"):
            st.error("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다!")
            return None
        
        os.makedirs("./.cache/quiz_files", exist_ok=True)
        os.makedirs("./.cache/quiz_embeddings", exist_ok=True)
        
        file_content = file.read()
        file_path = f"./.cache/quiz_files/{file.name}"
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        

        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )
        loader = UnstructuredFileLoader(file_path, mode="elements")
        docs = loader.load()
        docs = splitter.split_documents(docs)
        return docs

@st.cache_data(show_spinner="Generating quiz...")
def run_quiz(_docs, topic):
    chain = {"context":questions_chain}|formatting_chain|output_parser
    result = chain.invoke(_docs)
    return result

def run_wikipedia_quiz(topic):
    retrieval = WikipediaRetriever(top_k_results=5, search_kwargs={"srsearch": topic})
    docs = retrieval.get_relevant_documents(topic)
    return docs

with st.sidebar:
    docs = None
    choice = st.selectbox("Choose what you want to use",("File","Wikipedia Article"))

    if choice == "File":
        file = st.file_uploader("Upload a file", type=["pdf","txt","docx"])
        if file:
            st.write("File uploaded successfully")
            docs = split_file(file)

    if choice == "Wikipedia Article":
        topic = st.text_input("Enter a topic")
        if topic:
            docs = run_wikipedia_quiz(topic)



if not docs:
     st.markdown("""Welcome to QuizGQP.           
I will make a quiz from Wikipedia Article.              
Get started by selecting a topic.
                 
""")


else:
 

    

    start = st.button("Generate Quiz")

    if start:
      
        result = run_quiz(docs, topic if topic else file.name)
        st.write(result)







