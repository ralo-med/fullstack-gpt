import streamlit as st
import time

st.set_page_config(
    page_title="Document GPT",
    page_icon="📄",
    layout="wide",
)

st.title("Document GPT")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        st.session_state["messages"].append({"role": role, "message": message})


for message in st.session_state["messages"]:
    send_message(message["message"], message["role"], save=False)




message = st.chat_input("Send a message to the ai")

if message:
    send_message(message, "human")
    time.sleep(1)
    send_message(f"You said: {message}", "ai")





