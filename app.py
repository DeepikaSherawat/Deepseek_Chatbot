import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    ChatMessagePromptTemplate)

st.markdown("""
<style>
    /* Your styles here */
</style>
""", unsafe_allow_html=True)

st.title("DeeSeek chatbot")
st.caption("DeeSeek chatbot can chat with you in multiple languages, powered by the LangChain Ollama model.")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    select_model = st.selectbox("Select model", ["deepseek-r1:1.5b", "deepseek-r1:3b", "deepseek-r1:6b"], index=0)

    st.markdown("### Chatbot settings")
    max_tokens = st.slider("Max tokens", 50, 500, 100)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.5)
    st.markdown("""
                - Data Science Expert
                - Python Expert
                - Debugging Assistant
                - Solutions to Problems
                """)
    st.markdown("Build with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

# Initialize the chatbot
llm_engine = ChatOllama(model=select_model,
                        base_url="http://localhost:11434", 
                        max_tokens=max_tokens, 
                        temperature=temperature
                        )

# System message prompt
system_message_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions."
    "with strategic print statements for debugging. Always respond in English."
)

# Session State management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hello, I am DeeSeek. How can I help you today?"}]

# Chat container
chat_container = st.container()

# Display Chat messages
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# User input
user_input = st.chat_input("Your message")

def generate_ai_response(prompt_chain):
    processing_pipeline=prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain():
    prompt_sequence = [system_message_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_input:
    # Add user message to log
    st.session_state.message_log.append({"role": "user", "content": user_input})
    
    # Generate AI response
    with st.spinner("🧠 Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
    # Add AI response to log
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    
    # Rerun to update chat display
    st.rerun()