import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangSmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With Ollama"

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's queries clearly."),
    ("user", "Question: {question}")
])

def generate_response(question, model_name, temperature, max_tokens):
    # Initialize Ollama model
    llm = Ollama(model=model_name, temperature=temperature)
    
    # Define output parser
    output_parser = StrOutputParser()
    
    # Build chain
    chain = prompt | llm | output_parser
    
    # Invoke chain
    answer = chain.invoke({'question': question})
    
    # Optionally trim to max_tokens manually (since Ollama doesn’t have that param)
    return answer[:max_tokens]

# Streamlit app
st.title("🧠 Enhanced Q&A Chatbot (Ollama + LangChain)")

# Sidebar
llm_model = st.sidebar.selectbox("Select Open Source Model", ["mistral", "Gemma 3", "Phi 4 Mini"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 500, 150)

# User input
st.write("Go ahead and ask any question below 👇")
user_input = st.text_input("You:")

# Generate response
if user_input:
    with st.spinner("Thinking... 🤔"):
        response = generate_response(user_input, llm_model, temperature, max_tokens)
    st.write("**Assistant:**", response)
else:
    st.write("Please type your question above ☝️")

