import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the RAG initialization function from your existing script
from src.rag_system import initialize_rag_system

st.set_page_config(
    page_title="Domain-Specific RAG Document Assistant", layout="centered"
)

st.title("📄 Domain-Specific Document QA System (RAG)")
st.markdown(
    "Ask questions about your uploaded reference documents"
)

# Sidebar for configuration status
with st.sidebar:
  st.header("Configuration")
  pdf_path = "dataset/sample_policy.pdf"
  if os.path.exists(pdf_path):
    st.success("Target Document Loaded: sample_policy.pdf")
  else:
    st.error("Missing sample_policy.pdf in dataset/")

  api_key = os.getenv("GEMINI_API_KEY")
  if api_key:
    st.success("Gemini API Key detected.")
  else:
    st.error("API Key missing! Check your .env file.")

# Initialize the RAG chain (cached so it doesn't reload on every click)
@st.cache_resource
def load_rag():
  return initialize_rag_system("dataset/sample_policy.pdf")


try:
  rag_chain = load_rag()
except Exception as e:
  st.error(f"Error initializing RAG pipeline: {e}")
  rag_chain = None

# User Query Input
user_query = st.text_input(
    "Enter your question regarding the document:",
    placeholder="e.g., What are the core attendance regulations?",
)

if st.button("Generate Answer") and user_query:
  if rag_chain:
    with st.spinner("Retrieving relevant passages and generating answer..."):
      try:
        response = rag_chain.invoke({"input": user_query})
        st.subheader("Answer:")
        st.write(response["answer"])
      except Exception as e:
        st.error(f"An error occurred during generation: {e}")
  else:
    st.error("RAG pipeline is not initialized properly.")