import os
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Safely load the API key from your environment variables
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
  raise ValueError("GEMINI_API_KEY is missing from the .env file!")

os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY


def initialize_rag_system(pdf_path: str):
  # 2. Load the document
  if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found at {pdf_path}")

  loader = PyPDFLoader(pdf_path)
  docs = loader.load()

  # 3. Text chunking
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000, chunk_overlap=200
  )
  chunks = text_splitter.split_documents(docs)

  # 4. Vector embeddings & Vector Store (ChromaDB)
  embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
  vector_store = Chroma.from_documents(chunks, embeddings)

  # 5. Setup Retriever
  retriever = vector_store.as_retriever(
      search_type="similarity", search_kwargs={"k": 3}
  )

  # 6. LLM setup & RAG Chain prompt configuration
  llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2)

  system_prompt = (
      "You are an intelligent domain-specific research assistant. "
      "Use the following retrieved context pieces to answer "
      "the question. If you don't know the answer, say that you "
      "cannot find the answer in the provided documents.\n\n"
      "{context}"
  )

  prompt = ChatPromptTemplate.from_messages([
      ("system", system_prompt),
      ("human", "{input}"),
  ])

  question_answer_chain = create_stuff_documents_chain(llm, prompt)
  rag_chain = create_retrieval_chain(retriever, question_answer_chain)

  return rag_chain


if __name__ == "__main__":
  # Example execution
  pdf_file = "dataset/sample_policy.pdf"
  print("Initializing RAG pipeline...")
  chain = initialize_rag_system(pdf_file)

  query = "What are the core regulations outlined in the document?"
  print(f"\nQuery: {query}")
  response = chain.invoke({"input": query})
  print(f"\nAnswer:\n{response['answer']}")