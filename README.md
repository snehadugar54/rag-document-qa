# Domain-Specific Document Question Answering Using RAG

## 📋 Problem Statement Overview

This repository contains a complete, working implementation for **Problem Statement 9: Domain-Specific Document Question Answering Using RAG** as part of the NLP Practical CEC Challenge.

The system is designed to ingest lengthy documents, policies, or technical manuals, chunk them effectively, build a semantic vector index, and utilize Retrieval-Augmented Generation (RAG) powered by Google's Gemini LLM to answer user queries with precise source context and grounding.

## 🛠️ Tech Stack & Architecture

* **Language:** Python 3.10+

* **Orchestration & Framework:** LangChain / LangChain Google GenAI

* **Vector Database:** ChromaDB (Local vector storage)

* **Embeddings Model:** Google Generative AI Embeddings (`models/embedding-001`)

* **LLM Generation:** Gemini 1.5 Flash

### System Workflow Diagram

1. **Document Ingestion:** Loads domain-specific reference documents (PDF format) via `PyPDFLoader`.

2. **Text Chunking:** Splits large text segments into manageable chunks using `RecursiveCharacterTextSplitter`.

3. **Vector Embeddings:** Converts text chunks into dense vector embeddings and indexes them into ChromaDB.

4. **Semantic Retrieval & Generation:** Retrieves top-$k$ relevant passages matching user queries and passes them to Gemini to formulate grounded answers with references.

## ⚙️ Installation & Setup Instructions

### 1. Clone the Repository

```
git clone https://github.com/your-username/rag-document-qa.git
cd rag-document-qa

```

### 2. Install Dependencies

Ensure you have Python installed, then run:

```
pip install -r requirements.txt

```

### 3. Configure API Key

Set your Google AI Studio API key as an environment variable:

* **Linux/Mac:**

  ```
  export GEMINI_API_KEY="your_api_key_here"
  
  ```

* **Windows (CMD):**

  ```
  set GEMINI_API_KEY="your_api_key_here"
  
  ```

* **Windows (PowerShell):**

  ```
  $env:GEMINI_API_KEY="your_api_key_here"
  
  ```

### 4. Run the Application

Place your target PDF inside the `dataset/` folder (e.g., `dataset/sample_policy.pdf`) and execute the script:

```
python src/rag_system.py

```

## 📂 Project Structure

```
rag-document-qa/
├── dataset/
│   └── sample_policy.pdf      # Target domain documents for querying
├── src/
│   └── rag_system.py          # Core RAG pipeline script (Ingestion, ChromaDB, Gemini)
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation

```

## 