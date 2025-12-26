# RAG TriviaQA System

A **Retrieval-Augmented Generation (RAG)** system for question answering using **vector embeddings** and **LLM integration**. This project enables users to query a dataset (TriviaQA) and receive accurate answers along with the most relevant context.

---

## Features

- Text preprocessing and chunking (300–800 tokens per chunk)  
- Document embeddings using **SentenceTransformers**  
- Vector search with **FAISS**  
- Answer generation via **Ollama** or **FLAN-T5**  
- FastAPI REST API with `/query` endpoint  
- Evaluation support for:
  - Context retrieval accuracy  
  - Answer correctness (Correct / Partially Correct / Incorrect)  
  - Response latency  
- Dockerized for seamless deployment  
- Optional Colab + ngrok support for public demo 

---

## 📂 Project Structure
RAG-TriviaQA/
├── main.py # FastAPI entrypoint
├── config.py # Configuration parameters
├── document_processor.py # Data preprocessing & chunking
├── retriever.py # Vector retrieval logic
├── llm_handler.py # LLM integration (Ollama / HuggingFace)
├── rag_pipeline.py # Orchestration of RAG queries
├── evaluation.py # Evaluation script & result table generation
├── requirements.txt # Python dependencies
├── Dockerfile # Docker container setup
├── docker-compose.yml # Optional Docker Compose
└── README.md # Project documentation



## Architecture
User Query → Embedding → FAISS Search → Top-K Contexts → LLM → Answer


---

## Quick Start

### Option 1: Docker (Recommended)
```bash
# Build and run the container
docker-compose up --build

# API available at: http://localhost:8000

# 1. Install dependencies
pip install -r requirements.txt

# 2. Install Ollama (for local LLM inference)
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen2.5:0.5b

# 3. Run FastAPI locally
uvicorn main:app --reload

# 4. Run evaluation script
python evaluation.py

{
  "question": "What is the capital of France?",
  "top_k": 3
}

## Docker Usage
# Build the Docker image
docker build -t rag-triviaqa .

# Run container locally
docker run -p 8000:8000 rag-triviaqa

# Access Swagger UI at: http://localhost:8000/docs

## Colab Demo with Gradio + FastAPI + ngrok

You can run the RAG system directly in Google Colab with a public URL using **ngrok**. This allows you to interact with both a **FastAPI backend** and a **Gradio interface**.

---

### Install Dependencies

```python
!pip install -q fastapi uvicorn pyngrok gradio sentence-transformers faiss-cpu
notebook link: [https://colab.research.google.com/drive/1GCFALUcsXouv992LZZp8PkDNl4DMDR3i?usp=sharing]
