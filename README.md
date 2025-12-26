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

## Architecture
User Query → Embedding → FAISS Search → Top-K Contexts → LLM → Answer


---

## Quick Start
colab notebook link: [https://colab.research.google.com/drive/1GCFALUcsXouv992LZZp8PkDNl4DMDR3i?usp=sharing]

### Option 1: Docker 
```bash
# Build and run
docker-compose up --build

# API available at: http://localhost:8000
```

### Option 2: Local Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen2.5:0.5b

# 3. Run API
uvicorn main:app --reload

# 4. Run evaluation
python evaluate.py
```

## API Usage

### Query Endpoint
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the capital of France?",
    "top_k": 3
  }'
```

### Response
```json
{
  "question": "What is the capital of France?",
  "answer": "Paris",
  "retrieved_context": ["...", "...", "..."],
  "latency_ms": 234
}
```

## Configuration

Edit `.env` file:
```env
DATASET_SIZE=1000
CHUNK_SIZE=400
TOP_K_RETRIEVAL=3
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OLLAMA_MODEL=qwen2.5:0.5b
```

## API Documentation

Interactive docs: http://localhost:8000/docs
