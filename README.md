# RAG TriviaQA System

This project implements a **Retrieval-Augmented Generation (RAG)** system using **vector embeddings** and **LLM integration**, allowing you to ask questions and get accurate answers with context retrieval.

---

## Features

- Text preprocessing and chunking (300-800 tokens per chunk)
- Document embeddings using **SentenceTransformers**
- Vector search using **FAISS**
- LLM answer generation (**Ollama** / **FLAN-T5**)
- FastAPI REST API with `/query` endpoint
- Evaluation support (accuracy, latency, retrieved context)
- Dockerized for easy deployment
- Colab + ngrok support for public demo
- Optional features: query reranking, cleaner logging, custom chunking

---

## 🗂️ Project Structure
RAG-TriviaQA/
├── api.py # FastAPI application
├── config.py # Configuration parameters
├── document_processor.py # Preprocessing & chunking
├── retriever.py # Vector retrieval
├── llm_handler.py # LLM integration (Ollama / HuggingFace)
├── rag_pipeline.py # Main RAG pipeline
├── evaluation.py # Evaluation script & table generation
├── requirements.txt
├── Dockerfile
└── README.md


---

## Installation

### Clone the repo

```bash
git clone https://github.com/<username>/RAG-TriviaQA.git
cd RAG-TriviaQA

### Install dependencies

pip install -r requirements.txt

Running API locally

uvicorn api:app --host 0.0.0.0 --port 8000

Example request:

POST /query
{
  "question": "What is the capital of France?",
  "top_k": 3
}


Example response:

{
  "question": "What is the capital of France?",
  "answer": "The capital of France is Paris.",
  "contexts": [
    "Question: What is the capital of France? Answer: Paris",
    "Question: France is in which continent? Answer: Europe"
  ],
  "scores": [0.80, 0.65],
  "latency_ms": 2800
}

Running on Colab with ngrok (public URL)
!pip install pyngrok
from pyngrok import ngrok

# Add your ngrok auth token
!ngrok authtoken <YOUR_NGROK_AUTH_TOKEN>

# Open tunnel for FastAPI
public_url = ngrok.connect(8000)
print("Public URL:", public_url)

# Run FastAPI
!uvicorn api:app --host 0.0.0.0 --port 8000

Evaluation

Use evaluation.py to test multiple questions

Measures:

Context retrieval accuracy

Answer correctness (Correct / Partially Correct / Incorrect)

Response latency

Outputs:

DataFrame with all results

Summary text for quick report

Example usage:

from evaluation import Evaluator
from rag_pipeline import rag_pipeline

evaluator = Evaluator(rag_pipeline)

sample_questions = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"}
]

evaluator.evaluate_questions(sample_questions)
df = evaluator.get_results_df()
print(df)
print(evaluator.summary())

Docker Usage
docker build -t rag-triviaqa .
docker run -p 8000:8000 rag-triviaqa


Access Swagger UI: http://localhost:8000/docs

Configuration

Modify config.py for:

Dataset size

Chunk size / overlap

LLM backend (Ollama / HuggingFace)

Embedding model

Retrieval top_k

Generation parameters (temperature, max_length)



