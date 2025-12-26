RAG TriviaQA System

This project implements a Retrieval-Augmented Generation (RAG) system combining vector embeddings with LLM integration. Users can ask questions and get accurate answers along with the relevant context. The system supports both local LLMs (Ollama/FLAN-T5) and FastAPI deployment with Docker or Colab + ngrok demo.

Features

Text preprocessing and chunking (300–800 tokens per chunk)

Document embeddings using SentenceTransformers

Vector-based document retrieval via FAISS

Answer generation with Ollama or HuggingFace FLAN-T5

FastAPI REST API exposing /query endpoint

Evaluation of system performance (accuracy, latency, retrieved context)

Dockerized for easy deployment

Colab + ngrok support for public demo

Optional: query reranking, custom logging, architecture tuning

🗂️ Project Structure
RAG-TriviaQA/
├── api.py                 # FastAPI application
├── config.py              # Configuration parameters
├── document_processor.py  # Text preprocessing & chunking
├── retriever.py           # Vector retrieval
├── llm_handler.py         # LLM integration
├── rag_pipeline.py        # RAG query pipeline
├── evaluation.py          # Evaluation scripts & summary table
├── requirements.txt       # Python dependencies
├── Dockerfile             # Containerization
└── README.md              # This file

Installation
Clone repository
git clone https://github.com/<username>/RAG-TriviaQA.git
cd RAG-TriviaQA

Install dependencies
pip install -r requirements.txt

Running FastAPI Locally
uvicorn api:app --host 0.0.0.0 --port 8000


Example POST request:

POST /query
{
  "question": "What is the capital of France?",
  "top_k": 3
}


Example Response:

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

Running on Colab with ngrok
!pip install pyngrok

from pyngrok import ngrok

# Add your ngrok auth token
!ngrok authtoken <YOUR_NGROK_AUTH_TOKEN>

# Open tunnel for FastAPI
public_url = ngrok.connect(8000)
print("Public URL:", public_url)

# Run FastAPI
!uvicorn api:app --host 0.0.0.0 --port 8000


This allows external access to your FastAPI service via a public URL.

Ideal for quick demos and sharing with collaborators.

Evaluation

Use evaluation.py to test multiple questions:

Measures:

Context retrieval accuracy

Answer correctness: Correct / Partially Correct / Incorrect

Response latency

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


Access Swagger UI at: http://localhost:8000/docs

Configuration

Modify config.py for:

Dataset size

Chunk size & overlap

LLM backend: Ollama / HuggingFace

Embedding model

Retrieval top_k

Generation parameters: temperature, max_length

Colab Notebook (Gradio Demo)

A Colab notebook is provided to quickly test the RAG system with a Gradio interface:

Interactively ask questions

View retrieved contexts and generated answers

Monitor latency and query history

Steps:

Open the notebook: [https://colab.research.google.com/drive/1GCFALUcsXouv992LZZp8PkDNl4DMDR3i?usp=sharing]

Run all cells to initialize the system (dataset, embeddings, FAISS, LLM, RAG pipeline)

Use Gradio UI to input questions

Optional: Adjust top_k for context retrieval
