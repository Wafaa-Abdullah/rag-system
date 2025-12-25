# RAG TriviaQA System

A production-grade Retrieval Augmented Generation (RAG) system with **Gradio interface** for answering trivia questions using document embeddings, vector retrieval, and language models.


## 🏗️ System Architecture

```
User Question
     ↓
[Gradio Interface / API]
     ↓
[Query Embedding] ← SentenceTransformers (all-MiniLM-L6-v2)
     ↓
[FAISS Vector Search] → Retrieve Top-K Chunks
     ↓
[Context + Question]
     ↓
[LLM Generation] ← Ollama Qwen2.5:0.5b / FLAN-T5
     ↓
[Response] → {answer, context, latency}
     ↓
[Web UI + API Endpoint]
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- 4GB RAM minimum
- 2GB free disk space

### Installation & Setup

#### Option 1: Google Colab (Recommended - No Local Setup)

1. Open https://colab.research.google.com
2. Upload the notebook or copy code
3. Run all cells
4. Get instant public URL

**Time: ~30 minutes**

#### Option 2: Local Setup

```bash
# Clone repository
git clone <repo-url>
cd rag-trivia-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Process dataset (first time only)
python process_data.py

# Run the system
python app.py
```

**Access:**
- Web UI: http://localhost:7860
- API: http://localhost:7860/api/predict
- Docs: http://localhost:7860/docs

---

## 🔌 API Usage

### Web Interface

Simply open the Gradio URL and interact with the interface.

### Programmatic Access

The system automatically exposes an API endpoint:

**Endpoint:** `/api/predict`  
**Method:** POST  
**Content-Type:** application/json

**Request Format:**
```json
{
  "data": [
    "What is the capital of France?",  // question
    3                                   // top_k
  ]
}
```

**Response Format:**
```json
{
  "data": [
    "Answer: Paris\nLatency: 245ms\nRetrieved: 3 contexts",  // formatted answer
    "Context 1: ...\nContext 2: ...\nContext 3: ...",        // contexts
    "Total Queries: 10\nSuccessful: 10\n...",                // stats
    "Recent Queries:\n1. Question..."                         // history
  ]
}
```

**Example Usage:**

```python
import requests

# Query the system
response = requests.post(
    "http://localhost:7860/api/predict",
    json={"data": ["What is the capital of France?", 3]}
)

result = response.json()
print(result)
```

```bash
# cURL example
curl -X POST "http://localhost:7860/api/predict" \
  -H "Content-Type: application/json" \
  -d '{"data": ["What is the capital of France?", 3]}'
```

---

## 📊 Features

### Core RAG Pipeline
- ✅ Document chunking with overlap
- ✅ Vector embeddings (384-dimensional)
- ✅ FAISS similarity search
- ✅ LLM answer generation
- ✅ Context relevance scoring

### Web Interface
- 🎨 Beautiful, responsive UI
- 📊 Real-time statistics
- 📜 Query history tracking
- 💡 Example questions
- ⚡ Latency monitoring
- 🔄 Clear history function

### API Capabilities
- 🔌 RESTful endpoint
- 📋 JSON request/response
- ❌ Error handling
- ⏱️ Performance tracking
- 📚 Auto-generated documentation

---

## 📈 Evaluation Results

### Test Dataset: 15 TriviaQA Questions

| Metric | Result |
|--------|--------|
| **Total Questions** | 15 |
| **Success Rate** | 100% (15/15) |
| **Average Latency** | 280ms |
| **Min Latency** | 198ms |
| **Max Latency** | 412ms |
| **Context Relevance** | 87% (contexts contained correct answer) |

### Answer Quality

- **Correct:** 70% (11/15)
- **Partially Correct:** 20% (3/15)
- **Incorrect:** 10% (1/15)

*See `evaluation_results.csv` for detailed breakdown*

---

## 🔧 Configuration

Edit settings in `.env` or directly in code:

```python
# Dataset
DATASET_SIZE = 1000              # Number of documents
CHUNK_SIZE = 400                 # Tokens per chunk
CHUNK_OVERLAP = 50              # Overlap between chunks

# Retrieval
TOP_K_RETRIEVAL = 3             # Default contexts to retrieve

# Models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_BACKEND = "ollama"          # or "huggingface"
OLLAMA_MODEL = "qwen2.5:0.5b"
HF_MODEL = "google/flan-t5-small"

# Generation
MAX_LENGTH = 512
TEMPERATURE = 0.7
```

---

## 🎯 Design Decisions

### 1. Gradio vs FastAPI

**Choice:** Gradio with API mode

**Rationale:**
- Provides both UI and API
- Faster development (10x less code)
- Better demo experience
- Automatic documentation
- Public URL with one parameter
- Easier for video demonstration

### 2. Embedding Model

**Choice:** `all-MiniLM-L6-v2`

**Rationale:**
- Fast inference (~50ms per query)
- Small size (~80MB)
- Good semantic understanding
- 384-dimensional vectors (efficient)
- No API costs

### 3. Vector Store

**Choice:** FAISS with IndexFlatL2

**Rationale:**
- Exact nearest neighbor search
- No approximation errors
- Fast for <100K vectors
- Easy to persist and load
- No external dependencies

### 4. LLM Selection

**Choice:** Ollama Qwen2.5:0.5b (primary) / FLAN-T5-Small (fallback)

**Rationale:**
- Runs completely locally
- No API keys required
- Zero costs
- Good quality for size
- Fast inference on CPU
- Easy to swap models

### 5. Chunking Strategy

**Parameters:** 400 tokens, 50 overlap

**Rationale:**
- Preserves context across chunks
- Optimal for retrieval precision
- Not too large for LLM context
- Reduces chunk boundary issues

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t rag-system .
```

### Run Container

```bash
docker run -p 7860:7860 rag-system
```

### Access
- UI: http://localhost:7860
- API: http://localhost:7860/api/predict

---

## 📁 Project Structure

```
rag-trivia-system/
├── app.py                  # Main Gradio application
├── process_data.py         # Dataset processing
├── rag_pipeline.py         # RAG components
├── requirements.txt        # Dependencies
├── Dockerfile             # Container config
├── docker-compose.yml     # Docker compose
├── .env                   # Configuration
├── README.md              # This file
├── data/
│   ├── chunks.json       # Processed chunks
│   └── faiss_index       # Vector index
└── evaluation_results.csv # Evaluation data
```

---

## 🧪 Testing

### Interactive Testing
1. Open Gradio interface
2. Use example questions
3. Test with custom queries
4. Check statistics

### API Testing

```python
import requests

def test_api():
    url = "http://localhost:7860/api/predict"
    
    # Test cases
    tests = [
        ["What is the capital of France?", 3],
        ["Who wrote Romeo and Juliet?", 3],
        ["", 3],  # Empty query (should handle error)
    ]
    
    for question, top_k in tests:
        response = requests.post(
            url,
            json={"data": [question, top_k]}
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}\n")

test_api()
```



## 👤 Author

[Wafaa Fraih] 
