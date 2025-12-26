from dataclasses import dataclass
import os

@dataclass
class Config:
    DATASET_SIZE: int = 1000
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 50
    TOP_K_RETRIEVAL: int = 3
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_BACKEND: str = "ollama"
    OLLAMA_MODEL: str = "qwen2.5:0.5b"
    OLLAMA_URL: str = "http://localhost:11434"
    HF_MODEL: str = "google/flan-t5-small"
    MAX_LENGTH: int = 512
    TEMPERATURE: float = 0.7
    DATA_DIR: str = "./data"
    FAISS_INDEX_PATH: str = "./data/faiss_index"
    CHUNKS_PATH: str = "./data/chunks.json"

os.makedirs(Config.DATA_DIR, exist_ok=True)
