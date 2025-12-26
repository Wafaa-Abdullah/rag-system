from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import RAGPipeline
from retriever import Retriever
from llm_handler import LLMHandler
from document_processor import DocumentProcessor

# Initialize
processor = DocumentProcessor()
processor.load_and_process_dataset()
embeddings = processor.create_embeddings()
processor.build_faiss_index(embeddings)
retriever = Retriever(processor.chunks, processor.index)
llm_handler = LLMHandler()
rag_pipeline = RAGPipeline(retriever, llm_handler)

app = FastAPI(title="RAG TriviaQA API")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

@app.post("/query")
def query_rag(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    return rag_pipeline.query(req.question, req.top_k)
