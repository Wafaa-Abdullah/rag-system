import time
from retriever import Retriever
from llm_handler import LLMHandler
from typing import Dict, List

class RAGPipeline:
    def __init__(self, retriever: Retriever, llm_handler: LLMHandler):
        self.retriever = retriever
        self.llm_handler = llm_handler
        self.query_history = []

    def query(self, question: str, top_k: int = 3) -> Dict:
        start = time.time()
        contexts, scores = self.retriever.retrieve(question, top_k)
        answer = self.llm_handler.generate_answer(question, contexts)
        latency = int((time.time() - start) * 1000)
        result = {"question": question, "answer": answer, "contexts": contexts, "scores": scores, "latency_ms": latency}
        self.query_history.append(result)
        return result
