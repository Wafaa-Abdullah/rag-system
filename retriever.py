from sentence_transformers import SentenceTransformer
from config import Config
from typing import List, Tuple

class Retriever:
    def __init__(self, chunks, index):
        self.chunks = chunks
        self.index = index
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)

    def retrieve(self, query: str, top_k: int = None) -> Tuple[List[str], List[float]]:
        if top_k is None:
            top_k = Config.TOP_K_RETRIEVAL
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)
        contexts, scores = [], []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                contexts.append(self.chunks[idx]['text'])
                scores.append(1 / (1 + dist))
        return contexts, scores
