import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from typing import List, Dict
from tqdm.auto import tqdm
from config import Config

class DocumentProcessor:
    def __init__(self):
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.chunks = []
        self.index = None

    def load_and_process_dataset(self):
        dataset = load_dataset("trivia_qa", "unfiltered.nocontext", split="train")
        dataset = dataset.select(range(min(Config.DATASET_SIZE, len(dataset))))

        for idx, item in enumerate(tqdm(dataset, desc="Processing")):
            question = item['question']
            answer = item['answer']['value']
            document = f"Question: {question} Answer: {answer}"
            if len(document.strip()) < 20:
                continue
            self.chunks.extend(self._chunk_text(document, idx))
        return self.chunks

    def _chunk_text(self, text: str, doc_id: int) -> List[Dict]:
        words = text.split()
        chunks = []
        chunk_size_words = Config.CHUNK_SIZE // 4
        overlap_words = Config.CHUNK_OVERLAP // 4
        for i in range(0, len(words), chunk_size_words - overlap_words):
            chunk_words = words[i:i + chunk_size_words]
            if len(chunk_words) < 10:
                continue
            chunks.append({"text": " ".join(chunk_words), "doc_id": doc_id, "chunk_id": f"{doc_id}_{i}"})
        return chunks

    def create_embeddings(self):
        texts = [c['text'] for c in self.chunks]
        return self.embedding_model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)

    def build_faiss_index(self, embeddings: np.ndarray):
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        return self.index

    def save_artifacts(self):
        with open(Config.CHUNKS_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False)
        faiss.write_index(self.index, Config.FAISS_INDEX_PATH)

    def load_artifacts(self):
        with open(Config.CHUNKS_PATH, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        self.index = faiss.read_index(Config.FAISS_INDEX_PATH)
        return self.chunks, self.index
