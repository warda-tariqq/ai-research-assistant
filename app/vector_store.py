import faiss
import numpy as np
import pickle
from typing import List, Dict


class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def add(self, embeddings: np.ndarray, chunks: List[Dict]):
        self.index.add(embeddings)
        self.metadata.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 3):
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i in indices[0]:
            if i < len(self.metadata):
                results.append(self.metadata[i])

        return results

    def save(self, index_path: str, metadata_path: str):
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, index_path: str, metadata_path: str):
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)