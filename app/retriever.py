from typing import List, Dict
from app.embeddings import EmbeddingModel
from app.vector_store import VectorStore


class Retriever:
    def __init__(self, embedder: EmbeddingModel, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve top-k most relevant chunks for a user query.
        """
        query_embedding = self.embedder.encode_query(query)
        results = self.store.search(query_embedding, top_k=top_k)
        return results